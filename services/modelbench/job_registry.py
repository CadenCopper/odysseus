"""Single-concurrency asyncio job registry for ModelBench in-dashboard runs.

Backs the `bench_jobs` table (ORM `BenchJob` in core/database.py) with an
in-process asyncio registry that enforces a single active run at a time (the
12GB GPU is single-tenant). A future runs-API card wires HTTP routes on top
of this registry; this module owns only the lifecycle: submit, dispatch,
cancel, and stale-heartbeat reconciliation.
"""

import asyncio
import logging
import uuid

from core import database as core_db
from core.database import BenchJob, utcnow_naive

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = ("queued", "running")
_TERMINAL_STATUSES = ("done", "failed", "cancelled")


class ConcurrentJobError(Exception):
    """Raised by submit() when a bench run is already active.

    The route layer maps this to HTTP 409/423.
    """


class JobCancelled(Exception):
    """Raised by JobContext.check_cancelled() when the job's cancel flag is set."""


def _now():
    """Current naive UTC timestamp, matching core.database's convention."""
    return utcnow_naive()


def _serialize(row):
    """Convert a BenchJob row to a plain dict with isoformat datetimes."""

    def _iso(dt):
        return dt.isoformat() if dt is not None else None

    return {
        "job_id": row.id,
        "id": row.id,
        "run_id": row.run_id,
        "status": row.status,
        "model_tag": row.model_tag,
        "think": row.think,
        "ctx_target": row.ctx_target,
        "prompt": row.prompt,
        "n_samples": row.n_samples,
        "progress": row.progress,
        "message": row.message,
        "error": row.error,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
        "heartbeat_at": _iso(row.heartbeat_at),
    }


class JobContext:
    """Per-job handle passed to the runner coroutine.

    Owns all DB writes for progress/heartbeat/run_id so the runner never
    touches the ORM directly, and exposes a cooperative cancellation check.
    """

    def __init__(self, registry, job_id):
        self._registry = registry
        self._job_id = job_id

    @property
    def job_id(self):
        """The BenchJob.id this context is bound to."""
        return self._job_id

    async def set_progress(self, progress: float, message: str | None = None) -> None:
        """Update progress (clamped to 0..1) and optionally the status message."""
        progress = max(0.0, min(1.0, progress))
        db = core_db.SessionLocal()
        try:
            row = db.query(BenchJob).filter(BenchJob.id == self._job_id).first()
            if row is None:
                return
            row.progress = progress
            if message is not None:
                row.message = message
            row.updated_at = _now()
            db.commit()
        finally:
            db.close()

    async def set_run_id(self, run_id: str) -> None:
        """Attach the samples-run key once the run actually starts producing samples."""
        db = core_db.SessionLocal()
        try:
            row = db.query(BenchJob).filter(BenchJob.id == self._job_id).first()
            if row is None:
                return
            row.run_id = run_id
            row.updated_at = _now()
            db.commit()
        finally:
            db.close()

    async def heartbeat(self) -> None:
        """Bump heartbeat_at to now, resetting the supervisor's stale-job clock."""
        db = core_db.SessionLocal()
        try:
            row = db.query(BenchJob).filter(BenchJob.id == self._job_id).first()
            if row is None:
                return
            row.heartbeat_at = _now()
            db.commit()
        finally:
            db.close()

    def cancel_requested(self) -> bool:
        """Cheap, synchronous poll of the registry's cancel flag for this job."""
        return self._job_id in self._registry._cancelled

    async def check_cancelled(self) -> None:
        """Raise JobCancelled if the job's cancel flag is set.

        Runners must call this between units of work (e.g. before each
        sample) so a cancelled run drains its current unit of work and stops
        without committing a partial result.
        """
        if self.cancel_requested():
            raise JobCancelled()


class BenchJobRegistry:
    """Single-concurrency asyncio registry over the bench_jobs table.

    Only one job may be queued or running at a time. `submit()` raises
    ConcurrentJobError while a job is active. A background supervisor task
    fails jobs whose heartbeat has gone stale (crashed/hung runner).
    """

    def __init__(self, runner=None, *, heartbeat_interval: float = 15.0, stale_timeout: float = 1800.0):
        """runner: optional async callable runner(JobContext) -> None for a full run.

        If None, the job body no-ops straight to done (so the registry is
        testable without a real ModelBench harness wired in).
        """
        self.loop = asyncio.get_event_loop()
        self._runner = runner
        self._heartbeat_interval = heartbeat_interval
        self._stale_timeout = stale_timeout
        self._active_job_id = None
        self._running = False
        self._supervisor_task = None
        self._lock = asyncio.Lock()
        self._cancelled = set()

    def attach_runner(self, runner) -> None:
        """Set (or replace) the async callable used to execute a job's run."""
        self._runner = runner

    async def submit(self, *, model_tag, think=None, ctx_target=None, prompt, n_samples) -> dict:
        """Create a queued BenchJob row and schedule its dispatch.

        Raises ConcurrentJobError if a run is already queued or running.
        """
        async with self._lock:
            if self._running:
                raise ConcurrentJobError(
                    "A bench run is already active; only one run is allowed on the 12GB GPU."
                )
            job_id = uuid.uuid4().hex
            now = _now()
            row = BenchJob(
                id=job_id,
                run_id=None,
                status="queued",
                model_tag=model_tag,
                think=think,
                ctx_target=ctx_target,
                prompt=prompt,
                n_samples=n_samples,
                progress=0.0,
                message="",
                error=None,
                created_at=now,
                updated_at=now,
                heartbeat_at=None,
            )
            db = core_db.SessionLocal()
            try:
                db.add(row)
                db.commit()
                result = _serialize(row)
            finally:
                db.close()
            self._active_job_id = job_id
            self._running = True

        self.loop.create_task(self._dispatch(job_id))
        return result

    async def _dispatch(self, job_id) -> None:
        """Run one job end to end: queued -> running -> a terminal status."""
        async with self._lock:
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                self._running = False
                self._active_job_id = None
                return
            db = core_db.SessionLocal()
            try:
                row = db.query(BenchJob).filter(BenchJob.id == job_id).first()
                if row is None:
                    self._running = False
                    self._active_job_id = None
                    return
                row.status = "running"
                row.updated_at = _now()
                db.commit()
            finally:
                db.close()

        runner = self._runner
        ctx = JobContext(self, job_id)
        try:
            if runner is not None:
                await runner(ctx)
            await self._finish(job_id, "done", message="completed")
        except JobCancelled:
            await self._finish(job_id, "cancelled", message="cancelled by user")
        except asyncio.CancelledError:
            await self._finish(job_id, "cancelled", message="cancelled by user")
        except Exception as e:
            logger.warning(f"bench job {job_id} failed: {e}")
            await self._finish(job_id, "failed", error=str(e), message="failed")
        finally:
            self._running = False
            self._active_job_id = None
            self._cancelled.discard(job_id)

    async def _finish(self, job_id, status, *, message=None, error=None) -> None:
        """Write a terminal status (done|failed|cancelled) under the registry lock.

        No-ops if the row is already terminal — e.g. the supervisor already
        marked it 'failed' for a stale heartbeat before the runner observed
        its own cancellation and unwound.
        """
        async with self._lock:
            db = core_db.SessionLocal()
            try:
                row = db.query(BenchJob).filter(BenchJob.id == job_id).first()
                if row is None or row.status in _TERMINAL_STATUSES:
                    return
                row.status = status
                if message is not None:
                    row.message = message
                if error is not None:
                    row.error = error
                row.updated_at = _now()
                db.commit()
            finally:
                db.close()

    async def cancel(self, job_id) -> bool:
        """Cancel a queued or running job immediately.

        The row is flipped to 'cancelled' right away; the runner is expected
        to observe check_cancelled() and stop before committing any partial
        work. Returns True on a state change, False if the job was already
        terminal or does not exist.
        """
        db = core_db.SessionLocal()
        try:
            row = db.query(BenchJob).filter(BenchJob.id == job_id).first()
            if row is None or row.status not in _ACTIVE_STATUSES:
                return False
            row.status = "cancelled"
            row.message = "cancelled by user"
            row.updated_at = _now()
            db.commit()
        finally:
            db.close()
        self._cancelled.add(job_id)
        return True

    async def get(self, job_id) -> dict | None:
        """Fetch one job as a dict, or None if it does not exist."""
        db = core_db.SessionLocal()
        try:
            row = db.query(BenchJob).filter(BenchJob.id == job_id).first()
            return _serialize(row) if row is not None else None
        finally:
            db.close()

    async def list(self) -> list:
        """List all bench_jobs rows, most recently created first."""
        db = core_db.SessionLocal()
        try:
            rows = db.query(BenchJob).order_by(BenchJob.created_at.desc()).all()
            return [_serialize(row) for row in rows]
        finally:
            db.close()

    async def reconcile_on_startup(self) -> int:
        """Instance-level entry point for the module-level startup reconciliation."""
        return await reconcile_interrupted_jobs()

    def start_supervisor(self) -> None:
        """Spawn the stale-heartbeat supervisor loop, if not already running."""
        if self._supervisor_task is None or self._supervisor_task.done():
            self._supervisor_task = self.loop.create_task(self._supervisor_loop())

    async def stop_supervisor(self) -> None:
        """Cancel the supervisor loop task and wait for it to unwind."""
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass
            self._supervisor_task = None

    async def _supervisor_loop(self):
        """Periodically fail active jobs whose heartbeat has gone stale."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            now = _now()
            db = core_db.SessionLocal()
            try:
                rows = db.query(BenchJob).filter(BenchJob.status.in_(_ACTIVE_STATUSES)).all()
                for row in rows:
                    last_activity = row.heartbeat_at or row.created_at
                    if last_activity is None:
                        continue
                    elapsed = (now - last_activity).total_seconds()
                    if elapsed > self._stale_timeout:
                        self._cancelled.add(row.id)
                        row.status = "failed"
                        row.error = f"stale heartbeat: no activity in {self._stale_timeout}s"
                        row.message = "failed"
                        row.updated_at = now
                        if row.id == self._active_job_id:
                            self._running = False
                            self._active_job_id = None
                db.commit()
            finally:
                db.close()


async def reconcile_interrupted_jobs() -> int:
    """Mark every queued/running bench_jobs row as failed (process restart).

    Idempotent — safe to call on every boot. Returns the number of rows
    changed.
    """
    db = core_db.SessionLocal()
    try:
        rows = db.query(BenchJob).filter(BenchJob.status.in_(_ACTIVE_STATUSES)).all()
        now = _now()
        count = 0
        for row in rows:
            row.status = "failed"
            row.error = "interrupted by restart"
            row.updated_at = now
            count += 1
        db.commit()
        return count
    finally:
        db.close()
