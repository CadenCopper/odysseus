"""Tests for services/modelbench/job_registry.py (BenchJobRegistry).

Exercises the single-concurrency asyncio job registry against a temp
file-backed sqlite database bound onto core.database.SessionLocal, so the
registry code under test is exactly what ships — only the database it talks
to is swapped out. No network, no ollama.
"""
import asyncio
from pathlib import Path

import pytest

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

from sqlalchemy import inspect

from core import database as core_db
from core.database import BenchJob
from services.modelbench.job_registry import (
    BenchJobRegistry,
    ConcurrentJobError,
    reconcile_interrupted_jobs,
)

EXPECTED_COLUMNS = {
    "id",
    "run_id",
    "status",
    "model_tag",
    "think",
    "ctx_target",
    "ctx_series",
    "prompt",
    "n_samples",
    "progress",
    "message",
    "error",
    "created_at",
    "updated_at",
    "heartbeat_at",
}


class FakeRunner:
    """Gated fake run: produces samples one at a time under test control.

    Checks ``ctx.check_cancelled()`` BEFORE each sample (never mid-sample),
    so a cancellation observed between samples drains the sample already in
    flight but never starts building the next one.
    """

    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.committed = 0
        self.completed_samples = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.done = False

    async def __call__(self, ctx):
        for i in range(self.n_samples):
            await ctx.check_cancelled()
            self.started.set()
            await self.release.wait()
            self.completed_samples += 1
            self.committed += 1
            await ctx.set_progress((i + 1) / self.n_samples, f"sample {i + 1}/{self.n_samples}")
        self.done = True


async def sleeping_runner(ctx):
    """Runner that never heartbeats or progresses, to exercise the stale-heartbeat supervisor."""
    while True:
        await ctx.check_cancelled()
        await asyncio.sleep(0.01)


def make_registry(monkeypatch, runner=None, **kwargs):
    """Build a BenchJobRegistry bound to a fresh temp sqlite database.

    job_registry.py reads SessionLocal off the core.database module at call
    time (via ``core_db.SessionLocal()``), so patching the attribute here is
    enough to isolate every write the registry makes from the real app db.
    """
    clear_fake_database_modules()
    temp_session_local, engine, tmpfile = make_temp_sqlite(core_db.Base.metadata)
    monkeypatch.setattr(core_db, "SessionLocal", temp_session_local)
    registry = BenchJobRegistry(runner=runner, **kwargs)
    return registry, engine, tmpfile, temp_session_local


async def _wait_for_status(registry, job_id, status, timeout=2.0):
    async def _poll():
        while True:
            job = await registry.get(job_id)
            if job is not None and job["status"] == status:
                return job
            await asyncio.sleep(0.005)

    return await asyncio.wait_for(_poll(), timeout=timeout)


async def _wait_until(predicate, timeout=2.0):
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _cleanup(engine, tmpfile):
    engine.dispose()
    Path(tmpfile.name).unlink(missing_ok=True)


async def test_submit_creates_queued_row_and_run_id_nullable_on_start(monkeypatch):
    registry, engine, tmpfile, _ = make_registry(monkeypatch, runner=FakeRunner(1))
    try:
        result = await registry.submit(
            model_tag="hf.co/models/model-a:Q4_K_M", prompt="hello", n_samples=1
        )
        job = await registry.get(result["job_id"])
        assert job["status"] == "queued"
        assert job["run_id"] is None
    finally:
        _cleanup(engine, tmpfile)


async def test_second_concurrent_start_raises_concurrent_job_error(monkeypatch):
    fake_runner = FakeRunner(2)
    registry, engine, tmpfile, _ = make_registry(monkeypatch, runner=fake_runner)
    try:
        await registry.submit(model_tag="model-a", prompt="hello", n_samples=2)
        await asyncio.wait_for(fake_runner.started.wait(), timeout=2.0)

        with pytest.raises(ConcurrentJobError):
            await registry.submit(model_tag="model-b", prompt="hello again", n_samples=1)
    finally:
        fake_runner.release.set()
        await _wait_until(lambda: registry._active_job_id is None)
        _cleanup(engine, tmpfile)


async def test_queued_to_running_to_done_transition(monkeypatch):
    fake_runner = FakeRunner(1)
    registry, engine, tmpfile, _ = make_registry(monkeypatch, runner=fake_runner)
    try:
        result = await registry.submit(model_tag="model-a", prompt="hello", n_samples=1)
        job_id = result["job_id"]

        fake_runner.release.set()
        job = await _wait_for_status(registry, job_id, "done")

        assert job["status"] == "done"
        assert job["message"] == "completed"
        assert job["progress"] == 1.0
    finally:
        _cleanup(engine, tmpfile)


async def test_cancel_mid_run_leaves_zero_partial_rows(monkeypatch):
    fake_runner = FakeRunner(3)
    registry, engine, tmpfile, _ = make_registry(monkeypatch, runner=fake_runner)
    try:
        result = await registry.submit(model_tag="model-a", prompt="hello", n_samples=3)
        job_id = result["job_id"]
        await asyncio.wait_for(fake_runner.started.wait(), timeout=2.0)

        cancelled = await registry.cancel(job_id)
        assert cancelled is True

        # Flipped immediately, before the runner has drained its in-flight sample.
        job = await registry.get(job_id)
        assert job["status"] == "cancelled"

        # Release the gate so the runner drains the sample already in flight
        # and then observes cancellation before starting a new one. Wait for
        # the dispatch task itself to settle (status flips to 'cancelled'
        # immediately on cancel(), so polling status alone would race ahead
        # of the drain).
        fake_runner.release.set()
        await _wait_until(lambda: registry._active_job_id is None)
        job = await registry.get(job_id)

        assert job["status"] == "cancelled"
        # No half-built sample: committed always tracks fully-completed
        # samples, and at most the one already in flight when cancel() fired
        # is allowed to drain — no new sample is ever started afterward.
        assert fake_runner.completed_samples == fake_runner.committed
        assert fake_runner.committed <= 1
        assert fake_runner.committed < fake_runner.n_samples
    finally:
        _cleanup(engine, tmpfile)


async def test_stale_heartbeat_auto_cancel_fires(monkeypatch):
    registry, engine, tmpfile, _ = make_registry(
        monkeypatch, runner=sleeping_runner, heartbeat_interval=0.02, stale_timeout=0.05
    )
    try:
        result = await registry.submit(model_tag="model-a", prompt="hello", n_samples=1)
        job_id = result["job_id"]

        registry.start_supervisor()
        job = await _wait_for_status(registry, job_id, "failed", timeout=3.0)

        assert job["status"] == "failed"
        assert "stale heartbeat" in job["error"]
        assert registry._running is False
    finally:
        await registry.stop_supervisor()
        _cleanup(engine, tmpfile)


async def test_reconcile_interrupted_jobs_marks_running_failed(monkeypatch):
    clear_fake_database_modules()
    temp_session_local, engine, tmpfile = make_temp_sqlite(core_db.Base.metadata)
    monkeypatch.setattr(core_db, "SessionLocal", temp_session_local)
    try:
        db = temp_session_local()
        try:
            row = BenchJob(
                id="job-running-1",
                run_id=None,
                status="running",
                model_tag="model-a",
                prompt="hello",
                n_samples=1,
                progress=0.5,
                message="in progress",
            )
            db.add(row)
            db.commit()
        finally:
            db.close()

        count = await reconcile_interrupted_jobs()
        assert count == 1

        db = temp_session_local()
        try:
            row = db.query(BenchJob).filter(BenchJob.id == "job-running-1").first()
            assert row.status == "failed"
            assert row.error == "interrupted by restart"
        finally:
            db.close()
    finally:
        _cleanup(engine, tmpfile)


def test_bench_jobs_table_columns():
    SessionLocal, engine, tmpfile = make_temp_sqlite(core_db.Base.metadata)
    try:
        inspector = inspect(engine)
        assert "bench_jobs" in inspector.get_table_names()
        columns = {col["name"] for col in inspector.get_columns("bench_jobs")}
        assert columns == EXPECTED_COLUMNS
    finally:
        _cleanup(engine, tmpfile)
