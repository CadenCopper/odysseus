"""Route tests for the ModelBench runs API
(start/list/poll/cancel — routes/modelbench/runs_routes.py).

Uses the REAL BenchJobRegistry bound to a temp sqlite DB (no runner attached,
mirroring tests/test_modelbench_job_registry.py's ``make_registry`` helper) so
submit/list/get/cancel run exactly as shipped. The registry's own dispatch
mechanics (background task scheduling, heartbeat, supervisor) are already
exercised end-to-end in test_modelbench_job_registry.py and are NOT
re-tested here; the poll/progress tests instead write directly to the
BenchJob row the same way the runner engine would, per the card's own
"or verify against a row you flip" allowance.

Model-tag / resident-model validation is faked via an injected ollama client
factory (mirrors test_modelbench_ollama.py's FakeClient) so these tests never
hit a live ollama.

Auth note: ``require_authenticated_request`` 401s any non-loopback caller
unless auth is disabled. TestClient presents a non-loopback host, so the
happy-path tests run with AUTH_ENABLED=false (single-user mode -> the route
lets them through) and the auth test runs with AUTH_ENABLED=true + a
configured manager so an unauthenticated caller is rejected with 401.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette.testclient")

from fastapi import FastAPI
from starlette.testclient import TestClient

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

from core import database as core_db  # noqa: E402
from core.database import BenchJob  # noqa: E402
from routes.modelbench import runs_routes as rroutes  # noqa: E402
from services.modelbench.job_registry import BenchJobRegistry, ConcurrentJobError  # noqa: E402
from services.modelbench.ollama_client import OllamaUnreachable  # noqa: E402


class FakeOllamaClient:
    """In-memory resident-model list stand-in (mirrors test_modelbench_ollama.FakeClient)."""

    def __init__(self, tags=None):
        self.tags_result = tags if tags is not None else []
        self.closed = False

    async def tags(self):
        if isinstance(self.tags_result, BaseException):
            raise self.tags_result
        return self.tags_result

    async def aclose(self):
        self.closed = True


def _resident(name, context_length=262144):
    return {
        "name": name,
        "model": name,
        "details": {"parameter_size": "9.0B", "context_length": context_length},
    }


VALID_BODY = {
    "model_tag": "test-model:latest",
    "think": False,
    "prompt": "hello world",
    "n_samples": 1,
}


@pytest.fixture
def registry(monkeypatch):
    """A real BenchJobRegistry bound to a fresh temp sqlite DB.

    No runner is attached by default, so a submitted job no-ops straight to
    done. Route tests that need *stateful* registry behavior (concurrency 409,
    cooperative cancel) use :class:`_StubRegistry` instead — the real registry
    is asyncio-lifetime and cannot hold ``_running``/cancel state across the
    separate event loops each TestClient request runs on.
    """
    clear_fake_database_modules()
    temp_session_local, engine, tmpfile = make_temp_sqlite(core_db.Base.metadata)
    monkeypatch.setattr(core_db, "SessionLocal", temp_session_local)
    reg = BenchJobRegistry()
    yield reg, temp_session_local
    engine.dispose()
    Path(tmpfile.name).unlink(missing_ok=True)


class _StubRegistry:
    """Deterministic registry stand-in for stateful route-behavior tests.

    ``submit`` raises ConcurrentJobError when ``active`` is set; ``cancel``
    returns per the pre-set ``job_status``. Lets the route tests assert the
    HTTP mapping (409 on active run, 200/cancelled on cooperative cancel)
    without depending on a live asyncio loop persisting across requests.
    """

    def __init__(self, active=False, cancel_status="running"):
        self.active = active
        self.job_status = cancel_status

    async def submit(self, **kwargs):
        if self.active:
            raise ConcurrentJobError("A bench run is already active; only one run is allowed on the 12GB GPU.")
        return {"job_id": "stub-job", "run_id": None, "status": "queued", **kwargs}

    async def list(self):
        return [{"job_id": "stub-job", "status": "queued"}]

    async def get(self, job_id):
        if job_id != "stub-job":
            return None
        return {"job_id": job_id, "status": self.job_status}

    async def cancel(self, job_id):
        if self.job_status in ("done", "failed", "cancelled"):
            return False
        self.job_status = "cancelled"
        return True



@pytest.fixture
def single_user(monkeypatch):
    """Run in single-user mode (auth disabled) so non-auth route tests pass."""
    monkeypatch.setenv("AUTH_ENABLED", "false")


@pytest.fixture
def auth_on(monkeypatch):
    """Run with auth enabled (the default) for the 401-rejection test."""
    monkeypatch.setenv("AUTH_ENABLED", "true")


def _client(registry_obj, tags, auth_configured=False):
    app = FastAPI()
    if auth_configured:
        app.state.auth_manager = SimpleNamespace(is_configured=True)
    app.include_router(rroutes.setup_runs_routes(
        registry_obj,
        ollama_client_factory=lambda: FakeOllamaClient(tags=tags),
    ))
    return TestClient(app, raise_server_exceptions=False)


def _set_row(session_local, job_id, **fields):
    db = session_local()
    try:
        row = db.query(BenchJob).filter(BenchJob.id == job_id).first()
        for k, v in fields.items():
            setattr(row, k, v)
        db.commit()
    finally:
        db.close()


# --- POST /api/modelbench/runs ----------------------------------------------

def test_start_run_creates_queued_job(registry, single_user):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    r = c.post("/api/modelbench/runs", json=VALID_BODY)
    assert r.status_code == 201
    body = r.json()
    assert body["job_id"]
    assert body["status"] == "queued"
    assert body["run_id"] is None


def test_second_concurrent_start_returns_409(single_user):
    reg = _StubRegistry(active=True)
    c = _client(reg, [_resident("test-model:latest")])
    r = c.post("/api/modelbench/runs", json=VALID_BODY)
    assert r.status_code in (409, 423)
    assert "already active" in r.text.lower()


def test_start_run_rejects_unknown_model_tag(registry, single_user):
    reg, _ = registry
    c = _client(reg, [_resident("other-model:latest")])
    r = c.post("/api/modelbench/runs", json=VALID_BODY)
    assert r.status_code in (400, 404)
    assert "model_tag" in r.text.lower()


def test_start_run_rejects_empty_prompt(registry, single_user):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    body = {**VALID_BODY, "prompt": "   "}
    r = c.post("/api/modelbench/runs", json=body)
    assert r.status_code == 400
    assert "prompt" in r.text.lower()


@pytest.mark.parametrize("n", [0, 21])
def test_start_run_rejects_out_of_range_n_samples(registry, single_user, n):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    body = {**VALID_BODY, "n_samples": n}
    r = c.post("/api/modelbench/runs", json=body)
    assert r.status_code == 400
    assert "n_samples" in r.text.lower()


@pytest.mark.parametrize("ctx_target", [0, -1])
def test_start_run_rejects_non_positive_ctx_target(registry, single_user, ctx_target):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    body = {**VALID_BODY, "ctx_target": ctx_target}
    r = c.post("/api/modelbench/runs", json=body)
    assert r.status_code == 400
    assert "ctx_target" in r.text.lower()


def test_start_run_rejects_ctx_target_over_cap(registry, single_user):
    reg, _ = registry
    # "test-model" matches no MODEL_CTX_CAPS key -> LOW_CTX_CAP=8192, unaffected
    # by the resident model's own (larger) advertised context_length.
    c = _client(reg, [_resident("test-model:latest", context_length=262144)])
    body = {**VALID_BODY, "ctx_target": 100000}
    r = c.post("/api/modelbench/runs", json=body)
    assert r.status_code == 400
    assert "ctx_target" in r.text.lower()


@pytest.mark.parametrize("bad_think", ["true", None, "null"])
def test_start_run_rejects_non_bool_think(registry, single_user, bad_think):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    body = {**VALID_BODY, "think": bad_think}
    r = c.post("/api/modelbench/runs", json=body)
    assert r.status_code == 400
    assert "think" in r.text.lower()


def test_start_run_maps_ollama_unreachable_to_502(registry, single_user):
    reg, _ = registry
    c = _client(reg, OllamaUnreachable("conn refused"))
    r = c.post("/api/modelbench/runs", json=VALID_BODY)
    assert r.status_code == 502


def test_start_run_unauthenticated_returns_401(registry, auth_on):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")], auth_configured=True)
    r = c.post("/api/modelbench/runs", json=VALID_BODY)
    assert r.status_code == 401


# --- GET /api/modelbench/runs ------------------------------------------------

def test_list_runs_includes_created_job(registry, single_user):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    started = c.post("/api/modelbench/runs", json=VALID_BODY).json()
    r = c.get("/api/modelbench/runs")
    assert r.status_code == 200
    ids = [j["job_id"] for j in r.json()["runs"]]
    assert started["job_id"] in ids


def test_list_runs_unauthenticated_returns_401(registry, auth_on):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")], auth_configured=True)
    r = c.get("/api/modelbench/runs")
    assert r.status_code == 401


# --- GET /api/modelbench/runs/{job_id} --------------------------------------

def test_poll_reflects_progress_and_run_id(registry, single_user):
    reg, session_local = registry
    c = _client(reg, [_resident("test-model:latest")])
    started = c.post("/api/modelbench/runs", json=VALID_BODY).json()
    job_id = started["job_id"]
    _set_row(
        session_local, job_id,
        status="running", progress=0.5, message="token test 1/2", run_id="run_abc123",
    )
    r = c.get(f"/api/modelbench/runs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] == "running"
    assert body["progress"] == 0.5
    assert body["run_id"] == "run_abc123"


def test_poll_reflects_terminal_status(registry, single_user):
    reg, session_local = registry
    c = _client(reg, [_resident("test-model:latest")])
    started = c.post("/api/modelbench/runs", json=VALID_BODY).json()
    job_id = started["job_id"]
    _set_row(session_local, job_id, status="done", progress=1.0, message="completed")
    r = c.get(f"/api/modelbench/runs/{job_id}")
    assert r.json()["status"] == "done"


def test_poll_unknown_job_returns_404(registry, single_user):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")])
    r = c.get("/api/modelbench/runs/no-such-job")
    assert r.status_code == 404


# --- POST /api/modelbench/runs/{job_id}/cancel ------------------------------
# Stateful cancel behavior (route maps registry.cancel -> HTTP) is tested via
# _StubRegistry for determinism; the cooperative cancel semantics themselves
# (0 partial rows, drains in-flight sample) are the registry's job and are
# covered in test_modelbench_job_registry.py.

def test_cancel_mid_run_returns_cancelled(single_user):
    reg = _StubRegistry(cancel_status="running")
    c = _client(reg, [_resident("test-model:latest")])
    r = c.post("/api/modelbench/runs/stub-job/cancel")
    assert r.status_code == 200
    assert r.json() == {"job_id": "stub-job", "status": "cancelled"}


def test_cancel_already_terminal_returns_current_status(single_user):
    reg = _StubRegistry(cancel_status="done")
    c = _client(reg, [_resident("test-model:latest")])
    r = c.post("/api/modelbench/runs/stub-job/cancel")
    assert r.status_code == 200
    assert r.json() == {"job_id": "stub-job", "status": "done"}


def test_cancel_unknown_job_returns_404(single_user):
    reg = _StubRegistry()
    c = _client(reg, [_resident("test-model:latest")])
    r = c.post("/api/modelbench/runs/no-such-job/cancel")
    assert r.status_code == 404


def test_cancel_unauthenticated_returns_401(registry, auth_on):
    reg, _ = registry
    c = _client(reg, [_resident("test-model:latest")], auth_configured=True)
    r = c.post("/api/modelbench/runs/some-job/cancel")
    assert r.status_code == 401
