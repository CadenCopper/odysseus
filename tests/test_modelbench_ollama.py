"""Route tests for the ollama-native ModelBench runner endpoints
(GET /api/modelbench/ollama/models + POST /api/modelbench/ollama/models/{tag}/pull).

The ollama HTTP surface is faked via an injected client factory; bench_samples
context (has-samples + vrram_fit annotation) is seeded in a temp sqlite DB,
per the repo's DB-route-test convention (see test_modelbench_routes.py).

Auth note: ``require_authenticated_request`` 401s any non-loopback caller unless
auth is disabled. TestClient presents a non-loopback host, so the happy-path
tests run with AUTH_ENABLED=false (single-user mode -> the route lets them
through) and the auth tests run with AUTH_ENABLED=true + a configured manager so
an unauthenticated caller is rejected with 401.
"""

from datetime import datetime, timedelta
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

from core.database import Base, BenchSample  # noqa: E402
from routes.modelbench import ollama_routes as oroutes  # noqa: E402
from services.modelbench.ollama_client import (  # noqa: E402
    OllamaClient,
    OllamaPullError,
    OllamaUnreachable,
)

BASE_TIME = datetime(2026, 1, 1, 0, 0, 0)


class FakeClient:
    """In-memory stand-in for OllamaClient (tags + streamed pull)."""

    def __init__(self, tags=None, pull_events=None, pull_error=None, root="http://fake:11434/api"):
        self.tags_result = tags if tags is not None else []
        self.pull_events = pull_events if pull_events is not None else []
        self.pull_error = pull_error
        self.root = root
        self.closed = False

    async def tags(self):
        if isinstance(self.tags_result, BaseException):
            raise self.tags_result
        return self.tags_result

    async def stream_pull(self, tag):
        if self.pull_error is not None:
            raise self.pull_error
        for ev in self.pull_events:
            yield ev

    async def aclose(self):
        self.closed = True


def _resident(name, size_gb, parameter_size="9.0B"):
    return {
        "name": name,
        "model": name,
        "size": int(size_gb * 1e9),
        "details": {"parameter_size": parameter_size, "context_length": 262144},
        "capabilities": ["completion"],
        "modified_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture(scope="module")
def seeded_db():
    SessionLocal, engine, tmpfile = make_temp_sqlite(Base.metadata)
    db = SessionLocal()
    try:
        rows = [
            BenchSample(
                run_id="df-001", model_tag="defiant-fable:latest", true_params=9.0,
                quant="Q4_K_M", ctx_len=116736, think=False, prompt_bytes=100,
                temperature=0.7, seed=1, ollama_version="0.32.13", vrram_fit="fit",
                output_tokens=50, thinking_tokens=0, content_tokens=50,
                tokens_per_sec=10.0, ttft_ms=100.0, latency_ms=1000.0,
                created_at=BASE_TIME,
            ),
            BenchSample(
                run_id="df-002", model_tag="defiant-fable:latest", true_params=9.0,
                quant="Q4_K_M", ctx_len=131072, think=False, prompt_bytes=100,
                temperature=0.7, seed=2, ollama_version="0.32.13", vrram_fit="partial",
                output_tokens=50, thinking_tokens=0, content_tokens=50,
                tokens_per_sec=9.0, ttft_ms=110.0, latency_ms=1010.0,
                created_at=BASE_TIME + timedelta(seconds=1),
            ),
            BenchSample(
                run_id="q-001", model_tag="qwen3-14b-ctx:latest", true_params=14.8,
                quant="Q4_K_M", ctx_len=40960, think=False, prompt_bytes=100,
                temperature=0.7, seed=3, ollama_version="0.32.13", vrram_fit="partial",
                output_tokens=50, thinking_tokens=0, content_tokens=50,
                tokens_per_sec=15.0, ttft_ms=90.0, latency_ms=900.0,
                created_at=BASE_TIME + timedelta(seconds=2),
            ),
        ]
        db.add_all(rows)
        db.commit()
    finally:
        db.close()
    yield SessionLocal
    engine.dispose()
    Path(tmpfile.name).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def release_pull_gate():
    """Ensure the module-level single-pull gate is never leaked between tests."""
    yield
    if oroutes._pull_gate.active:
        oroutes._pull_gate.release()


@pytest.fixture
def single_user(monkeypatch):
    """Run in single-user mode (auth disabled) so non-auth route tests pass."""
    monkeypatch.setenv("AUTH_ENABLED", "false")


@pytest.fixture
def auth_on(monkeypatch):
    """Run with auth enabled (the default) for the 401-rejection tests."""
    monkeypatch.setenv("AUTH_ENABLED", "true")


def _client(factory_result, bench_registry=None, auth_configured=False):
    app = FastAPI()
    if auth_configured:
        app.state.auth_manager = SimpleNamespace(is_configured=True)
    app.include_router(oroutes.setup_ollama_routes(
        ollama_client_factory=lambda: factory_result,
        bench_registry=bench_registry,
    ))
    return TestClient(app, raise_server_exceptions=False)


def _sse_events(text):
    """Parse an SSE body into a list of JSON objects."""
    import json as _json

    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                try:
                    out.append(_json.loads(line[len("data: "):]))
                except ValueError:
                    continue
    return out


# --- GET /api/modelbench/ollama/models -------------------------------------

def test_ollama_models_lists_resident_annotated(seeded_db, monkeypatch, single_user):
    monkeypatch.setattr(oroutes, "SessionLocal", seeded_db)
    tags = [
        _resident("defiant-fable:latest", 6.83),
        _resident("qwen3-14b-ctx:latest", 9.28),
        _resident("hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M", 16.81, "27.3B"),
        _resident("brand-new:latest", 5.0),
    ]
    c = _client(FakeClient(tags=tags))
    r = c.get("/api/modelbench/ollama/models")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["resident_count"] == 4
    assert body["meta"]["bench_sampled_count"] == 2
    assert body["meta"]["max_pull_size_gb"] == 16.0

    by_name = {m["name"]: m for m in body["models"]}
    # Has samples + recorded vrram_fit (fit is the mode of the 2 defiant-fable rows).
    fable = by_name["defiant-fable:latest"]
    assert fable["has_samples"] is True
    assert fable["vrram_fit"] == "fit"
    assert fable["size_gb"] == 6.83
    assert fable["parameter_size"] == "9.0B"
    # 27B pair has no samples yet -> has_samples False, vrram_fit unknown.
    big = by_name["hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M"]
    assert big["has_samples"] is False
    assert big["vrram_fit"] is None
    assert big["size_gb"] == 16.81
    # Brand-new model: no samples, unknown fit.
    assert by_name["brand-new:latest"]["has_samples"] is False
    assert by_name["brand-new:latest"]["vrram_fit"] is None


def test_ollama_models_unreachable_returns_502(seeded_db, monkeypatch, single_user):
    monkeypatch.setattr(oroutes, "SessionLocal", seeded_db)
    c = _client(FakeClient(tags=OllamaUnreachable("conn refused")))
    r = c.get("/api/modelbench/ollama/models")
    assert r.status_code == 502


# --- POST /api/modelbench/ollama/models/{tag}/pull --------------------------

def test_ollama_pull_rejects_concurrent_pull(single_user):
    # Simulate an in-flight pull occupying the single slot.
    assert oroutes._pull_gate.try_acquire() is True
    c = _client(FakeClient(tags=[_resident("x:latest", 1.0)]))
    r = c.post("/api/modelbench/ollama/models/x:latest/pull", json={})
    assert r.status_code in (409, 423)
    assert "already in progress" in r.text.lower()
    oroutes._pull_gate.release()


def test_ollama_pull_rejects_oversize_without_confirm(seeded_db, monkeypatch, single_user):
    monkeypatch.setattr(oroutes, "SessionLocal", seeded_db)
    # 27B resident model is 16.81GB > 16GB threshold -> refused unless confirm.
    tags = [_resident("hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M", 16.81, "27.3B")]
    c = _client(FakeClient(tags=tags))
    r = c.post(
        "/api/modelbench/ollama/models/hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M/pull",
        json={},
    )
    assert r.status_code == 400
    assert "safety threshold" in r.text.lower()
    assert "confirm" in r.text.lower()


def test_ollama_pull_oversize_allowed_with_explicit_confirm(single_user):
    tags = [_resident("hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M", 16.81, "27.3B")]
    pull_events = [{"status": "pulling manifest"}, {"status": "success"}]
    c = _client(FakeClient(tags=tags, pull_events=pull_events))
    r = c.post(
        "/api/modelbench/ollama/models/hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M/pull",
        json={"confirm": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[0]["event"] == "started"
    assert events[0]["model"] == "hf.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED:Q4_K_M"
    assert {"pulling manifest", "success"} == {e.get("status") for e in events if "status" in e}
    assert events[-1] == {"event": "done", "ok": True}


def test_ollama_pull_streams_error_and_done_not_ok(single_user):
    c = _client(FakeClient(
        tags=[],
        pull_error=OllamaPullError("pull model manifest: file does not exist"),
    ))
    r = c.post("/api/modelbench/ollama/models/no-such-tag:latest/pull", json={})
    assert r.status_code == 200  # SSE; failure is in-band, per live ollama contract
    events = _sse_events(r.text)
    error = next(e for e in events if e.get("event") == "error")
    assert "file does not exist" in error["error"]
    assert events[-1] == {"event": "done", "ok": False}


def test_ollama_pull_refuses_while_bench_run_active(single_user):
    busy_registry = SimpleNamespace(has_active_run=lambda: True)
    c = _client(FakeClient(tags=[_resident("x:latest", 1.0)]), bench_registry=busy_registry)
    r = c.post("/api/modelbench/ollama/models/x:latest/pull", json={})
    assert r.status_code == 409
    assert "bench run is active" in r.text.lower()


def test_ollama_pull_unauthenticated_returns_401(auth_on):
    # auth configured + no current_user + non-loopback TestClient host -> 401.
    c = _client(FakeClient(tags=[_resident("x:latest", 1.0)]), auth_configured=True)
    r = c.post("/api/modelbench/ollama/models/x:latest/pull", json={})
    assert r.status_code == 401


def test_ollama_models_unauthenticated_returns_401(auth_on):
    c = _client(FakeClient(tags=[_resident("x:latest", 1.0)]), auth_configured=True)
    r = c.get("/api/modelbench/ollama/models")
    assert r.status_code == 401


# --- OllamaClient HTTP layer (MockTransport) --------------------------------

def test_client_root_normalization():
    assert OllamaClient(root="http://x:11434/api").root == "http://x:11434/api"
    assert OllamaClient(root="http://x:11434").root == "http://x:11434/api"
    assert OllamaClient(root="http://x:11434/api/").root == "http://x:11434/api"


def test_client_tags_hits_native_endpoint():
    import httpx as _httpx

    def handler(request):
        assert request.url == "http://x:11434/api/tags", request.url
        assert request.method == "GET"
        return _httpx.Response(200, json={"models": [{"name": "m1", "size": 1000}]})

    transport = _httpx.MockTransport(handler)
    client = OllamaClient(root="http://x:11434", http=_httpx.AsyncClient(transport=transport))
    import asyncio as _asyncio

    models = _asyncio.run(client.tags())
    assert models[0]["name"] == "m1"


def test_client_stream_pull_scans_inband_error():
    import httpx as _httpx
    import asyncio as _asyncio

    stream_body = (
        '{"status":"pulling manifest"}\n'
        '{"error":"pull model manifest: file does not exist"}\n'
    )

    def handler(request):
        assert request.url == "http://x:11434/api/pull", request.url
        assert request.method == "POST"
        return _httpx.Response(200, text=stream_body)  # HTTP 200 + in-band error

    transport = _httpx.MockTransport(handler)
    client = OllamaClient(root="http://x:11434", http=_httpx.AsyncClient(transport=transport))

    with pytest.raises(OllamaPullError) as exc:
        _asyncio.run(_collect(client.stream_pull("no-such")))
    assert "file does not exist" in str(exc.value)


async def _collect(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out

