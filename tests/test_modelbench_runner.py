"""Tests for services/modelbench/runner.py (the ModelBench run_bench engine).

Mirrors tests/test_modelbench_job_registry.py: a temp file-backed sqlite
database is bound onto core.database.SessionLocal so the runner code under
test is exactly what ships -- only the database and the ollama HTTP transport
are swapped out (httpx.MockTransport fakes ollama; no network).
"""
import json
from pathlib import Path

import httpx
import pytest

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

from core import database as core_db
from core.database import BenchJob, BenchSample
from services.modelbench import runner
from services.modelbench.job_registry import JobContext


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------

class FakeRegistry:
    """Minimal stand-in for BenchJobRegistry: JobContext only reads _cancelled."""

    def __init__(self):
        self._cancelled = set()


def make_db(monkeypatch):
    clear_fake_database_modules()
    temp_session_local, engine, tmpfile = make_temp_sqlite(core_db.Base.metadata)
    monkeypatch.setattr(core_db, "SessionLocal", temp_session_local)
    return temp_session_local, engine, tmpfile


def cleanup(engine, tmpfile):
    engine.dispose()
    Path(tmpfile.name).unlink(missing_ok=True)


def make_job(session_local, *, model_tag="test-model:latest", think=None,
             ctx_target=None, ctx_sweep=None, prompt="Say hello.", n_samples=2):
    db = session_local()
    try:
        job = BenchJob(
            id="job-1",
            run_id=None,
            status="running",
            model_tag=model_tag,
            think=think,
            ctx_target=ctx_target,
            ctx_sweep=ctx_sweep,
            prompt=prompt,
            n_samples=n_samples,
            progress=0.0,
            message="",
        )
        db.add(job)
        db.commit()
        return job.id
    finally:
        db.close()


def make_ctx(job_id):
    return JobContext(FakeRegistry(), job_id)


def ndjson_chat_response(*, thinking_deltas=(), content_deltas=(),
                          eval_count=20, eval_duration=500_000_000,
                          prompt_eval_duration=100_000_000, total_duration=1_000_000_000):
    """Build a native-thinking-shaped NDJSON body (thinking/content on separate channels)."""
    lines = []
    for t in thinking_deltas:
        lines.append(json.dumps({
            "model": "test-model:latest", "created_at": "t",
            "message": {"role": "assistant", "content": "", "thinking": t},
            "done": False,
        }))
    for c in content_deltas:
        lines.append(json.dumps({
            "model": "test-model:latest", "created_at": "t",
            "message": {"role": "assistant", "content": c, "thinking": ""},
            "done": False,
        }))
    lines.append(json.dumps({
        "model": "test-model:latest", "created_at": "t",
        "message": {"role": "assistant", "content": "", "thinking": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": total_duration,
        "load_duration": 1000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
    }))
    return ("\n".join(lines) + "\n").encode("utf-8")


TAGS_RESPONSE = {
    "models": [
        {
            "name": "test-model:latest",
            "details": {
                "parameter_size": "7.6B",
                "quantization_level": "Q4_K_M",
                "context_length": 2048,
            },
            "capabilities": ["completion", "thinking"],
        }
    ]
}

PS_RESPONSE_FIT = {
    "models": [{"name": "test-model:latest", "size": 1000, "size_vram": 1000}]
}

PS_RESPONSE_OFFLOAD = {
    "models": [{"name": "test-model:latest", "size": 1000, "size_vram": 0}]
}

VERSION_RESPONSE = {"version": "0.32.13"}


def make_client(chat_handler, *, ps_response=PS_RESPONSE_FIT, ps_status=200,
                 tags_response=TAGS_RESPONSE):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return chat_handler(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags_response)
        if request.url.path == "/api/ps":
            if ps_status != 200:
                return httpx.Response(ps_status)
            return httpx.Response(200, json=ps_response)
        if request.url.path == "/api/version":
            return httpx.Response(200, json=VERSION_RESPONSE)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(base_url=runner.OLLAMA_URL, transport=transport)


def always_ok_chat_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=ndjson_chat_response(
        thinking_deltas=["thinking a bit "], content_deltas=["ok"],
    ))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty_is_zero():
    assert runner.estimate_tokens("") == 0
    assert runner.estimate_tokens(None) == 0


def test_estimate_tokens_deterministic_nonzero():
    text = "abcdefgh"  # 8 chars -> ceil(8/4) = 2
    assert runner.estimate_tokens(text) == 2
    assert runner.estimate_tokens(text) == runner.estimate_tokens(text)


def test_parse_true_params_from_parameter_size():
    assert runner.parse_true_params("27.3B") == 27.3
    assert runner.parse_true_params("7.6B") == 7.6
    assert runner.parse_true_params(None) is None
    assert runner.parse_true_params("") is None


def test_resolve_ollama_url_refuses_known_empty_bench_host(monkeypatch):
    """OLLAMA_URL pointing at the research-verified near-empty 10.0.0.188
    bench instance is overridden to the real copper-wsl ollama (10.0.0.8) --
    the deployment trap the research card flagged. Any OTHER explicit env is
    respected verbatim."""
    monkeypatch.setenv("OLLAMA_URL", "http://10.0.0.188:11434")
    assert runner._resolve_ollama_url() == "http://10.0.0.8:11434"

    monkeypatch.setenv("OLLAMA_URL", "http://10.0.0.8:11434")
    assert runner._resolve_ollama_url() == "http://10.0.0.8:11434"

    monkeypatch.setenv("OLLAMA_URL", "http://10.0.0.9:11434/")
    assert runner._resolve_ollama_url() == "http://10.0.0.9:11434"

    monkeypatch.delenv("OLLAMA_URL", raising=False)
    assert runner._resolve_ollama_url() == "http://10.0.0.8:11434"


def test_ctx_sweep_points_bounded_and_increasing():
    points = runner.ctx_sweep_points(200_000)
    assert len(points) <= runner.MAX_CTX_SWEEP_PROBES
    assert points == sorted(points)
    assert all(p <= 200_000 for p in points)


def test_ctx_sweep_points_small_cap_is_short():
    points = runner.ctx_sweep_points(100)
    assert points == [100]


def test_resolve_sweep_points_uses_provided_series():
    # A cap-validated user series (JSON in the row) is used verbatim.
    assert runner.resolve_sweep_points("[512, 768]", 200_000) == [512, 768]


def test_resolve_sweep_points_falls_back_when_absent():
    # NULL / empty / malformed all degrade to the doubling fallback.
    assert runner.resolve_sweep_points(None, 2048) == runner.ctx_sweep_points(2048)
    assert runner.resolve_sweep_points("", 2048) == runner.ctx_sweep_points(2048)
    assert runner.resolve_sweep_points("not json", 2048) == runner.ctx_sweep_points(2048)
    assert runner.resolve_sweep_points("{}", 2048) == runner.ctx_sweep_points(2048)


def test_resolve_sweep_points_sorts_dedupes_and_clamps_each_point():
    # A user series is normalized: sorted ascending, deduped, and every point
    # clamped to the model's effective cap (MB-Dash-3b decision 2).
    assert runner.resolve_sweep_points("[768, 512, 768, 99999]", 2048) == [512, 768, 2048]
    # Over-cap entries collapse onto the cap rather than being dropped.
    assert runner.resolve_sweep_points("[99999]", 4096) == [4096]
    # Cap not exceeded -> points pass through sorted + deduped.
    assert runner.resolve_sweep_points("[4096, 1024, 2048, 1024]", 8192) == [1024, 2048, 4096]


def test_resolve_sweep_points_is_bounded_to_max_probes():
    raw = "[" + ",".join(str(1024 * i) for i in range(1, 30)) + "]"
    out = runner.resolve_sweep_points(raw, 1_000_000)
    assert len(out) <= runner.MAX_CTX_SWEEP_PROBES
    assert out == sorted(out)


# ---------------------------------------------------------------------------
# 1. think/content split -- separate channels, not summed
# ---------------------------------------------------------------------------

def test_think_and_content_channels_kept_separate():
    raw = ndjson_chat_response(
        thinking_deltas=["Let me think ", "about this."],
        content_deltas=["The answer is 42."],
    )
    thinking_text, content_text, done_event = runner.parse_ndjson_stream(raw)

    assert thinking_text == "Let me think about this."
    assert content_text == "The answer is 42."

    metrics = runner.compute_metrics(done_event, thinking_text, content_text)
    assert metrics["thinking_tokens"] == runner.estimate_tokens(thinking_text)
    assert metrics["content_tokens"] == runner.estimate_tokens(content_text)
    # Distinct channels, not a single summed/concatenated count.
    assert metrics["thinking_tokens"] != metrics["content_tokens"]
    assert metrics["thinking_tokens"] + metrics["content_tokens"] != runner.estimate_tokens(
        thinking_text + content_text
    ) or metrics["thinking_tokens"] != metrics["content_tokens"]


def test_content_templated_think_tags_split_into_thinking_channel():
    raw = ndjson_chat_response(
        thinking_deltas=[],
        content_deltas=["<think>reasoning here</think>final answer"],
    )
    thinking_text, content_text, done_event = runner.parse_ndjson_stream(raw)

    assert thinking_text == "reasoning here"
    assert content_text == "final answer"


# ---------------------------------------------------------------------------
# 2. tok/s + TTFT + latency from a known done event
# ---------------------------------------------------------------------------

def test_compute_metrics_numeric_values_from_done_event():
    done_event = {
        "eval_count": 100,
        "eval_duration": 2_000_000_000,  # 2s -> 50 tok/s
        "prompt_eval_duration": 250_000_000,  # 0.25s -> 250ms TTFT
        "total_duration": 3_000_000_000,  # 3s -> 3000ms latency
    }
    metrics = runner.compute_metrics(done_event, "", "")
    assert metrics["output_tokens"] == 100
    assert metrics["tokens_per_sec"] == pytest.approx(50.0)
    assert metrics["ttft_ms"] == pytest.approx(250.0)
    assert metrics["latency_ms"] == pytest.approx(3000.0)


def test_compute_metrics_guards_divide_by_zero_eval_duration():
    done_event = {
        "eval_count": 10, "eval_duration": 0,
        "prompt_eval_duration": 100_000_000, "total_duration": 500_000_000,
    }
    metrics = runner.compute_metrics(done_event, "", "")
    assert metrics["tokens_per_sec"] is None


# ---------------------------------------------------------------------------
# 3. vrram_fit capture
# ---------------------------------------------------------------------------

def test_vrram_fit_from_ps_fit_and_offload():
    fit, incomplete = runner.vrram_fit_from_ps(PS_RESPONSE_FIT, "test-model:latest")
    assert fit == "fit"
    assert incomplete is False

    offload, incomplete = runner.vrram_fit_from_ps(PS_RESPONSE_OFFLOAD, "test-model:latest")
    assert offload == "offload"
    assert incomplete is False


def test_vrram_fit_from_ps_partial():
    partial_ps = {"models": [{"name": "test-model:latest", "size": 1000, "size_vram": 400}]}
    fit, incomplete = runner.vrram_fit_from_ps(partial_ps, "test-model:latest")
    assert fit == "partial"
    assert incomplete is False


def test_vrram_fit_unavailable_flags_provenance_incomplete():
    fit, incomplete = runner.vrram_fit_from_ps(None, "test-model:latest")
    assert fit is None
    assert incomplete is True

    fit, incomplete = runner.vrram_fit_from_ps({"models": []}, "test-model:latest")
    assert fit is None
    assert incomplete is True


# ---------------------------------------------------------------------------
# 4. transactional insert: one sample fails -> 0 rows inserted
# ---------------------------------------------------------------------------

async def test_sample_failure_inserts_zero_rows(monkeypatch):
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        job_id = make_job(session_local, n_samples=2)
        ctx = make_ctx(job_id)

        calls = {"n": 0}

        def flaky_chat_handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 2:
                return httpx.Response(500, content=b"internal error")
            return httpx.Response(200, content=ndjson_chat_response(
                thinking_deltas=["hi "], content_deltas=["ok"],
            ))

        client = make_client(flaky_chat_handler)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        db = session_local()
        try:
            count = db.query(BenchSample).count()
        finally:
            db.close()
        assert count == 0
    finally:
        cleanup(engine, tmpfile)


# ---------------------------------------------------------------------------
# 5 & 6. provenance-complete rows + prompt_text/collector on every row
# ---------------------------------------------------------------------------

async def test_successful_run_inserts_provenance_complete_rows(monkeypatch):
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        prompt = "Say hello."
        job_id = make_job(session_local, n_samples=2, prompt=prompt)
        ctx = make_ctx(job_id)

        client = make_client(always_ok_chat_handler)
        try:
            await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        db = session_local()
        try:
            rows = db.query(BenchSample).all()
        finally:
            db.close()

        assert len(rows) >= 2  # at least the 2 token-test samples
        for row in rows:
            for field in runner._PROVENANCE_FIELDS:
                assert getattr(row, field) is not None, f"{field} is None on row {row.run_id}"
            assert row.prompt_text == prompt
            assert row.collector == "gui-runner"
    finally:
        cleanup(engine, tmpfile)


# ---------------------------------------------------------------------------
# 7. run_id idempotence
# ---------------------------------------------------------------------------

async def test_insert_samples_idempotent_skips_existing_run_ids(monkeypatch):
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        row = {
            "run_id": "run_fixed_1:tok0",
            "model_tag": "test-model:latest",
            "true_params": 7.6,
            "quant": "Q4_K_M",
            "ctx_len": 4096,
            "think": False,
            "prompt_bytes": 10,
            "temperature": 0.0,
            "seed": 42,
            "ollama_version": "0.32.13",
            "vrram_fit": "fit",
            "output_tokens": 20,
            "thinking_tokens": 0,
            "content_tokens": 5,
            "tokens_per_sec": 40.0,
            "ttft_ms": 100.0,
            "latency_ms": 500.0,
            "created_at": core_db.utcnow_naive(),
            "prompt_text": "hello there",
            "collector": "gui-runner",
        }

        first = await runner.insert_samples_idempotent([row])
        assert first == 1

        second = await runner.insert_samples_idempotent([dict(row)])
        assert second == 0

        db = session_local()
        try:
            count = db.query(BenchSample).filter(BenchSample.run_id == row["run_id"]).count()
        finally:
            db.close()
        assert count == 1
    finally:
        cleanup(engine, tmpfile)


async def test_run_bench_same_job_twice_is_idempotent_at_row_level(monkeypatch):
    """Running the identical row set through insert_samples_idempotent twice
    (the mechanism run_bench relies on for idempotence) inserts nothing the
    second time -- the pre-query skip."""
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        rows = [
            {
                "run_id": f"run_fixed_2:tok{i}",
                "model_tag": "test-model:latest",
                "true_params": 7.6,
                "quant": "Q4_K_M",
                "ctx_len": 4096,
                "think": False,
                "prompt_bytes": 10,
                "temperature": 0.0,
                "seed": 42,
                "ollama_version": "0.32.13",
                "vrram_fit": "fit",
                "output_tokens": 20,
                "thinking_tokens": 0,
                "content_tokens": 5,
                "tokens_per_sec": 40.0,
                "ttft_ms": 100.0,
                "latency_ms": 500.0,
                "created_at": core_db.utcnow_naive(),
                "prompt_text": "hello there",
                "collector": "gui-runner",
            }
            for i in range(2)
        ]

        inserted_first = await runner.insert_samples_idempotent([dict(r) for r in rows])
        assert inserted_first == 2

        inserted_second = await runner.insert_samples_idempotent([dict(r) for r in rows])
        assert inserted_second == 0

        db = session_local()
        try:
            total = db.query(BenchSample).count()
        finally:
            db.close()
        assert total == 2
    finally:
        cleanup(engine, tmpfile)


# ---------------------------------------------------------------------------
# Cancellation: 0 partial rows
# ---------------------------------------------------------------------------

async def test_cancel_before_first_sample_inserts_nothing(monkeypatch):
    from services.modelbench.job_registry import JobCancelled

    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        job_id = make_job(session_local, n_samples=3)
        registry = FakeRegistry()
        registry._cancelled.add(job_id)
        ctx = JobContext(registry, job_id)

        client = make_client(always_ok_chat_handler)
        try:
            with pytest.raises(JobCancelled):
                await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        db = session_local()
        try:
            count = db.query(BenchSample).count()
        finally:
            db.close()
        assert count == 0
    finally:
        cleanup(engine, tmpfile)


# ---------------------------------------------------------------------------
# Unknown model -> raises, registry marks failed
# ---------------------------------------------------------------------------

async def test_unknown_model_raises(monkeypatch):
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        job_id = make_job(session_local, model_tag="does-not-exist:latest", n_samples=1)
        ctx = make_ctx(job_id)

        client = make_client(always_ok_chat_handler)
        try:
            with pytest.raises(ValueError):
                await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        db = session_local()
        try:
            count = db.query(BenchSample).count()
        finally:
            db.close()
        assert count == 0
    finally:
        cleanup(engine, tmpfile)


# ---------------------------------------------------------------------------
# Requested think class is recorded even for content-templated models (R3)
# ---------------------------------------------------------------------------

# Content-templated model: NO native `thinking` capability (like defiant-fable,
# the 27B pair). Its reasoning arrives as templated <think> tags inside content.
TAGS_TEMPLATED = {
    "models": [
        {
            "name": "templated-model:latest",
            "details": {
                "parameter_size": "9.0B",
                "quantization_level": "Q4_K_M",
                "context_length": 262144,
            },
            "capabilities": ["completion"],  # no native thinking
        }
    ]
}


def make_templated_client(chat_handler, *, ps_response=PS_RESPONSE_FIT):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return chat_handler(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=TAGS_TEMPLATED)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps_response)
        if request.url.path == "/api/version":
            return httpx.Response(200, json=VERSION_RESPONSE)
        return httpx.Response(404)

    return httpx.AsyncClient(
        base_url=runner.OLLAMA_URL, transport=httpx.MockTransport(handler)
    )


async def test_content_templated_model_records_requested_think_class(monkeypatch):
    """A think=true run on a content-templated model (no native capability)
    must record think=True (the requested class) and split the templated
    <think> reasoning into the thinking channel -- not silently demote it to
    the think=false bucket (R3: never conflate classes)."""
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        job_id = make_job(session_local, model_tag="templated-model:latest",
                          think=True, n_samples=1)
        ctx = make_ctx(job_id)

        captured_payloads = []

        def templated_chat_handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content))
            return httpx.Response(200, content=ndjson_chat_response(
                thinking_deltas=[],
                content_deltas=["<think>I should reason carefully</think>The answer is 42."],
                eval_count=12, eval_duration=600_000_000,
            ))

        client = make_templated_client(templated_chat_handler)
        try:
            await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        # Payload: think field OMITTED (templated model would reject think:true).
        assert "think" not in captured_payloads[0]
        assert captured_payloads[0]["options"]["temperature"] == 0.0
        assert captured_payloads[0]["options"]["seed"] == 42

        db = session_local()
        try:
            row = db.query(BenchSample).first()
        finally:
            db.close()
        assert row.think is True  # requested class preserved, not conflated
        assert row.thinking_tokens == runner.estimate_tokens("I should reason carefully")
        assert row.content_tokens == runner.estimate_tokens("The answer is 42.")
        assert row.collector == "gui-runner"
        assert row.prompt_text == "Say hello."
    finally:
        cleanup(engine, tmpfile)


# ---------------------------------------------------------------------------
# User-controllable ctx sweep series (MB-Dash-3b)
# ---------------------------------------------------------------------------

async def test_run_bench_uses_provided_ctx_sweep_for_sweep(monkeypatch):
    """A job carrying a user ctx_sweep runs the sweep loop at exactly those
    points (no implicit doubling) -- acceptance (d)."""
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        # n_samples=1 token test, then the user's sweep [512, 768].
        job_id = make_job(session_local, ctx_sweep="[512, 768]", n_samples=1)
        ctx = make_ctx(job_id)

        num_ctx_calls = []

        def capturing_handler(request):
            num_ctx_calls.append(json.loads(request.content)["options"]["num_ctx"])
            return httpx.Response(200, content=ndjson_chat_response(
                thinking_deltas=["thinking a bit "], content_deltas=["ok"],
            ))

        client = make_client(capturing_handler)
        try:
            await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        # First num_ctx is the token test (ctx_target); the rest are the sweep.
        assert num_ctx_calls[1:] == [512, 768]
    finally:
        cleanup(engine, tmpfile)


async def test_run_bench_without_ctx_sweep_uses_doubling_fallback(monkeypatch):
    """A job without ctx_sweep keeps the backward-compatible ctx_sweep_points
    sweep -- acceptance (e)."""
    session_local, engine, tmpfile = make_db(monkeypatch)
    try:
        job_id = make_job(session_local, n_samples=1)  # no ctx_sweep
        ctx = make_ctx(job_id)

        num_ctx_calls = []

        def capturing_handler(request):
            num_ctx_calls.append(json.loads(request.content)["options"]["num_ctx"])
            return httpx.Response(200, content=ndjson_chat_response(
                thinking_deltas=["thinking a bit "], content_deltas=["ok"],
            ))

        client = make_client(capturing_handler)
        try:
            await runner.run_bench(ctx, client=client)
        finally:
            await client.aclose()

        # test-model:latest context_length=2048 -> cap=2048.
        cap = runner.effective_ctx_cap("test-model:latest", 2048)
        assert num_ctx_calls[1:] == runner.ctx_sweep_points(cap)
    finally:
        cleanup(engine, tmpfile)

