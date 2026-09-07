"""Pin the pure helpers in the modelbench/ frontend module — driven through
`node --input-type=module`, same idiom as tests/test_compare_js.py.

Only state.js, format.js, and api.js are exercised here: they are plain
objects / pure functions with no DOM or fetch calls. index.js (the DOM-heavy
panel) is not imported in node and is left to manual/Playwright verification.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_HAS_NODE = shutil.which("node") is not None


@pytest.fixture(scope="module")
def node_available():
    if not _HAS_NODE:
        pytest.skip("node binary not on PATH")


def _run_node(script: str) -> dict:
    res = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=_REPO,
        capture_output=True,
        timeout=15,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(f"node failed:\n{res.stderr}")
    out_lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    if not out_lines:
        raise AssertionError("node produced no stdout")
    return json.loads(out_lines[-1])


# ── state.js ───────────────────────────────────────────────────────

def test_state_reset_preserves_api_base_clears_transient(node_available):
    script = textwrap.dedent("""
        const { default: state, reset } = await import('./static/js/modelbench/state.js');
        state.API_BASE = 'http://x';
        state.isOpen = true;
        state.filters = { fit: 'partial', think: 'true', model: 'llama3' };
        state.models = { models: [1, 2] };
        state.metrics = { groups: [1] };
        state.samples = { rows: [1, 2, 3] };
        reset();
        console.log(JSON.stringify({
          api_base_sticky: state.API_BASE,
          is_open_cleared: state.isOpen,
          filters_cleared: state.filters,
          models_cleared: state.models,
          metrics_cleared: state.metrics,
          samples_cleared: state.samples,
        }));
    """)
    out = _run_node(script)
    assert out == {
        "api_base_sticky": "http://x",
        "is_open_cleared": False,
        "filters_cleared": {"fit": "", "think": "", "model": ""},
        "models_cleared": None,
        "metrics_cleared": None,
        "samples_cleared": None,
    }


# ── format.js ──────────────────────────────────────────────────────

def test_fmt_tps_and_ms_and_ctx_edge_values(node_available):
    script = textwrap.dedent("""
        const { fmtTps, fmtMs, fmtCtx } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          tps_normal: fmtTps(12.345),
          tps_zero: fmtTps(0),
          tps_null: fmtTps(null),
          tps_undefined: fmtTps(undefined),
          ms_whole: fmtMs(120),
          ms_frac: fmtMs(120.55),
          ms_null: fmtMs(null),
          ctx_null: fmtCtx(null),
          ctx_small: fmtCtx(512),
          ctx_thousands: fmtCtx(32768),
        }));
    """)
    out = _run_node(script)
    assert out["tps_normal"] == "12.3"
    assert out["tps_zero"] == "0.0"
    assert out["tps_null"] == "—"
    assert out["tps_undefined"] == "—"
    assert out["ms_whole"] == "120"
    assert out["ms_frac"] == "120.6"
    assert out["ms_null"] == "—"
    assert out["ctx_null"] == "—"
    assert out["ctx_small"] == "512"
    assert out["ctx_thousands"] == "32,768"


def test_ctx_cliff_class(node_available):
    script = textwrap.dedent("""
        const { ctxCliffClass } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          cliff: ctxCliffClass(32768, 16384, true),
          not_swept: ctxCliffClass(32768, null, false),
          ok: ctxCliffClass(32768, 32768, true),
          neutral_null_advertised: ctxCliffClass(null, 16384, true),
        }));
    """)
    out = _run_node(script)
    assert out == {
        "cliff": "cliff",
        "not_swept": "not-swept",
        "ok": "ok",
        "neutral_null_advertised": "",
    }


def test_fit_color_known_and_unknown(node_available):
    script = textwrap.dedent("""
        const { fitColor } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          fit: fitColor('fit'),
          partial: fitColor('partial'),
          offload: fitColor('offload'),
          unknown: fitColor('bogus'),
          nullish: fitColor(null),
        }));
    """)
    out = _run_node(script)
    for key in ("fit", "partial", "offload", "unknown", "nullish"):
        assert isinstance(out[key], str) and out[key].strip(), f"{key} must be a non-empty CSS color string"
    assert len({out["fit"], out["partial"], out["offload"], out["unknown"]}) == 4


def test_bar_pct_guards_div_by_zero(node_available):
    script = textwrap.dedent("""
        const { barPct } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          half: barPct(5, 10),
          over: barPct(15, 10),
          zero_max: barPct(5, 0),
          negative_max: barPct(5, -3),
        }));
    """)
    out = _run_node(script)
    assert out == {"half": 50, "over": 100, "zero_max": 0, "negative_max": 0}


# ── api.js ─────────────────────────────────────────────────────────

def test_models_url(node_available):
    script = textwrap.dedent("""
        const { modelsUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          none: modelsUrl(),
          empty: modelsUrl(''),
          fit: modelsUrl('fit'),
        }));
    """)
    out = _run_node(script)
    assert out["none"] == "/api/modelbench/models"
    assert out["empty"] == "/api/modelbench/models"
    assert out["fit"] == "/api/modelbench/models?fit=fit"


def test_metrics_url(node_available):
    script = textwrap.dedent("""
        const { metricsUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          bare: metricsUrl('llama3:8b'),
          with_think: metricsUrl('llama3:8b', { think: true }),
          with_think_false: metricsUrl('llama3:8b', { think: false }),
          with_fit: metricsUrl('llama3:8b', { fit: 'partial' }),
          custom_percentiles: metricsUrl('llama3:8b', { percentiles: [50, 99] }),
        }));
    """)
    out = _run_node(script)
    assert out["bare"] == "/api/modelbench/models".replace("models", "metrics") + "?model=llama3%3A8b&percentiles=50%2C90%2C95"
    assert out["with_think"] == "/api/modelbench/metrics?model=llama3%3A8b&percentiles=50%2C90%2C95&think=true"
    assert out["with_think_false"] == "/api/modelbench/metrics?model=llama3%3A8b&percentiles=50%2C90%2C95&think=false"
    assert out["with_fit"] == "/api/modelbench/metrics?model=llama3%3A8b&percentiles=50%2C90%2C95&fit=partial"
    assert out["custom_percentiles"] == "/api/modelbench/metrics?model=llama3%3A8b&percentiles=50%2C99"
    for url in out.values():
        assert url.startswith("/api/modelbench")


def test_samples_url_omits_empty_params(node_available):
    script = textwrap.dedent("""
        const { samplesUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          none: samplesUrl(),
          all_empty: samplesUrl({ model: '', think: '', fit: '' }),
          with_model: samplesUrl({ model: 'llama3:8b' }),
          full: samplesUrl({ model: 'llama3:8b', think: 'true', fit: 'fit', limit: 50, offset: 100 }),
        }));
    """)
    out = _run_node(script)
    assert out["none"] == "/api/modelbench/samples"
    assert out["all_empty"] == "/api/modelbench/samples"
    assert out["with_model"] == "/api/modelbench/samples?model=llama3%3A8b"
    assert out["full"] == "/api/modelbench/samples?model=llama3%3A8b&think=true&fit=fit&limit=50&offset=100"
    for url in out.values():
        assert url.startswith("/api/modelbench")


def test_sample_url_appends_run_id_path(node_available):
    script = textwrap.dedent("""
        const { sampleUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          plain: sampleUrl('run-123'),
          needs_encoding: sampleUrl('run/with spaces'),
        }));
    """)
    out = _run_node(script)
    assert out["plain"] == "/api/modelbench/samples/run-123"
    assert out["needs_encoding"] == "/api/modelbench/samples/run%2Fwith%20spaces"
    for url in out.values():
        assert url.startswith("/api/modelbench")


def test_runs_url(node_available):
    script = textwrap.dedent("""
        const { runsUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({ runs: runsUrl() }));
    """)
    out = _run_node(script)
    assert out["runs"] == "/api/modelbench/runs"


def test_run_url_encodes_job_id(node_available):
    script = textwrap.dedent("""
        const { runUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          plain: runUrl('job-123'),
          needs_encoding: runUrl('j/ob'),
        }));
    """)
    out = _run_node(script)
    assert out["plain"] == "/api/modelbench/runs/job-123"
    assert out["needs_encoding"] == "/api/modelbench/runs/j%2Fob"


def test_run_cancel_url_encodes_job_id(node_available):
    script = textwrap.dedent("""
        const { runCancelUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          plain: runCancelUrl('job-123'),
          needs_encoding: runCancelUrl('j/ob'),
        }));
    """)
    out = _run_node(script)
    assert out["plain"] == "/api/modelbench/runs/job-123/cancel"
    assert out["needs_encoding"] == "/api/modelbench/runs/j%2Fob/cancel"


def test_ollama_models_url(node_available):
    script = textwrap.dedent("""
        const { ollamaModelsUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({ url: ollamaModelsUrl() }));
    """)
    out = _run_node(script)
    assert out["url"] == "/api/modelbench/ollama/models"


def test_pull_url_encodes_tag(node_available):
    script = textwrap.dedent("""
        const { pullUrl } = await import('./static/js/modelbench/api.js');
        console.log(JSON.stringify({
          plain: pullUrl('llama3:8b'),
          needs_encoding: pullUrl('my/model:v1'),
        }));
    """)
    out = _run_node(script)
    assert out["plain"] == "/api/modelbench/ollama/models/llama3%3A8b/pull"
    assert out["needs_encoding"] == "/api/modelbench/ollama/models/my%2Fmodel%3Av1/pull"


# ── runner.js ──────────────────────────────────────────────────────
# runner.js holds the runner's decision logic (single-run lock, poll
# terminal-state detection, refresh-on-completion) behind dependency
# injection so it is testable under plain node with no DOM/fetch.

def test_runner_is_terminal_and_progress_pct(node_available):
    script = textwrap.dedent("""
        const { isTerminal, progressPct } = await import('./static/js/modelbench/runner.js');
        console.log(JSON.stringify({
          done: isTerminal('done'),
          failed: isTerminal('failed'),
          cancelled: isTerminal('cancelled'),
          queued: isTerminal('queued'),
          running: isTerminal('running'),
          pct_zero: progressPct({ progress: 0 }),
          pct_half: progressPct({ progress: 0.5 }),
          pct_full: progressPct({ progress: 1 }),
          pct_over: progressPct({ progress: 1.4 }),
          pct_negative: progressPct({ progress: -0.2 }),
          pct_null_job: progressPct(null),
          pct_missing_progress: progressPct({}),
        }));
    """)
    out = _run_node(script)
    assert out["done"] is True
    assert out["failed"] is True
    assert out["cancelled"] is True
    assert out["queued"] is False
    assert out["running"] is False
    assert out["pct_zero"] == 0
    assert out["pct_half"] == 50
    assert out["pct_full"] == 100
    assert out["pct_over"] == 100
    assert out["pct_negative"] == 0
    assert out["pct_null_job"] == 0
    assert out["pct_missing_progress"] == 0


def test_runner_single_run_lock_refuses_concurrent_start(node_available):
    script = textwrap.dedent("""
        const { createRunner } = await import('./static/js/modelbench/runner.js');
        let startCalls = 0;
        const scheduled = [];
        const runner = createRunner({
          startRun: async () => { startCalls++; return { job_id: 'j1', status: 'queued', progress: 0 }; },
          pollRun: async (id) => ({ job_id: id, status: 'running', progress: 0.5 }),
          cancelRun: async (id) => ({ job_id: id, status: 'cancelled' }),
          refresh: async () => {},
          setTimeoutFn: (fn) => { scheduled.push(fn); return scheduled.length; },
          clearTimeoutFn: () => {},
        });
        const job1 = await runner.start({ model_tag: 'a' });
        const job2 = await runner.start({ model_tag: 'a' });
        console.log(JSON.stringify({
          startCalls,
          sameJob: job1.job_id === job2.job_id,
          scheduledCount: scheduled.length,
          isActive: runner.isActive(),
        }));
    """)
    out = _run_node(script)
    assert out == {
        "startCalls": 1,
        "sameJob": True,
        "scheduledCount": 1,
        "isActive": True,
    }


def test_runner_completed_poll_triggers_refresh(node_available):
    script = textwrap.dedent("""
        const { createRunner } = await import('./static/js/modelbench/runner.js');
        let refreshCalls = 0;
        const scheduled = [];
        let pollReturn = { job_id: 'j1', status: 'running', progress: 0.2 };
        const runner = createRunner({
          startRun: async () => ({ job_id: 'j1', status: 'queued', progress: 0 }),
          pollRun: async () => pollReturn,
          cancelRun: async () => ({}),
          refresh: async () => { refreshCalls++; },
          setTimeoutFn: (fn) => { scheduled.push(fn); return scheduled.length; },
          clearTimeoutFn: () => {},
        });
        await runner.start({ model_tag: 'a' });
        pollReturn = { job_id: 'j1', status: 'done', progress: 1 };
        await scheduled[0]();
        console.log(JSON.stringify({
          refreshCalls,
          isActive: runner.isActive(),
          status: runner.getJob().status,
          scheduledCount: scheduled.length,
        }));
    """)
    out = _run_node(script)
    assert out == {
        "refreshCalls": 1,
        "isActive": False,
        "status": "done",
        "scheduledCount": 1,
    }


def test_runner_reduce_pull_event_and_pct(node_available):
    script = textwrap.dedent("""
        const { reducePullEvent, pullProgressPct } = await import('./static/js/modelbench/runner.js');
        let state = { status: '', completed: 0, total: 0, done: false, ok: null, error: null };
        state = reducePullEvent(state, { event: 'started', size_bytes: 4000000000, max_pull_size_gb: 8 });
        const afterStarted = { ...state, pct: pullProgressPct(state) };
        state = reducePullEvent(state, { status: 'downloading', completed: 50, total: 100 });
        const afterProgress = { ...state, pct: pullProgressPct(state) };
        state = reducePullEvent(state, { event: 'done', ok: true });
        const afterDone = { ...state, pct: pullProgressPct(state) };
        console.log(JSON.stringify({
          afterStartedStatus: afterStarted.status,
          afterStartedPct: afterStarted.pct,
          afterProgressPct: afterProgress.pct,
          afterDoneFlags: { done: afterDone.done, ok: afterDone.ok },
          zeroTotalPct: pullProgressPct({ completed: 0, total: 0 }),
          nullStatePct: pullProgressPct(null),
        }));
    """)
    out = _run_node(script)
    assert out == {
        "afterStartedStatus": "downloading",
        "afterStartedPct": 0,
        "afterProgressPct": 50,
        "afterDoneFlags": {"done": True, "ok": True},
        "zeroTotalPct": 0,
        "nullStatePct": 0,
    }


def test_runner_cancel_path_returns_to_inactive(node_available):
    script = textwrap.dedent("""
        const { createRunner } = await import('./static/js/modelbench/runner.js');
        const scheduled = [];
        let status = 'running';
        let cancelCalls = 0;
        const runner = createRunner({
          startRun: async () => ({ job_id: 'j1', status: 'queued', progress: 0 }),
          pollRun: async (id) => ({ job_id: id, status, progress: 0.3 }),
          cancelRun: async (id) => { cancelCalls++; status = 'cancelled'; return { job_id: id, status }; },
          refresh: async () => {},
          setTimeoutFn: (fn) => { scheduled.push(fn); return scheduled.length; },
          clearTimeoutFn: () => {},
        });
        await runner.start({ model_tag: 'a' });
        const activeBefore = runner.isActive();
        await runner.cancel();
        console.log(JSON.stringify({
          cancelCalls,
          activeBefore,
          activeAfter: runner.isActive(),
          statusAfter: runner.getJob().status,
        }));
    """)
    out = _run_node(script)
    assert out == {
        "cancelCalls": 1,
        "activeBefore": True,
        "activeAfter": False,
        "statusAfter": "cancelled",
    }
