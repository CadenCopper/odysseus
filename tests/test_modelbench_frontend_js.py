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
