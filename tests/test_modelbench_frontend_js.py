"""Pin the pure helpers in the modelbench/ frontend module — driven through
`node --input-type=module`, same idiom as tests/test_compare_js.py.

Only state.js, format.js, and api.js are exercised here: they are plain
objects / pure functions with no DOM or fetch calls. index.js (the DOM-heavy
panel) is not imported in node and is left to manual/Playwright verification.
"""

import json
import re
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


# ── markup.js (tooltips + interpretability) ────────────────────────
# markup.js holds the pure DOM-free builders behind the models table and the
# metric blocks (headers with ?-tooltip buttons, the unambiguous think split,
# fit badges, provenance) so the exact strings that ship to the page are
# unit-testable under node like the other pure helpers. index.js itself stays
# out of node (DOM panel), so the pieces that moved here carry the coverage.

# The 11 provenance fields, mirrored from markup.js PROVENANCE_FIELDS and
# routes/modelbench/modelbench_routes.py `_PROVENANCE_FIELDS`.
_EXPECTED_PROVENANCE_FIELDS = [
    "run_id", "model_tag", "true_params", "quant", "ctx_len", "think",
    "prompt_bytes", "temperature", "seed", "ollama_version", "vrram_fit",
]

_TOOLTIP_KEYS = ["think", "provenance", "fit_split", "context",
                 "tokens_per_sec", "ttft", "latency"]

# The aria-label prefix the ?-bubble button must carry: "What is <col>?".
_ARIA_LABEL_PREFIX = "What is "


def _render_models_table_script():
    return textwrap.dedent("""\
        const { modelsTableHtml } = await import('./static/js/modelbench/markup.js');
        const html = modelsTableHtml([{
          model_tag: 'llama3:8b', true_params: '8B', quant: 'Q4_K_M',
          sample_count: 7, think: { false: 2, true: 5 },
          fit_split: { fit: 1, partial: 2, offload: 0 },
          ctx: { advertised: 8192, achieved: 8192, achieved_swept: true },
          provenance: { complete: 7, total: 7 },
        }], null);
        console.log(JSON.stringify({ html }));
    """)


def _render_metric_blocks_script():
    return textwrap.dedent("""\
        const { metricBlockHtml } = await import('./static/js/modelbench/markup.js');
        const fmt = (v) => String(v);
        const sample = { mean: 12.3, min: 1, max: 99, count: 20, percentiles: { p50: 11, p99: 88 } };
        console.log(JSON.stringify({
          tps: metricBlockHtml('Tokens/sec', 'tokens_per_sec', sample, fmt),
          ttft: metricBlockHtml('TTFT (ms)', 'ttft', sample, fmt),
          latency: metricBlockHtml('Latency (ms)', 'latency', sample, fmt),
          no_data: metricBlockHtml('TTFT (ms)', 'ttft', null, fmt),
        }));
    """)


def _extract_data_tooltips(markup):
    return re.findall(r'data-tooltip="([^"]*)"', markup)


# ── think split ────────────────────────────────────────────────────

def test_models_table_think_split_is_labeled_and_unambiguous(node_available):
    out = _run_node(_render_models_table_script())
    html = out["html"]
    # Unambiguous labeled inline form (operator ask #2).
    assert "no 2 · yes 5" in html, "think cell must read 'no N · yes M'"
    # Scope to the think cell (5th <td> of the row); provenance (7/7) legitimately
    # keeps a bare N/M form, so we must not grep the whole row for it.
    cells = re.findall(r"<td[^>]*>(.*?)</td>", html)
    assert cells[4] == "no 2 · yes 5", f"think cell wrong: {cells[4]!r}"
    assert "2 / 5" not in cells[4], "old ambiguous bare split still present"


# ── tooltip markup presence ────────────────────────────────────────

def test_models_table_required_headers_carry_tooltip_spans(node_available):
    out = _run_node(_render_models_table_script())
    thead = out["html"].split("<thead><tr>", 1)[1].split("</tr>", 1)[0]
    ths = re.findall(r"<th[^>]*>(.*?)</th>", thead)

    def th_for(prefix):
        for t in ths:
            if t.strip().startswith(prefix):
                return t
        raise AssertionError(f"no <th> for {prefix!r}: {ths}")

    for label in ("Think (no / yes)", "Fit split",
                  "Context (advertised → achieved)", "Provenance"):
        th = th_for(label)
        assert "modelbench-tip" in th, f"{label!r} header missing the ? trigger"
        assert "data-tooltip=" in th, f"{label!r} header missing data-tooltip"

    # Model / Params / Quant must stay exactly as before (no tooltip noise).
    for label in ("Model", "Params", "Quant", "Samples"):
        assert "data-tooltip=" not in th_for(label), \
            f"{label!r} header must not gain a tooltip (keep as-is)"


def test_metric_labels_carry_tooltip_spans(node_available):
    out = _run_node(_render_metric_blocks_script())
    for key in ("tps", "ttft", "latency"):
        block = out[key]
        label = block.split("modelbench-metric-label")[1].split("</div>")[0]
        assert "modelbench-tip" in label, f"{key} metric label missing ? trigger"
        tips = _extract_data_tooltips(label)
        assert tips and tips[0].strip(), f"{key} metric label tooltip empty"
    # The No-data branch keeps its label tooltip too.
    assert "modelbench-tip" in out["no_data"]


# ── tooltip content ────────────────────────────────────────────────

def test_tooltip_content_explains_columns_and_metrics(node_available):
    script = textwrap.dedent("""\
        const { tipButton } = await import('./static/js/modelbench/markup.js');
        const keys = %s;
        const buttons = {};
        for (const k of keys) buttons[k] = tipButton(k);
        console.log(JSON.stringify({ buttons }));
    """ % json.dumps(_TOOLTIP_KEYS))
    buttons = _run_node(script)["buttons"]
    content = {k: _extract_data_tooltips(v)[0] for k, v in buttons.items()}
    for k in _TOOLTIP_KEYS:
        assert content[k], f"tooltip {k!r} has no data-tooltip content"

    prov = content["provenance"]
    assert str(len(_EXPECTED_PROVENANCE_FIELDS)) in prov, "provenance tooltip must state the field count"
    for f in _EXPECTED_PROVENANCE_FIELDS:
        assert f in prov, f"provenance tooltip must enumerate {f!r}"
    assert "complete/total" in prov, "provenance tooltip must say it renders complete/total"

    fit = content["fit_split"]
    assert "size_vram/size" in fit
    for word in ("fit", "partial", "offload", "12GB", "/api/ps"):
        assert word in fit, f"fit_split tooltip must mention {word!r}"

    ctx = content["context"]
    assert "advertised" in ctx and "achieved" in ctx
    assert "cliff" in ctx, "context tooltip must explain the cliff flag"
    think_lc = content["think"].lower()
    assert "thinking" in think_lc and "reasoning" in think_lc
    assert "output tokens" in content["tokens_per_sec"]
    assert "first output token" in content["ttft"]
    assert "generation time" in content["latency"]


def test_tooltip_trigger_is_focusable_button_with_grounded_aria_label(node_available):
    # The ?-bubble helper must render a real, keyboard-focusable <button
    # type="button"> whose aria-label is grounded: "What is <col>?".
    script = textwrap.dedent("""\
        const { tipButton } = await import('./static/js/modelbench/markup.js');
        const keys = %s;
        const buttons = keys.map((k) => tipButton(k)).join('\\n');
        console.log(JSON.stringify({ buttons }));
    """ % json.dumps(_TOOLTIP_KEYS))
    markup = _run_node(script)["buttons"]
    buttons = re.findall(r"<button[^>]*>", markup)
    assert buttons, "no <button> triggers rendered"
    assert len(buttons) == len(_TOOLTIP_KEYS), f"expected {len(_TOOLTIP_KEYS)} triggers, got {len(buttons)}"
    for b in buttons:
        assert re.search(r'\btype="button"', b), f"trigger must be type=button: {b}"
        assert "modelbench-tip" in b, f"trigger must carry modelbench-tip class: {b}"
        m = re.search(r'aria-label="([^"]*)"', b)
        assert m, f"trigger missing aria-label: {b}"
        assert m.group(1).startswith(_ARIA_LABEL_PREFIX) and m.group(1).endswith("?"), \
            f"aria-label must be 'What is <col>?': {m.group(1)!r}"
    # Keyboard-focusable natively (a real button), no tabindex hack needed but
    # none of the forbidden inline handlers may appear.
    assert re.search(r"\son\w+\s*=\s*['\"]", markup, re.I) is None
    assert "style=" not in markup, "tooltip trigger must carry no inline style"


# ── CSP guard ──────────────────────────────────────────────────────

def test_modelbench_markup_no_inline_handlers_or_scripts(node_available):
    # Everything the tooltip/interpretability change injects (models table,
    # metric blocks, tooltip buttons) must stay inside the CSP nonce contract:
    # no inline on*= event-handler attributes, no <script> elements, no new
    # external script src.
    script = textwrap.dedent("""\
        const { modelsTableHtml, metricBlockHtml, tipButton } = await import('./static/js/modelbench/markup.js');
        const fmt = (v) => String(v);
        const sample = { mean: 12.3, min: 1, max: 99, count: 20, percentiles: { p50: 11, p99: 88 } };
        const table = modelsTableHtml([{
          model_tag: 'llama3:8b', true_params: '8B', quant: 'Q4_K_M',
          sample_count: 7, think: { false: 2, true: 5 },
          fit_split: { fit: 1, partial: 2, offload: 0 },
          ctx: { advertised: 8192, achieved: 8192, achieved_swept: true },
          provenance: { complete: 7, total: 7 },
        }], null);
        const tps = metricBlockHtml('Tokens/sec', 'tokens_per_sec', sample, fmt);
        const ttft = metricBlockHtml('TTFT (ms)', 'ttft', sample, fmt);
        const lat = metricBlockHtml('Latency (ms)', 'latency', sample, fmt);
        const tips = ['think','provenance','fit_split','context','tokens_per_sec','ttft','latency'].map((k) => tipButton(k)).join('');
        console.log(JSON.stringify({ combined: table + tps + ttft + lat + tips }));
    """)
    combined = _run_node(script)["combined"].lower()
    for handler in ("onclick", "onmouseenter", "onmouseleave", "onfocus", "onblur"):
        assert re.search(rf"\b{handler}\s*=", combined) is None, \
            f"inline {handler}= handler injected"
    assert re.search(r"\son\w+\s*=\s*['\"]", combined) is None, "inline on*= handler injected"
    assert "<script" not in combined, "a <script> element was injected"
    assert "http://" not in combined and "https://" not in combined, \
        "external script src injected"


# ── ctx-sweep (runner form) ──────────────────────────────────────
# parseCtxSweep (format.js) / ctxSweepFieldHtml (markup.js) back the
# optional "Context sweep points" runner field wired in index.js
# _onRunnerSubmit. Blank input means "omit ctx_sweep -> backend default
# sweep"; any parse/range/count violation blocks the POST client-side.

def test_ctx_sweep_field_renders_input(node_available):
    script = textwrap.dedent("""
        const { ctxSweepFieldHtml } = await import('./static/js/modelbench/markup.js');
        const html = ctxSweepFieldHtml({ placeholder: '1024,2048,4096,…' });
        console.log(JSON.stringify({ html }));
    """)
    html = _run_node(script)["html"]
    assert 'id="mb-runner-ctx-sweep"' in html
    assert 'Context sweep points' in html
    assert '1024,2048,4096,…' in html
    assert 'id="mb-runner-ctx-sweep-error"' in html
    assert re.search(r"\son\w+\s*=\s*['\"]", html, re.I) is None
    assert "<script" not in html


def test_parse_ctx_sweep_valid(node_available):
    script = textwrap.dedent("""
        const { parseCtxSweep } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          plain: parseCtxSweep("1024,2048,4096", 8192),
          spaced: parseCtxSweep(" 1024, 2048 ,4096 ", 8192),
        }));
    """)
    out = _run_node(script)
    for key in ("plain", "spaced"):
        assert out[key] == {"points": [1024, 2048, 4096], "error": None}, out[key]


def test_parse_ctx_sweep_blank_defaults(node_available):
    script = textwrap.dedent("""
        const { parseCtxSweep } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          empty: parseCtxSweep("", 8192),
          whitespace: parseCtxSweep("   ", 8192),
          nullish: parseCtxSweep(null, 8192),
        }));
    """)
    out = _run_node(script)
    for key in ("empty", "whitespace", "nullish"):
        assert out[key] == {"points": None, "error": None}, out[key]


def test_parse_ctx_sweep_rejects_out_of_range(node_available):
    script = textwrap.dedent("""
        const { parseCtxSweep } = await import('./static/js/modelbench/format.js');
        console.log(JSON.stringify({
          over_cap: parseCtxSweep("1024,99999", 8192),
          zero: parseCtxSweep("0", 8192),
          negative: parseCtxSweep("1024,-5", 8192),
          non_numeric: parseCtxSweep("abc", 8192),
          float_token: parseCtxSweep("1024.5", 8192),
        }));
    """)
    out = _run_node(script)
    for key, val in out.items():
        assert val["points"] is None, f"{key} should reject: {val}"
        assert val["error"] and isinstance(val["error"], str), f"{key} must carry a human-readable error"


def test_parse_ctx_sweep_rejects_too_many_points(node_available):
    script = textwrap.dedent("""
        const { parseCtxSweep } = await import('./static/js/modelbench/format.js');
        const raw = Array.from({ length: 13 }, (_, i) => 1024 + i).join(',');
        console.log(JSON.stringify(parseCtxSweep(raw, 100000)));
    """)
    out = _run_node(script)
    assert out["points"] is None
    assert out["error"]


def test_runner_submit_wires_comma_series_to_params_ctx_sweep(node_available):
    # Source-level wire-contract guard (index.js is DOM-heavy and not importable
    # in node, so we pin the submit mapping + field wiring by source scan — the
    # same technique as the tooltip/source-injection guards below). The GUI must
    # send the comma-separated series as params.ctx_sweep and never the stale
    # ctx_series name (MB-Dash-3b canonical contract).
    src = (_REPO / "static/js/modelbench" / "index.js").read_text(encoding="utf-8")
    assert "params.ctx_sweep = points" in src
    assert "params.ctx_series" not in src
    # The field is wired by id and parsed through parseCtxSweep.
    assert "getElementById('mb-runner-ctx-sweep')" in src
    assert "parseCtxSweep(" in src
    assert "ctx_series" not in src


def test_modelbench_js_source_has_no_script_injection_or_inline_handlers(node_available):
    # Source-level guard: the modelbench frontend files we touch must not grow
    # script injection or inline HTML event-handler attributes (CSP contract).
    jsdir = _REPO / "static/js/modelbench"
    for name in ("index.js", "markup.js", "format.js", "api.js", "state.js", "runner.js"):
        src = (jsdir / name).read_text(encoding="utf-8")
        # Drop full-line // comment lines so prose that merely *mentions*
        # "<script>" (e.g. the CSP note in markup.js) does not trip the check.
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("//"))
        assert "createElement('script')" not in code, name
        assert "document.write" not in code, name
        assert "new Function(" not in code, name
        assert re.search(r"<script\s", code) is None, name
        assert re.search(r"\son\w+\s*=\s*['\"]", code, re.I) is None, name

