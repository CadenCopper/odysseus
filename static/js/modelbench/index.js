// modelbench/index.js — read-only ModelBench measurement dashboard.
// Mounted as a draggable/resizable/z-ordered tool window on the app shell
// (notes-style pane + backdrop on document.body), NOT a fullscreen panel.
// Own close button + Escape, no modalManager registration since the window
// has no minimize-to-chip state — closing removes it entirely.

import state, { reset } from './state.js';
import { fmtTps, fmtMs, fmtCtx, fitColor } from './format.js';
import { modelsTableHtml, metricBlockHtml } from './markup.js';
import { modelsUrl, metricsUrl, samplesUrl, sampleUrl, runsUrl, runUrl, runCancelUrl, ollamaModelsUrl, pullUrl } from './api.js';
import { createRunner, isTerminal, progressPct, pullProgressPct } from './runner.js';
import { makeWindowDraggable } from '../windowDrag.js';
import { topToolWindowZ } from '../toolWindowZOrder.js';
import { applyEdgeDock, clearDockSide } from '../modalSnap.js';
import uiModule from '../ui.js';

const escapeHtml = uiModule.esc;

let _panelEl = null;
let _backdropEl = null;
let _keydownHandler = null;

// Float-geometry persistence (FAIL #3): the window remembers its last
// FLOATING position+size so a close→reopen restores it instead of always
// re-docking right at the default. windowResize keeps the size under
// `winsize-modelbench-panel`; this key carries the full float rect so the
// reopen path has an exact left/top/width/height to restore.
const MB_FLOAT_RECT_KEY = 'modelbench-float-rect';
// Rect captured when entering fullscreen, so drag-down "unsnaps" back to the
// windowed geometry the window had before the top-edge snap.
let _fsPreRect = null;

// ────────────────────────────────────────────────────────────────────────────
// ── fetch helper ──
// ────────────────────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  let data = {};
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) {
    const err = new Error(data.detail || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/** Streamed SSE pull: yields one parsed JSON object per `data: ` line. */
async function* _streamPull(tag, confirm) {
  const res = await fetch(state.API_BASE + pullUrl(tag), {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ confirm: !!confirm }),
  });
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (_) { /* non-JSON error body */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = chunk.split('\n').find((l) => l.startsWith('data: '));
      if (!line) continue;
      try {
        yield JSON.parse(line.slice(6));
      } catch (_) { /* malformed frame */ }
    }
  }
}

const _runner = createRunner({
  startRun: (params) => postJSON(state.API_BASE + runsUrl(), params),
  pollRun: (jobId) => fetchJSON(state.API_BASE + runUrl(jobId)),
  cancelRun: (jobId) => postJSON(state.API_BASE + runCancelUrl(jobId), {}),
  startPull: (tag, confirm) => _streamPull(tag, confirm),
  refresh: () => refresh(),
  onJobChange: _renderRunnerJob,
  onPullChange: _renderRunnerPull,
});

let _runnerThink = false;

// ────────────────────────────────────────────────────────────────────────────
// ── public API ──
// ────────────────────────────────────────────────────────────────────────────

function init(apiBase) {
  state.API_BASE = apiBase;
}

function isActive() {
  return state.isOpen;
}

function open() {
  if (state.isOpen) {
    if (_panelEl) _panelEl.focus?.();
    return;
  }
  state.isOpen = true;
  _setRailActive(true);
  _mountPanel();
  document.body.classList.add('modelbench-open');
  _keydownHandler = (e) => {
    if (e.key !== 'Escape') return;
    const backdrop = document.getElementById('mb-drilldown-backdrop');
    if (backdrop && backdrop.style.display !== 'none') {
      _closeDrilldown();
      return;
    }
    close();
  };
  document.addEventListener('keydown', _keydownHandler);
  refresh();
  _loadOllamaModels();
}

function close() {
  if (!state.isOpen) return;
  if (_keydownHandler) {
    document.removeEventListener('keydown', _keydownHandler);
    _keydownHandler = null;
  }
  _setRailActive(false);
  // Snapshot the current floating geometry (no-op while docked/fullscreen, so
  // the last genuinely-floated rect is what a reopen restores).
  _saveModelbenchFloat(_panelEl);
  if (_backdropEl) {
    _backdropEl.remove();
    _backdropEl = null;
  } else if (_panelEl) {
    _panelEl.remove();
  }
  _panelEl = null;
  document.body.classList.remove('modelbench-open');
  reset();
}

function _setRailActive(on) {
  const btn = document.getElementById('tool-modelbench-btn');
  if (btn) btn.classList.toggle('active', on);
}

async function refresh() {
  _hideError();
  await Promise.all([_loadModels(), _loadSamples()]);
}

const modelbenchModule = { init, open, close, isActive, refresh };
export default modelbenchModule;
window.modelbenchModule = modelbenchModule;

// ────────────────────────────────────────────────────────────────────────────
// ── panel mounting ──
// ────────────────────────────────────────────────────────────────────────────

function _mountPanel() {
  if (_panelEl) return _panelEl;

  const panel = document.createElement('div');
  panel.className = 'modelbench-panel';
  panel.id = 'modelbench-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'ModelBench dashboard');
  panel.innerHTML = `
    <div class="modelbench-header">
      <div class="modelbench-header-left">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
        <h4 class="modelbench-title">ModelBench</h4>
      </div>
      <div class="modelbench-controls">
        <label class="modelbench-filter-label" for="mb-filter-fit">Fit</label>
        <select id="mb-filter-fit" aria-label="Filter by fit class">
          <option value="">All</option>
          <option value="fit">Fit</option>
          <option value="partial">Partial</option>
          <option value="offload">Offload</option>
        </select>
        <div class="modelbench-think-toggle" id="mb-think-toggle" role="group" aria-label="Filter by think mode">
          <button type="button" data-think="" class="active">Both</button>
          <button type="button" data-think="true">Think</button>
          <button type="button" data-think="false">No-think</button>
        </div>
        <button type="button" class="modelbench-close-btn" id="modelbench-close-btn" title="Close ModelBench" aria-label="Close ModelBench">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>
    </div>
    <div class="modelbench-banner" id="mb-error-banner" role="alert" style="display:none"></div>
    <div class="modelbench-body">
      <section class="modelbench-section modelbench-runner-section" id="mb-runner-section">
        <h5 class="modelbench-section-title">Runner</h5>
        <div class="modelbench-runner-row">
          <label class="modelbench-runner-label" for="mb-runner-model">Model</label>
          <select id="mb-runner-model" aria-label="Select model to run">
            <option value="">Select a model…</option>
          </select>
          <button type="button" id="mb-runner-pull-btn" class="modelbench-runner-btn" style="display:none">Pull</button>
        </div>
        <div id="mb-runner-model-detail" class="modelbench-runner-model-detail"></div>
        <div id="mb-runner-pull-progress" class="modelbench-runner-pull-progress" style="display:none"></div>
        <form id="mb-runner-form" class="modelbench-runner-form">
          <div class="modelbench-runner-row">
            <div class="modelbench-think-toggle" id="mb-runner-think-toggle" role="group" aria-label="Think mode for this run">
              <button type="button" data-think="false" class="active">No-think</button>
              <button type="button" data-think="true">Think</button>
            </div>
            <label class="modelbench-runner-label" for="mb-runner-ctx">Ctx target</label>
            <input type="number" id="mb-runner-ctx" min="1" placeholder="optional" />
            <label class="modelbench-runner-label" for="mb-runner-n">Samples</label>
            <input type="number" id="mb-runner-n" min="1" max="20" value="5" />
          </div>
          <textarea id="mb-runner-prompt" class="modelbench-runner-prompt" rows="2" aria-label="Prompt">Write a haiku about context windows</textarea>
          <div class="modelbench-runner-row modelbench-runner-row-end">
            <span id="mb-runner-prompt-bytes" class="modelbench-runner-meta"></span>
            <button type="button" id="mb-runner-cancel-btn" class="modelbench-runner-btn" style="display:none">Cancel</button>
            <button type="submit" id="mb-runner-run-btn" class="modelbench-runner-btn modelbench-runner-btn-primary">Run</button>
          </div>
        </form>
        <div id="mb-runner-job-panel" class="modelbench-runner-job-panel" style="display:none"></div>
      </section>
      <section class="modelbench-section">
        <h5 class="modelbench-section-title">Models</h5>
        <div class="modelbench-fit-legend" aria-hidden="true">
          <span><span class="modelbench-legend-dot" style="background:${fitColor('fit')}"></span>fit</span>
          <span><span class="modelbench-legend-dot" style="background:${fitColor('partial')}"></span>partial</span>
          <span><span class="modelbench-legend-dot" style="background:${fitColor('offload')}"></span>offload</span>
        </div>
        <div id="mb-models-table-wrap"><div class="modelbench-loading">Loading…</div></div>
      </section>
      <section class="modelbench-section" id="mb-metrics-section" style="display:none">
        <h5 class="modelbench-section-title" id="mb-metrics-title">Metrics</h5>
        <div id="mb-metrics-body"></div>
      </section>
      <section class="modelbench-section">
        <h5 class="modelbench-section-title">Samples</h5>
        <div id="mb-samples-table-wrap"><div class="modelbench-loading">Loading…</div></div>
      </section>
    </div>
    <div class="modelbench-drilldown-backdrop" id="mb-drilldown-backdrop" style="display:none" role="dialog" aria-modal="true" aria-label="Sample provenance detail">
      <div class="modelbench-drilldown-modal" id="mb-drilldown-modal"></div>
    </div>
  `;

  const backdrop = document.createElement('div');
  backdrop.className = 'modelbench-panel-backdrop';
  backdrop.id = 'modelbench-panel-backdrop';
  // Backdrop click-to-dismiss (notes/tool-window convention, notes.js
  // L1236-38): clicking the outer backdrop surface — i.e. OUTSIDE the panel —
  // closes the window. close() removes the backdrop element entirely, so a
  // backdrop dismiss leaves no stray backdrop / pointer trap behind.
  backdrop.addEventListener('click', (ev) => {
    if (ev.target === backdrop) close();
  });

  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);
  _panelEl = panel;
  _backdropEl = backdrop;

  _wirePanelEvents(panel);
  _wireRunnerEvents(panel);
  _renderRunnerJob(_runner.getJob());
  _renderRunnerPull(_runner.getPullState());
  _wireModelbenchWindow(panel);
  if (window.innerWidth > 768) {
    // Restore the last floating geometry when one was saved; only dock right
    // (the default) when the user has never floated the window.
    if (!_restoreModelbenchFloatGeometry(panel)) {
      _restoreModelbenchDock(panel);
    }
  } else {
    _applyModelbenchMobileSheet(panel);
  }
  _bringModelbenchToFront(panel);
  return panel;
}

// ── draggable tool-window wiring (notes-style pane + backdrop) ──
function _wireModelbenchWindow(pane) {
  if (!pane || pane.dataset.windowDragWired === '1') return;
  const header = pane.querySelector('.modelbench-header');
  if (!header) return;
  pane.dataset.windowDragWired = '1';

  const enterFs = () => {
    // Remember the rect we're leaving so a drag-down unsnap can restore it
    // instead of dropping the window into an arbitrary spot.
    try {
      const r = pane.getBoundingClientRect();
      if (r.width && r.height) {
        _fsPreRect = { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) };
      }
    } catch (_) {}
    // Drop any stale dock/floating inline geometry first so the CSS
    // `.modelbench-window-fullscreen` rule (fixed; inset:0) takes over cleanly.
    _clearModelbenchSnapStyles(pane);
    pane.classList.add('modelbench-window-fullscreen');
  };
  const exitFs = () => {
    const pre = _fsPreRect;
    _fsPreRect = null;
    if (pre) _applyModelbenchFloatRect(pane, pre); // unsnap to the pre-fullscreen windowed geometry
    else _restoreModelbenchDock(pane);
  };

  makeWindowDraggable(pane, {
    content: pane,
    header,
    fsClass: 'modelbench-window-fullscreen',
    skipSelector: 'button, input, select, textarea, .modelbench-close-btn, #mb-think-toggle',
    enableDock: true,
    enableLeftDock: true,
    onEnterFullscreen: enterFs,
    onExitFullscreen: exitFs,
    onDragEnd: () => _saveModelbenchFloat(pane),
  });

  // Bring the window to the front on header pointer/focus interaction.
  const bring = () => _bringModelbenchToFront(pane);
  pane.addEventListener('pointerdown', bring, true);
  pane.addEventListener('focusin', bring, true);
  header.addEventListener('click', bring);
  header.addEventListener('focus', bring);
}

function _clearModelbenchSnapStyles(pane) {
  if (!pane) return;
  const hadLeft = pane.classList.contains('modal-left-docked');
  const hadRight = pane.classList.contains('modal-right-docked');
  pane.classList.remove('modelbench-window-fullscreen', 'modal-left-docked', 'modal-right-docked');
  if (hadLeft) clearDockSide('left', pane);
  if (hadRight) clearDockSide('right', pane);
  ['position', 'left', 'top', 'right', 'bottom', 'width', 'max-width', 'height',
    'max-height', 'margin', 'transform', 'border-radius']
    .forEach((prop) => pane.style.removeProperty(prop));
  delete pane.dataset._tilePreSnap;
  delete pane.dataset._tileZone;
  delete pane._preDockSnapshot;
  delete pane._dockSide;
  delete pane._dockSuspended;
}

function _restoreModelbenchDock(pane) {
  if (!pane || window.innerWidth <= 768) return;
  _clearModelbenchSnapStyles(pane);
  if (!pane.isConnected) return;
  applyEdgeDock(pane, 'right');
}

// ── floating-rect persistence (FAIL #3) ────────────────────────────────────
// Reads the last saved float rect, or null when none exists (fresh user).
function _readModelbenchFloat() {
  if (typeof localStorage === 'undefined') return null;
  try {
    const g = JSON.parse(localStorage.getItem(MB_FLOAT_RECT_KEY) || 'null');
    if (!g) return null;
    if (![g.x, g.y, g.w, g.h].every((n) => Number.isFinite(n) && n > 0)) return null;
    return g;
  } catch (_) {
    return null;
  }
}

// Save the current floating geometry. No-op while docked/fullscreen/mobile so
// the "last genuinely floated" rect survives a docked close (the window is
// still re-openable docked; the float is what persistence restores).
function _saveModelbenchFloat(pane) {
  if (!pane || !pane.isConnected) return;
  if (window.innerWidth <= 768) return;
  const docked = pane.classList.contains('modal-right-docked')
    || pane.classList.contains('modal-left-docked')
    || pane.classList.contains('modelbench-window-fullscreen');
  if (docked) return;
  let r;
  try { r = pane.getBoundingClientRect(); } catch (_) { return; }
  if (!r.width || !r.height) return;
  const g = { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) };
  try {
    localStorage.setItem(MB_FLOAT_RECT_KEY, JSON.stringify(g));
    // Keep the shared windowResize key in sync so its deferred on-open restore
    // (storageKey 'winsize-<id>') agrees with the rect we restore instead of
    // fighting it with a stale size.
    localStorage.setItem('winsize-modelbench-panel', JSON.stringify({ w: g.w, h: g.h }));
  } catch (_) {}
}

// Apply a float rect as inline fixed geometry, clamping to the viewport so a
// restored window is never left off-screen.
function _applyModelbenchFloatRect(pane, g) {
  if (!pane || !g) return;
  _clearModelbenchSnapStyles(pane);
  if (!pane.isConnected) return;
  const vw = window.innerWidth, vh = window.innerHeight;
  // Floors mirror makeWindowResizable's MIN_W/MIN_H so a legitimately small
  // saved window isn't inflated on restore; only off-screen geometry is clamped.
  const minW = 320, minH = 200;
  const w = Math.max(minW, Math.min(g.w || 486, vw - 12));
  const h = Math.max(minH, Math.min(g.h || 577, vh - 12));
  const x = Math.max(4, Math.min(g.x ?? (vw - w - 4), vw - w - 4));
  const y = Math.max(4, Math.min(g.y ?? 4, vh - h - 4));
  pane.style.position = 'fixed';
  pane.style.left = x + 'px';
  pane.style.top = y + 'px';
  pane.style.width = w + 'px';
  pane.style.height = h + 'px';
  pane.style.maxWidth = 'none';
  pane.style.maxHeight = 'none';
}

// Restore the last saved float geometry on open. Returns true when a float was
// restored (callers should NOT then dock-right); false when the user has no
// saved float (fall through to the default right-dock).
function _restoreModelbenchFloatGeometry(pane) {
  const g = _readModelbenchFloat();
  if (!g) return false;
  _applyModelbenchFloatRect(pane, g);
  return true;
}

function _applyModelbenchMobileSheet(pane) {
  if (!pane) return;
  pane.style.position = 'fixed';
  pane.style.left = '0';
  pane.style.right = '0';
  pane.style.top = 'auto';
  pane.style.bottom = '0';
  pane.style.width = '100%';
  pane.style.maxWidth = '100%';
  pane.style.height = '92vh';
  pane.style.maxHeight = '92vh';
  pane.style.borderRadius = '14px 14px 0 0';
}

// The window's own stacking surface participates in the shared tool-window
// z-order (topToolWindowZ scans body > .modelbench-panel-backdrop).
function _bringModelbenchToFront(pane = document.getElementById('modelbench-panel')) {
  if (!pane) return;
  const backdrop = document.getElementById('modelbench-panel-backdrop') || pane.parentElement;
  const z = topToolWindowZ({ exclude: backdrop }) + 1;
  if (backdrop) backdrop.style.setProperty('z-index', String(z), 'important');
  try {
    window.dispatchEvent(new CustomEvent('odysseus:modal-opened', {
      detail: { id: 'modelbench-panel', modal: pane },
    }));
  } catch (_) {}
}

function _wirePanelEvents(panel) {
  panel.querySelector('#modelbench-close-btn')?.addEventListener('click', () => close());

  panel.querySelector('#mb-filter-fit')?.addEventListener('change', (e) => {
    state.filters.fit = e.target.value;
    refresh();
    if (state.filters.model) _loadMetricsForModel(state.filters.model);
  });

  const thinkToggle = panel.querySelector('#mb-think-toggle');
  thinkToggle?.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-think]');
    if (!btn) return;
    state.filters.think = btn.dataset.think;
    thinkToggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === btn));
    refresh();
    if (state.filters.model) _loadMetricsForModel(state.filters.model);
  });

  const modelsWrap = panel.querySelector('#mb-models-table-wrap');
  if (modelsWrap) _wireRowActivation(modelsWrap, '.modelbench-model-row', (row) => _loadMetricsForModel(row.dataset.model));

  const samplesWrap = panel.querySelector('#mb-samples-table-wrap');
  if (samplesWrap) _wireRowActivation(samplesWrap, '.modelbench-sample-row', (row) => _onSampleRowClick(row.dataset.runId));

  const backdrop = panel.querySelector('#mb-drilldown-backdrop');
  backdrop?.addEventListener('click', (e) => {
    if (e.target === backdrop) _closeDrilldown();
  });
  panel.querySelector('#mb-drilldown-modal')?.addEventListener('click', (e) => {
    if (e.target.closest('#mb-drilldown-close')) _closeDrilldown();
  });
}

/** Wire click + Enter/Space keyboard activation for delegated rows (a11y). */
function _wireRowActivation(wrap, rowSelector, onActivate) {
  wrap.addEventListener('click', (e) => {
    const row = e.target.closest(rowSelector);
    if (row) onActivate(row);
  });
  wrap.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const row = e.target.closest(rowSelector);
    if (!row) return;
    e.preventDefault();
    onActivate(row);
  });
}

// ────────────────────────────────────────────────────────────────────────────
// ── error banner ──
// ────────────────────────────────────────────────────────────────────────────

function _showError(msg) {
  const banner = document.getElementById('mb-error-banner');
  if (!banner) return;
  banner.textContent = msg;
  banner.style.display = '';
}

function _hideError() {
  const banner = document.getElementById('mb-error-banner');
  if (banner) banner.style.display = 'none';
}

// ────────────────────────────────────────────────────────────────────────────
// ── runner ──
// ────────────────────────────────────────────────────────────────────────────

function _wireRunnerEvents(panel) {
  panel.querySelector('#mb-runner-form')?.addEventListener('submit', _onRunnerSubmit);
  panel.querySelector('#mb-runner-cancel-btn')?.addEventListener('click', _onRunnerCancelClick);
  panel.querySelector('#mb-runner-pull-btn')?.addEventListener('click', _onRunnerPullClick);
  panel.querySelector('#mb-runner-model')?.addEventListener('change', _onRunnerModelSelected);
  panel.querySelector('#mb-runner-prompt')?.addEventListener('input', _updatePromptBytes);

  const thinkToggle = panel.querySelector('#mb-runner-think-toggle');
  thinkToggle?.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-think]');
    if (!btn) return;
    _runnerThink = btn.dataset.think === 'true';
    thinkToggle.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === btn));
  });

  _updatePromptBytes();
}

async function _loadOllamaModels() {
  try {
    const data = await fetchJSON(state.API_BASE + ollamaModelsUrl());
    state.residentModels = data;
    _renderRunnerModelSelect();
  } catch (err) {
    _showError('Failed to load ollama models: ' + err.message);
  }
}

function _renderRunnerModelSelect() {
  const select = document.getElementById('mb-runner-model');
  if (!select) return;
  const resident = (state.residentModels && state.residentModels.models) || [];
  const residentNames = new Set(resident.map((m) => m.name));
  const prev = select.value;
  let html = '<option value="">Select a model…</option>';
  for (const m of resident) {
    html += `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name)}${m.has_samples ? ' ✓' : ''}</option>`;
  }
  const benchOnly = ((state.models && state.models.models) || []).filter((m) => !residentNames.has(m.model_tag));
  for (const m of benchOnly) {
    html += `<option value="${escapeHtml(m.model_tag)}">${escapeHtml(m.model_tag)} (not resident)</option>`;
  }
  select.innerHTML = html;
  if (prev && [...select.options].some((o) => o.value === prev)) select.value = prev;
  _onRunnerModelSelected();
}

function _selectedResidentModel() {
  const select = document.getElementById('mb-runner-model');
  if (!select || !select.value) return null;
  const resident = (state.residentModels && state.residentModels.models) || [];
  return resident.find((m) => m.name === select.value) || null;
}

function _onRunnerModelSelected() {
  const select = document.getElementById('mb-runner-model');
  const detail = document.getElementById('mb-runner-model-detail');
  const pullBtn = document.getElementById('mb-runner-pull-btn');
  const ctxInput = document.getElementById('mb-runner-ctx');
  const tag = select?.value || '';
  const resident = _selectedResidentModel();
  if (detail) {
    if (!tag) {
      detail.innerHTML = '';
    } else if (resident) {
      detail.innerHTML = '<span class="modelbench-runner-badge modelbench-runner-badge-resident">Resident</span> ' +
        `${escapeHtml(String(resident.parameter_size ?? '—'))} · ${escapeHtml(String(resident.quantization_level ?? '—'))} · ` +
        `${fmtCtx(resident.context_length)} ctx · ${resident.size_gb ?? '—'} GB`;
    } else {
      detail.innerHTML = '<span class="modelbench-runner-badge modelbench-runner-badge-missing">Not resident</span> bench-sampled only';
    }
  }
  if (ctxInput) ctxInput.placeholder = resident?.context_length ? `max ${resident.context_length}` : 'optional';
  if (pullBtn) pullBtn.style.display = tag && !resident ? '' : 'none';
}

function _updatePromptBytes() {
  const ta = document.getElementById('mb-runner-prompt');
  const out = document.getElementById('mb-runner-prompt-bytes');
  if (!ta || !out) return;
  const bytes = new TextEncoder().encode(ta.value || '').length;
  out.textContent = `${bytes} bytes`;
}

async function _onRunnerSubmit(e) {
  e.preventDefault();
  const select = document.getElementById('mb-runner-model');
  const ctxInput = document.getElementById('mb-runner-ctx');
  const nInput = document.getElementById('mb-runner-n');
  const promptTa = document.getElementById('mb-runner-prompt');
  const modelTag = select?.value || '';
  if (!modelTag) {
    _showError('Select a model to run');
    return;
  }
  const params = {
    model_tag: modelTag,
    prompt: promptTa?.value || '',
    think: _runnerThink,
    n_samples: nInput?.value ? parseInt(nInput.value, 10) : 5,
  };
  const ctxVal = ctxInput?.value ? parseInt(ctxInput.value, 10) : NaN;
  if (Number.isFinite(ctxVal)) params.ctx_target = ctxVal;
  try {
    await _runner.start(params);
  } catch (err) {
    _showError('Failed to start run: ' + err.message);
  }
}

async function _onRunnerCancelClick() {
  try {
    await _runner.cancel();
  } catch (err) {
    _showError('Failed to cancel run: ' + err.message);
  }
}

async function _onRunnerPullClick() {
  const select = document.getElementById('mb-runner-model');
  const tag = select?.value || '';
  if (!tag) return;
  await _tryPull(tag, false);
}

async function _tryPull(tag, confirm) {
  const result = await _runner.pull(tag, { confirm });
  if (!result) return; // locked out: a run or another pull is active
  if (result.errorStatus === 400 && !confirm) {
    if (window.confirm(`${tag} exceeds the VRAM safety threshold. Pull anyway?`)) {
      await _tryPull(tag, true);
    }
    return;
  }
  if (result.error) {
    _showError('Pull failed: ' + result.error);
  }
}

function _setRunnerFormBusy(busy) {
  const runBtn = document.getElementById('mb-runner-run-btn');
  const cancelBtn = document.getElementById('mb-runner-cancel-btn');
  if (runBtn) runBtn.disabled = busy;
  if (cancelBtn) cancelBtn.style.display = busy ? '' : 'none';
}

function _renderRunnerJob(job) {
  const panel = document.getElementById('mb-runner-job-panel');
  if (!panel) return;
  if (!job) {
    panel.style.display = 'none';
    panel.innerHTML = '';
    _setRunnerFormBusy(false);
    return;
  }
  panel.style.display = '';
  const pct = progressPct(job);
  const terminal = isTerminal(job.status);
  const statusClass = job.status === 'failed' || job.status === 'cancelled'
    ? 'modelbench-runner-status-bad'
    : job.status === 'done' ? 'modelbench-runner-status-ok' : '';
  panel.innerHTML =
    `<div class="modelbench-runner-job-status ${statusClass}">${escapeHtml(job.status)}</div>` +
    `<div class="modelbench-bar-track modelbench-runner-progress-track"><span class="modelbench-bar-fill" style="width:${pct}%"></span></div>` +
    (job.message ? `<div class="modelbench-runner-job-message">${escapeHtml(job.message)}</div>` : '') +
    (job.run_id ? `<div class="modelbench-runner-job-meta">run_id: ${escapeHtml(job.run_id)}</div>` : '') +
    (job.error ? `<div class="modelbench-error-inline">${escapeHtml(job.error)}</div>` : '');
  _setRunnerFormBusy(!terminal);
}

function _renderRunnerPull(pullState) {
  const wrap = document.getElementById('mb-runner-pull-progress');
  const pullBtn = document.getElementById('mb-runner-pull-btn');
  if (!wrap) return;
  if (!pullState) {
    wrap.style.display = 'none';
    wrap.innerHTML = '';
    return;
  }
  wrap.style.display = '';
  const pct = pullProgressPct(pullState);
  wrap.innerHTML =
    `<div class="modelbench-runner-pull-status">${escapeHtml(pullState.status || 'pulling')}` +
    (pullState.error ? `: ${escapeHtml(pullState.error)}` : '') + '</div>' +
    `<div class="modelbench-bar-track modelbench-runner-progress-track"><span class="modelbench-bar-fill" style="width:${pct}%"></span></div>`;
  if (pullBtn) pullBtn.disabled = !pullState.done;
  if (pullState.done && pullState.ok) _loadOllamaModels();
}

// ────────────────────────────────────────────────────────────────────────────
// ── models table ──
// ────────────────────────────────────────────────────────────────────────────

async function _loadModels() {
  const wrap = document.getElementById('mb-models-table-wrap');
  if (wrap) wrap.innerHTML = '<div class="modelbench-loading">Loading…</div>';
  try {
    const data = await fetchJSON(state.API_BASE + modelsUrl(state.filters.fit));
    state.models = data;
    _renderModelsTable(data);
    _renderRunnerModelSelect();
  } catch (err) {
    _showError('Failed to load models: ' + err.message);
    if (wrap) wrap.innerHTML = '';
  }
}

function _renderModelsTable(data) {
  const wrap = document.getElementById('mb-models-table-wrap');
  if (!wrap) return;
  const models = data.models || [];
  if (models.length === 0) {
    wrap.innerHTML = '<div class="modelbench-empty">No samples yet</div>';
    return;
  }
  // Table markup — headers with ?-tooltips, the unambiguous labeled think
  // split, fit badges, ctx column and provenance — is built in markup.js so
  // the exact shipped HTML is unit-testable under plain node.
  wrap.innerHTML = modelsTableHtml(models, state.filters.model);
}

// ────────────────────────────────────────────────────────────────────────────
// ── metrics (per-model, per-think-bucket bars) ──
// ────────────────────────────────────────────────────────────────────────────

async function _loadMetricsForModel(tag) {
  state.filters.model = tag;
  document.querySelectorAll('.modelbench-model-row').forEach((r) => {
    r.classList.toggle('modelbench-row-active', r.dataset.model === tag);
  });
  const section = document.getElementById('mb-metrics-section');
  const title = document.getElementById('mb-metrics-title');
  const body = document.getElementById('mb-metrics-body');
  if (section) section.style.display = '';
  if (title) title.textContent = 'Metrics — ' + tag;
  if (body) body.innerHTML = '<div class="modelbench-loading">Loading…</div>';
  try {
    const think = state.filters.think === '' ? undefined : state.filters.think === 'true';
    const url = metricsUrl(tag, { think, fit: state.filters.fit || undefined });
    const data = await fetchJSON(state.API_BASE + url);
    state.metrics = data;
    _renderMetrics(data);
  } catch (err) {
    if (body) body.innerHTML = `<div class="modelbench-error-inline">Failed to load metrics: ${escapeHtml(err.message)}</div>`;
  }
}

function _renderMetrics(data) {
  const body = document.getElementById('mb-metrics-body');
  if (!body) return;
  const groups = data.groups || [];
  if (groups.length === 0) {
    body.innerHTML = '<div class="modelbench-empty">No samples yet</div>';
    return;
  }
  let html = '';
  for (const g of groups) {
    const label = g.think === true ? 'Think' : g.think === false ? 'No-think' : 'Unknown';
    html += '<div class="modelbench-metric-group">' +
      `<h6 class="modelbench-metric-group-title">${escapeHtml(label)} <span class="modelbench-metric-count">(${g.count} samples)</span></h6>` +
      metricBlockHtml('Tokens/sec', 'tokens_per_sec', g.metrics.tokens_per_sec, fmtTps) +
      metricBlockHtml('TTFT (ms)', 'ttft', g.metrics.ttft_ms, fmtMs) +
      metricBlockHtml('Latency (ms)', 'latency', g.metrics.latency_ms, fmtMs) +
      '</div>';
  }
  body.innerHTML = html;
}

// ────────────────────────────────────────────────────────────────────────────
// ── samples table ──
// ────────────────────────────────────────────────────────────────────────────

async function _loadSamples() {
  const wrap = document.getElementById('mb-samples-table-wrap');
  if (wrap) wrap.innerHTML = '<div class="modelbench-loading">Loading…</div>';
  try {
    const url = samplesUrl({
      model: state.filters.model || undefined,
      think: state.filters.think || undefined,
      fit: state.filters.fit || undefined,
      limit: 200,
    });
    const data = await fetchJSON(state.API_BASE + url);
    state.samples = data;
    _renderSamplesTable(data);
  } catch (err) {
    _showError('Failed to load samples: ' + err.message);
    if (wrap) wrap.innerHTML = '';
  }
}

function _shortRunId(id) {
  return typeof id === 'string' && id.length > 12 ? id.slice(0, 10) + '…' : id;
}

function _renderSamplesTable(data) {
  const wrap = document.getElementById('mb-samples-table-wrap');
  if (!wrap) return;
  const rows = data.rows || [];
  if (rows.length === 0) {
    wrap.innerHTML = '<div class="modelbench-empty">No samples yet</div>';
    return;
  }
  let html = `<div class="modelbench-samples-meta">${rows.length} of ${data.total} shown</div>` +
    '<table class="modelbench-table modelbench-samples-table" aria-label="Raw samples"><thead><tr>' +
    '<th scope="col">Run</th><th scope="col">Model</th><th scope="col">Think</th><th scope="col">Fit</th>' +
    '<th scope="col">Tok/s</th><th scope="col">TTFT</th><th scope="col">Latency</th><th scope="col">Created</th>' +
    '<th scope="col">Provenance</th></tr></thead><tbody>';
  for (const r of rows) {
    const fitC = fitColor(r.vrram_fit);
    html += `<tr class="modelbench-sample-row" data-run-id="${escapeHtml(r.run_id)}" tabindex="0" role="button" aria-label="Inspect sample ${escapeHtml(r.run_id)}">` +
      `<td class="modelbench-mono" title="${escapeHtml(r.run_id)}">${escapeHtml(_shortRunId(r.run_id))}</td>` +
      `<td>${escapeHtml(r.model_tag)}</td>` +
      `<td>${r.think === true ? 'yes' : r.think === false ? 'no' : '—'}</td>` +
      `<td><span style="color:${fitC}">${escapeHtml(r.vrram_fit ?? '—')}</span></td>` +
      `<td>${fmtTps(r.tokens_per_sec)}</td>` +
      `<td>${fmtMs(r.ttft_ms)}</td>` +
      `<td>${fmtMs(r.latency_ms)}</td>` +
      `<td>${escapeHtml(r.created_at ?? '—')}</td>` +
      `<td>${r.provenance_complete
        ? '<span class="modelbench-prov-ok">complete</span>'
        : '<span class="modelbench-prov-bad">incomplete</span>'}</td>` +
      '</tr>';
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ────────────────────────────────────────────────────────────────────────────
// ── row detail drill-down ──
// ────────────────────────────────────────────────────────────────────────────

const _DRILLDOWN_FIELDS = [
  'run_id', 'model_tag', 'true_params', 'quant', 'ctx_len', 'think',
  'prompt_bytes', 'temperature', 'seed', 'ollama_version', 'vrram_fit',
  'output_tokens', 'thinking_tokens', 'content_tokens', 'tokens_per_sec',
  'ttft_ms', 'latency_ms', 'created_at', 'prompt_text', 'collector',
];

function _fmtRaw(v) {
  return v === null || v === undefined ? '—' : String(v);
}

async function _onSampleRowClick(runId) {
  const backdrop = document.getElementById('mb-drilldown-backdrop');
  const modal = document.getElementById('mb-drilldown-modal');
  if (!backdrop || !modal) return;
  modal.innerHTML = '<div class="modelbench-loading">Loading…</div>';
  backdrop.style.display = '';
  try {
    const data = await fetchJSON(state.API_BASE + sampleUrl(runId));
    _renderDrilldown(data);
  } catch (err) {
    modal.innerHTML = `<div class="modelbench-error-inline">Failed to load sample: ${escapeHtml(err.message)}</div>`;
  }
}

function _renderDrilldown(row) {
  const modal = document.getElementById('mb-drilldown-modal');
  if (!modal) return;
  const rowsHtml = _DRILLDOWN_FIELDS.map(
    (k) => `<tr><th scope="row">${escapeHtml(k)}</th><td>${escapeHtml(_fmtRaw(row[k]))}</td></tr>`
  ).join('');
  modal.innerHTML =
    '<div class="modelbench-drilldown-header">' +
      `<h5>Sample ${escapeHtml(row.run_id)}</h5>` +
      '<button type="button" class="modelbench-drilldown-close" id="mb-drilldown-close" title="Close" aria-label="Close sample detail">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
      '</button>' +
    '</div>' +
    `<div class="modelbench-drilldown-badge ${row.provenance_complete ? 'modelbench-prov-ok' : 'modelbench-prov-bad'}">` +
      (row.provenance_complete ? 'Provenance complete' : 'Provenance incomplete') +
    '</div>' +
    '<table class="modelbench-table modelbench-drilldown-table"><tbody>' + rowsHtml + '</tbody></table>';
}

function _closeDrilldown() {
  const backdrop = document.getElementById('mb-drilldown-backdrop');
  if (backdrop) backdrop.style.display = 'none';
}
