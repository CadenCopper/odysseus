// modelbench/markup.js — pure, DOM-free HTML-string builders for the
// interpretability parts of the dashboard (column/metric tooltips, the labeled
// think split, the models table, metric blocks).
//
// Kept OUT of index.js (the DOM panel) so the exact strings that ship to the
// page can be unit-tested under plain node — the same dependency-injection
// idiom used for runner.js (see tests/test_modelbench_frontend_js.py). No DOM,
// no fetch, no ui.js import chain, so `node --input-type=module` can load it.
//
// CSP contract (core/middleware.py): script-src 'self' 'nonce-<per-request>'
// forbids inline event-handler attributes (onclick/onmouseenter/...), so each
// ?-bubble is a real focusable <button type="button"> whose aria-label and
// data-tooltip carry the grounded copy, and index.js wires show/hide via
// addEventListener onto ONE positioned tooltip element. No inline handlers, no
// inline <script>, no new vendored deps. Inline style="" attrs are allowed by
// the CSP (`style-src 'unsafe-inline'`) and are used only for pre-existing
// colored badges/bars.

import { fmtCtx, ctxCliffClass, fitColor, barPct } from './format.js';

// HTML-escape mirroring ui.js `esc` (the canonical impl at static/js/ui.js
// L786). Replicated here rather than importing ui.js because ui.js drags in the
// DOM module graph and would make this module un-importable under node in the
// tests. The escape table is identical, so rendered output is byte-identical.
const _ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export function esc(s) {
  return (s || '').replace(/[&<>"']/g, (m) => _ESC_MAP[m]);
}

// The 11 fields that must all be non-null for a sample to count as
// provenance-complete. Mirrors routes/modelbench/modelbench_routes.py
// `_PROVENANCE_FIELDS` (the source of truth for the Provenance column's
// "complete/total" value).
export const PROVENANCE_FIELDS = [
  'run_id', 'model_tag', 'true_params', 'quant', 'ctx_len', 'think',
  'prompt_bytes', 'temperature', 'seed', 'ollama_version', 'vrram_fit',
];

/**
 * Static tooltip copy keyed by the dashboard element it explains. Values are
 * author-controlled and constant; they are HTML-escaped on insert into the
 * data-tooltip attribute regardless, so they are safe even if later edited.
 * Copy is grounded in the operator/themis decision for this card.
 */
export const TOOLTIPS = {
  think:
    'Whether the run enabled the model\u2019s thinking/reasoning channel. ' +
    'A model is sampled in both classes when it supports thinking.',
  provenance:
    `Provenance-complete = the ${PROVENANCE_FIELDS.length} identity fields on the row are all non-null ` +
    `(${PROVENANCE_FIELDS.join(', ')}), so the sample can be re-verified. Renders complete/total.`,
  fit_split:
    'VRAM residency on the 12GB card: fit = model fully resident in VRAM ' +
    '(size_vram/size >= 1.0), partial = partially resident, offload = not ' +
    'resident in VRAM (size_vram ~0). Derived from ollama /api/ps.',
  context:
    'Advertised = the model\u2019s maximum context_length; achieved = the highest ' +
    'ctx point the sweep actually reached for this model\u2019s fit class. ' +
    '\u2018cliff\u2019 flag = achieved fell short of advertised.',
  tokens_per_sec:
    'Tokens/sec = output tokens produced per second across the selected samples.',
  ttft:
    'TTFT = time to first token, in ms: latency until the model emits its first output token.',
  latency:
    'Latency = total per-sample generation time, in ms (end-to-end).',
};

// Human column/metric name for each tooltip key, used to build the grounded
// aria-label ("What is <col>?") on the trigger button.
const TIP_COL = {
  think: 'Think',
  provenance: 'Provenance',
  fit_split: 'Fit split',
  context: 'Context',
  tokens_per_sec: 'Tokens/sec',
  ttft: 'TTFT (ms)',
  latency: 'Latency (ms)',
};

/**
 * ?-bubble tooltip trigger: a real, keyboard-focusable <button type="button">
 * that carries the grounded explanation in both a machine-readable
 * data-tooltip attribute and a human aria-label. index.js wires a single
 * positioned tooltip element to it via addEventListener on mouseenter/focus /
 * mouseleave/blur (CSP-safe — no inline event-handler attributes here).
 */
export function tipButton(key) {
  const text = TOOLTIPS[key];
  if (!text) return '';
  const enc = esc(text);
  const col = TIP_COL[key] || key;
  return `<button type="button" class="modelbench-tip" aria-label="What is ${esc(col)}?" data-tooltip="${enc}">?</button>`;
}

/**
 * Unambiguous think split for a model row. Accepts the row's `think`
 * tally ({ false: N, true: M }) and returns "no N · yes M" instead of the
 * ambiguous "N / M".
 */
export function fmtThinkSplit(think) {
  const t = think || {};
  const no = Number(t.false ?? 0);
  const yes = Number(t.true ?? 0);
  return `no ${no} · yes ${yes}`;
}

function _fitSplitBadgesHtml(split = {}) {
  return ['fit', 'partial', 'offload'].map((k) => {
    const n = split[k] || 0;
    const c = fitColor(k);
    return `<span class="modelbench-fit-badge" style="color:${c};border-color:${c};">${k} ${n}</span>`;
  }).join(' ');
}

/**
 * Full <table> markup for the models summary. The Model / Params / Quant
 * columns render byte-identically to before the tooltip/interpretability
 * change (their <th> and <td> fragments are unchanged); the Think / Fit split
 * / Context / Provenance headers gain tooltip spans and the Think cell shows
 * the labeled "no N · yes M" split.
 */
export function modelsTableHtml(models = [], activeModel) {
  const rows = models.map((m) => {
    const cliff = ctxCliffClass(m.ctx?.advertised, m.ctx?.achieved, m.ctx?.achieved_swept);
    const ctxText = m.ctx?.achieved_swept
      ? `${fmtCtx(m.ctx?.advertised)} → ${fmtCtx(m.ctx?.achieved)}`
      : `${fmtCtx(m.ctx?.advertised)} → not yet measured`;
    const cliffNote = cliff === 'cliff' ? ' <span class="modelbench-cliff-flag">cliff</span>' : '';
    const active = activeModel === m.model_tag ? ' modelbench-row-active' : '';
    const complete = Number(m.provenance?.complete ?? 0);
    const total = Number(m.provenance?.total ?? 0);
    return '<tr class="modelbench-model-row' + active +
      `" data-model="${esc(m.model_tag)}" tabindex="0" role="button"` +
      ` aria-label="Show metrics for ${esc(m.model_tag)}">` +
      `<td>${esc(m.model_tag)}</td>` +
      `<td>${esc(String(m.true_params ?? '—'))}</td>` +
      `<td>${esc(String(m.quant ?? '—'))}</td>` +
      `<td>${m.sample_count}</td>` +
      `<td>${fmtThinkSplit(m.think)}</td>` +
      `<td>${_fitSplitBadgesHtml(m.fit_split)}</td>` +
      `<td class="modelbench-ctx-cell modelbench-ctx-${cliff}">${ctxText}${cliffNote}</td>` +
      `<td>${complete}/${total}</td>` +
      '</tr>';
  }).join('');
  return '<table class="modelbench-table" aria-label="Model summary"><thead><tr>' +
    '<th scope="col">Model</th><th scope="col">Params</th><th scope="col">Quant</th>' +
    '<th scope="col">Samples</th>' +
    `<th scope="col">Think (no / yes) ${tipButton('think')}</th>` +
    `<th scope="col">Fit split ${tipButton('fit_split')}</th>` +
    `<th scope="col">Context (advertised → achieved) ${tipButton('context')}</th>` +
    `<th scope="col">Provenance ${tipButton('provenance')}</th>` +
    '</tr></thead><tbody>' + rows + '</tbody></table>';
}

/**
 * A metric block (label + DOM bar rows) for one per-model metric, e.g.
 * Tokens/sec / TTFT / Latency. `tipKey` selects the metric's tooltip copy;
 * `fmt` formats the numeric values.
 */
export function metricBlockHtml(label, tipKey, m, fmt) {
  const tip = tipButton(tipKey);
  if (!m) {
    return '<div class="modelbench-metric-block">' +
      `<div class="modelbench-metric-label">${esc(label)} ${tip}</div>` +
      '<div class="modelbench-empty">No data</div></div>';
  }
  const max = m.max;
  const rows = [['mean', m.mean], ...Object.entries(m.percentiles || {})];
  const barsHtml = rows.map(([tag, v]) => {
    const pct = barPct(v, max);
    return '<div class="modelbench-bar-row">' +
      `<span class="modelbench-bar-tag">${esc(tag)}</span>` +
      `<span class="modelbench-bar-track"><span class="modelbench-bar-fill" style="width:${pct}%"></span></span>` +
      `<span class="modelbench-bar-value">${fmt(v)}</span>` +
      '</div>';
  }).join('');
  return '<div class="modelbench-metric-block">' +
    `<div class="modelbench-metric-label">${esc(label)} ${tip}` +
    `<span class="modelbench-metric-meta">min ${fmt(m.min)} · max ${fmt(m.max)} · n=${m.count}</span></div>` +
    barsHtml +
    '</div>';
}
