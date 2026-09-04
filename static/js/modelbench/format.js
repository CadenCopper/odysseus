// modelbench/format.js — pure display-formatting helpers, no DOM.

/** tokens/sec -> 1-decimal string, or an em-dash for missing data. */
export function fmtTps(v) {
  if (v === null || v === undefined) return '—';
  return Number(v).toFixed(1);
}

/** milliseconds -> 0-or-1-decimal string (trailing .0 dropped), or an em-dash. */
export function fmtMs(v) {
  if (v === null || v === undefined) return '—';
  const rounded = Math.round(Number(v) * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

/** context length -> thousands-separated integer string, or an em-dash for null. */
export function fmtCtx(v) {
  if (v === null || v === undefined) return '—';
  return Math.round(Number(v)).toLocaleString('en-US');
}

/** vrram_fit class -> a CSS color usable directly in an inline style attribute. */
export function fitColor(fit) {
  switch (fit) {
    case 'fit': return 'var(--green, #50fa7b)';
    case 'partial': return 'var(--yellow, #f1fa8c)';
    case 'offload': return 'var(--orange, #ffb86c)';
    default: return 'color-mix(in srgb, var(--fg) 45%, transparent)';
  }
}

/**
 * Classify the real-ctx-vs-advertised-ctx relationship for a model row.
 * 'not-swept' when the fit class was never measured, 'cliff' when the
 * measured max fit context falls short of the advertised max, 'ok' when
 * they match, '' when swept but neither cliff nor exact match.
 */
export function ctxCliffClass(advertised, achieved, achieved_swept) {
  if (!achieved_swept) return 'not-swept';
  if (achieved !== null && advertised !== null && achieved < advertised) return 'cliff';
  if (achieved === advertised) return 'ok';
  return '';
}

/** value/maxValue as a 0-100 integer percentage, 0 when maxValue is non-positive. */
export function barPct(value, maxValue) {
  if (!maxValue || maxValue <= 0) return 0;
  return Math.min(100, Math.round((value / maxValue) * 100));
}
