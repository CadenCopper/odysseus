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

/**
 * Parse the runner's optional "Context sweep points" field into a validated
 * int list, mirroring the server's ctx_sweep rules (routes/modelbench).
 * Blank/whitespace/null input means "omit ctx_sweep" -> backend default
 * doubling sweep: { points: null, error: null }. `cap` is the effective ctx
 * cap (e.g. the selected resident model's context_length); pass null/undefined
 * to skip the cap check client-side (the server still enforces it).
 */
export function parseCtxSweep(raw, cap, maxProbes = 12) {
  const trimmed = (raw || '').trim();
  if (!trimmed) return { points: null, error: null };
  const tokens = trimmed.split(',').map((t) => t.trim());
  const points = [];
  for (const tok of tokens) {
    if (!/^-?\d+$/.test(tok)) {
      return { points: null, error: `"${tok}" is not a whole number` };
    }
    const n = parseInt(tok, 10);
    if (n < 1) {
      return { points: null, error: `"${tok}" must be at least 1` };
    }
    if (cap !== null && cap !== undefined && n > cap) {
      return { points: null, error: `"${tok}" exceeds the max context length (${cap})` };
    }
    points.push(n);
  }
  if (points.length > maxProbes) {
    return { points: null, error: `Too many context points (${points.length}); max is ${maxProbes}` };
  }
  return { points, error: null };
}

/**
 * Deterministic doubling-default preview (1024, 2048, 4096, ...) for the
 * ctx-sweep field placeholder, bounded to a short readable string. Mirrors
 * the backend's ctx_sweep_points(cap) shape without duplicating its cap logic.
 */
export function ctxSweepDefaultPreview(cap) {
  const c = Number(cap);
  if (!Number.isFinite(c) || c <= 0) return '1024,2048,4096,…';
  if (c < 1024) return String(Math.round(c));
  const points = [1024];
  while (points.length < 12 && points[points.length - 1] * 2 <= c) {
    points.push(points[points.length - 1] * 2);
  }
  return points.length <= 4 ? points.join(',') : points.slice(0, 3).join(',') + ',…';
}
