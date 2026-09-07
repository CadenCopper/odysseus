// modelbench/api.js — pure URL builders for /api/modelbench/*. No fetch here
// (that lives in index.js) so these stay importable/testable under plain node.

const BASE = '/api/modelbench';

function withQuery(path, params) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue;
    qs.set(k, String(v));
  }
  const s = qs.toString();
  return s ? `${path}?${s}` : path;
}

/** GET /api/modelbench/models[?fit=] */
export function modelsUrl(fit) {
  return withQuery(`${BASE}/models`, { fit });
}

/** GET /api/modelbench/metrics?model=&think=&fit=&percentiles= */
export function metricsUrl(model, { think, fit, percentiles = [50, 90, 95] } = {}) {
  return withQuery(`${BASE}/metrics`, {
    model,
    percentiles: percentiles.join(','),
    think: think === undefined ? undefined : String(think),
    fit,
  });
}

/** GET /api/modelbench/samples[?model=&think=&fit=&limit=&offset=] */
export function samplesUrl({ model, think, fit, limit, offset } = {}) {
  return withQuery(`${BASE}/samples`, { model, think, fit, limit, offset });
}

/** GET /api/modelbench/samples/{run_id} */
export function sampleUrl(runId) {
  return `${BASE}/samples/${encodeURIComponent(runId)}`;
}

/** POST /api/modelbench/runs, GET /api/modelbench/runs */
export function runsUrl() {
  return `${BASE}/runs`;
}

/** GET /api/modelbench/runs/{job_id} */
export function runUrl(jobId) {
  return `${BASE}/runs/${encodeURIComponent(jobId)}`;
}

/** POST /api/modelbench/runs/{job_id}/cancel */
export function runCancelUrl(jobId) {
  return `${BASE}/runs/${encodeURIComponent(jobId)}/cancel`;
}

/** GET /api/modelbench/ollama/models */
export function ollamaModelsUrl() {
  return `${BASE}/ollama/models`;
}

/** POST /api/modelbench/ollama/models/{tag}/pull */
export function pullUrl(tag) {
  return `${BASE}/ollama/models/${encodeURIComponent(tag)}/pull`;
}
