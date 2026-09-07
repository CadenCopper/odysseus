// modelbench/runner.js — runner control logic for the in-dashboard test
// runner. Pure decision logic (single-run lock, terminal-state detection,
// progress math, pull-event reduction) plus a factory that takes ALL side
// effects (fetch, timers, DOM) via dependency injection, so this module is
// importable and exercisable under plain node with no DOM/fetch at
// module-load time. DOM access only happens inside wire(), and only once a
// real panel element is passed in — index.js is the only caller of wire().

const TERMINAL_STATUSES = new Set(['done', 'failed', 'cancelled']);

/** 'done'|'failed'|'cancelled' -> true; 'queued'|'running' -> false. */
export function isTerminal(status) {
  return TERMINAL_STATUSES.has(status);
}

/** job.progress (0..1) -> 0..100 integer, clamped; missing/non-finite -> 0. */
export function progressPct(job) {
  if (!job) return 0;
  const p = Number(job.progress);
  if (!Number.isFinite(p)) return 0;
  return Math.max(0, Math.min(100, Math.round(p * 100)));
}

/**
 * Fold one parsed SSE pull event into the running pull-progress state.
 * Handles the three event shapes from POST .../pull (see routes/modelbench/
 * ollama_routes.py): {event:'started',...}, {event:'done',ok}, {event:'error',
 * error}, and bare ollama per-layer progress objects ({status, completed,
 * total}).
 */
export function reducePullEvent(state, ev) {
  const base = state || { status: '', completed: 0, total: 0, done: false, ok: null, error: null };
  if (!ev || typeof ev !== 'object') return base;
  if (ev.event === 'started') {
    return { ...base, status: 'downloading', sizeBytes: ev.size_bytes, maxPullSizeGb: ev.max_pull_size_gb };
  }
  if (ev.event === 'done') {
    return { ...base, done: true, ok: !!ev.ok };
  }
  if (ev.event === 'error') {
    return { ...base, error: ev.error };
  }
  const completed = typeof ev.completed === 'number' ? ev.completed : base.completed;
  const total = typeof ev.total === 'number' ? ev.total : base.total;
  return { ...base, status: ev.status || base.status, completed, total };
}

/** pull-progress state -> 0..100 integer percent; guards missing/zero total. */
export function pullProgressPct(pullState) {
  if (!pullState || !pullState.total) return 0;
  return Math.max(0, Math.min(100, Math.round((pullState.completed / pullState.total) * 100)));
}

/**
 * Factory for the runner's control logic. All side effects come through
 * `deps` so this can run under plain node in tests with fakes.
 *
 * deps:
 *   - startRun(params)   => Promise<job>   POST runs
 *   - pollRun(jobId)     => Promise<job>   GET runs/{id}
 *   - cancelRun(jobId)   => Promise<any>   POST runs/{id}/cancel
 *   - startPull(tag, confirm) => AsyncIterable<event>  parsed SSE events
 *   - refresh()          => void|Promise   dashboard reload, called once a run goes terminal
 *   - onJobChange(job|null)   => void      called whenever the active job changes (render hook)
 *   - onPullChange(pullState|null) => void called whenever pull state changes (render hook)
 *   - setTimeoutFn / clearTimeoutFn — default to the global timers; tests inject fakes
 *   - pollIntervalMs (default 2000)
 */
export function createRunner(deps = {}) {
  const {
    startRun,
    pollRun,
    cancelRun,
    startPull,
    refresh = () => {},
    onJobChange = () => {},
    onPullChange = () => {},
    setTimeoutFn = (typeof setTimeout !== 'undefined' ? setTimeout : null),
    clearTimeoutFn = (typeof clearTimeout !== 'undefined' ? clearTimeout : null),
    pollIntervalMs = 2000,
  } = deps;

  let activeJob = null;
  let pollTimer = null;
  let pullActive = false;
  let pullState = null;

  function isActive() {
    return !!activeJob && !isTerminal(activeJob.status);
  }

  function isPullActive() {
    return pullActive;
  }

  function getJob() {
    return activeJob;
  }

  function getPullState() {
    return pullState;
  }

  function setJobState(job) {
    activeJob = job;
    onJobChange(job);
  }

  function _stopPolling() {
    if (pollTimer != null) {
      clearTimeoutFn(pollTimer);
      pollTimer = null;
    }
  }

  async function _pollOnce(jobId) {
    const job = await pollRun(jobId);
    setJobState(job);
    if (isTerminal(job.status)) {
      _stopPolling();
      await refresh();
      return job;
    }
    pollTimer = setTimeoutFn(() => _pollOnce(jobId), pollIntervalMs);
    return job;
  }

  /** Start a run. No-op (returns the current job) while one is already active. */
  async function start(params) {
    if (isActive()) return activeJob;
    const job = await startRun(params);
    setJobState(job);
    if (!isTerminal(job.status)) {
      pollTimer = setTimeoutFn(() => _pollOnce(job.job_id), pollIntervalMs);
    }
    return job;
  }

  /** Cancel the active job, then poll immediately to reflect the new status. */
  async function cancel() {
    if (!activeJob) return null;
    const jobId = activeJob.job_id;
    await cancelRun(jobId);
    _stopPolling();
    return _pollOnce(jobId);
  }

  /** Pull a model tag. No-op while a run or another pull is active. */
  async function pull(tag, { confirm = false } = {}) {
    if (isActive() || pullActive) return null;
    pullActive = true;
    pullState = { status: 'starting', completed: 0, total: 0, done: false, ok: null, error: null };
    onPullChange(pullState);
    try {
      for await (const ev of startPull(tag, confirm)) {
        pullState = reducePullEvent(pullState, ev);
        onPullChange(pullState);
        if (pullState.done) break;
      }
    } catch (err) {
      pullState = {
        ...pullState,
        done: true,
        ok: false,
        error: String((err && err.message) || err),
        errorStatus: err && err.status,
      };
      onPullChange(pullState);
    } finally {
      pullActive = false;
    }
    return pullState;
  }

  /** Bind to a real panel element. Only ever called by index.js with a live DOM node. */
  function wire(panelEl) {
    if (!panelEl) return;
    // DOM wiring is intentionally thin here — index.js owns element lookup
    // and markup; this just exposes the handlers it should attach.
    return {
      onSubmit: start,
      onCancel: cancel,
      onPull: pull,
    };
  }

  return {
    start,
    cancel,
    pull,
    isActive,
    isPullActive,
    getJob,
    getPullState,
    setJobState,
    wire,
  };
}
