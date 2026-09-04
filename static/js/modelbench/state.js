// modelbench/state.js — shared mutable state for the ModelBench dashboard
const state = {
  API_BASE: '',
  isOpen: false,
  filters: { fit: '', think: '', model: '' },
  models: null,   // last /api/modelbench/models response
  metrics: null,  // last /api/modelbench/metrics response (for the selected model)
  samples: null,  // last /api/modelbench/samples response
};

/** Reset transient state to defaults — keeps API_BASE sticky across close/reopen. */
export function reset() {
  state.isOpen = false;
  state.filters = { fit: '', think: '', model: '' };
  state.models = null;
  state.metrics = null;
  state.samples = null;
}

export default state;
