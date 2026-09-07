// modelbench/state.js — shared mutable state for the ModelBench dashboard
const state = {
  API_BASE: '',
  isOpen: false,
  filters: { fit: '', think: '', model: '' },
  models: null,   // last /api/modelbench/models response
  metrics: null,  // last /api/modelbench/metrics response (for the selected model)
  samples: null,  // last /api/modelbench/samples response
  residentModels: null, // last /api/modelbench/ollama/models response (runner model picker)
};

/** Reset transient state to defaults — keeps API_BASE sticky across close/reopen. */
export function reset() {
  state.isOpen = false;
  state.filters = { fit: '', think: '', model: '' };
  state.models = null;
  state.metrics = null;
  state.samples = null;
  state.residentModels = null;
}

export default state;
