"""Regression test for the modelbench route shim.

Mirrors ``test_compare_routes_shim.py``: the shim at
``routes/modelbench_routes.py`` must resolve to the *same* module object as
the canonical ``routes/modelbench/modelbench_routes.py`` so that legacy
imports and monkeypatches (e.g. on ``SessionLocal``) apply to the code
FastAPI actually runs.
"""

import importlib

import routes.modelbench_routes as _shim_modelbench  # noqa: F401


def test_legacy_and_canonical_modelbench_module_are_same_object():
    """``import routes.modelbench_routes`` must alias the canonical module."""
    legacy = importlib.import_module("routes.modelbench_routes")
    canonical = importlib.import_module("routes.modelbench.modelbench_routes")
    assert legacy is canonical, (
        "routes.modelbench_routes shim must resolve to the canonical "
        "routes.modelbench.modelbench_routes module object"
    )
