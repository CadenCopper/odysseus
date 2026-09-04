"""Backward-compat shim — canonical location is
routes/modelbench/modelbench_routes.py.

This module is replaced in ``sys.modules`` by the canonical module object so
that ``import routes.modelbench_routes``, ``from routes.modelbench_routes
import X``, ``importlib.import_module("routes.modelbench_routes")``, and the
``import ... as mbr`` + ``monkeypatch.setattr(mbr, "SessionLocal", ...)``
pattern all operate on the *same* object the application actually uses.
Mirrors routes/compare_routes.py exactly.
"""

import sys as _sys

from routes.modelbench import modelbench_routes as _canonical  # noqa: F401

_sys.modules[__name__] = _canonical
