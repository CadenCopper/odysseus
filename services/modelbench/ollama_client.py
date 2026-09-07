"""Ollama-native client for the ModelBench runner.

The runner GUI needs two capabilities the existing ``bench_samples`` dashboard
does not provide:

1. listing the models actually resident on the GPU ollama host (native
   ``GET /api/tags``), and
2. pulling a new model from the registry (native ``POST /api/pull``), streaming
   progress back so the UI can render a live bar.

This module owns the HTTP conversation with ollama and the safety/serialization
helpers the routes layer wires on top. It is deliberately lightweight and
injectable (``OllamaClient`` takes an optional ``httpx.AsyncClient`` and the
routes take a client factory) so the route tests run fully offline.

API root resolution
-------------------
The bench research (R-ModelBench-runner-ollama-contract.md, card ``t_a090857f``)
found the P-file's recorded ``OLLAMA_URL`` (10.0.0.188) serves a *different,
near-empty* ollama instance, while the real copper-wsl ollama holding every
resident Qwen model is at 10.0.0.8. ``ollama_root()`` honors
``MODELBENCH_OLLAMA_URL`` when set (the operator-corrected override), then
``OLLAMA_URL``/``OLLAMA_BASE_URL``, normalized to a native API root. Pointing
the runner at the wrong instance would list/pull nothing useful.

Pull semantics (verified live against 0.32.13)
----------------------------------------------
- A nonexistent tag returns HTTP 200 with a *streamed* ``{"error": ...}``
  object, NOT a 404. Callers must scan the stream body for an ``error`` key,
  never trust the status code.
- Network / registry / disk-full / insufficient-storage failures all surface as
  the same in-band streamed ``error`` object.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Hard VRAM budget the bench was built around (12GB card). The pull-size safety
# threshold defaults to a bit above the resident working set (16GB) and blocks
# any pull larger than it unless the operator explicitly confirms an offload
# pull.
_DEFAULT_MAX_PULL_SIZE_GB = 16.0


class OllamaUnreachable(Exception):
    """Ollama could not be reached (transport / non-2xx on the request itself).

    Distinct from :class:`OllamaPullError`, which is a pull that *started* and
    then failed (streamed ``error`` from the server).
    """


class OllamaPullError(Exception):
    """A streamed pull failed. ``message`` is the server's error string."""


def _native_api_root(url: str) -> str:
    """Return a native ollama API root (scheme://host[:port]/api) for ``url``.

    Mirrors the shaping in ``src/llm_core._ollama_api_root`` so a configured
    URL that already points at ``/api/chat``, ``/api/tags`` or ``/api`` is
    normalized to the shared ``/api`` root instead of being double-suffixed.
    An OpenAI-compat ``/v1`` URL is accepted as-is (its native ``/api`` root
    is the base).
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    for suffix in ("/api/chat", "/api/tags", "/api/generate", "/api/pull"):
        if path.endswith(suffix):
            return url[: -len(suffix.split("/")[-1])].rstrip("/")
    if path.endswith("/api"):
        return url
    if path.endswith("/v1") or path.endswith("/v1/"):
        return url[: -len("/v1")].rstrip("/") + "/api"
    if path == "":
        return url + "/api"
    return url + "/api"


def ollama_root() -> str:
    """Resolve the native ollama API root the runner should drive.

    Precedence: ``MODELBENCH_OLLAMA_URL`` > ``OLLAMA_URL`` > ``OLLAMA_BASE_URL``.
    The explicit ``MODELBENCH_OLLAMA_URL`` override exists because the bench
    research showed the P-file's recorded ``OLLAMA_URL`` points at the wrong
    (near-empty) instance.
    """
    raw = (
        os.getenv("MODELBENCH_OLLAMA_URL")
        or os.getenv("OLLAMA_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://127.0.0.1:11434"
    )
    root = _native_api_root(raw)
    if not root:
        logger.warning("resolved empty ollama root from %r; falling back to localhost", raw)
        return "http://127.0.0.1:11434/api"
    return root


def get_max_pull_size_gb() -> float:
    """Operator-configurable pull-size safety threshold in GB (default 16.0)."""
    raw = os.getenv("MODELBENCH_MAX_PULL_SIZE_GB", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("ignoring non-numeric MODELBENCH_MAX_PULL_SIZE_GB=%r", raw)
    return _DEFAULT_MAX_PULL_SIZE_GB


class PullGate:
    """Single-flight slot for ollama pulls.

    Threading-lock guarded boolean so the slot works across asyncio event loops
    (TestClient's portal vs the app's own loop) without the
    "bound to a different event loop" failure asyncio.Lock trips on.

    ``try_acquire`` returns False while a pull is already in flight; callers map
    that to HTTP 409/423. The gate is deliberately independent of the bench-job
    registry so a pull can run while nothing else does, but routes additionally
    refuse a pull while a bench run is active.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active:
                return False
            self._active = True
            return True

    def release(self) -> None:
        with self._lock:
            self._active = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active


def annotate_resident_models(
    models: Iterable[Mapping[str, Any]],
    sample_tags: set[str] | frozenset[str],
    fit_by_tag: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Annotate raw ``/api/tags`` model objects for the runner list endpoint.

    Each resident model gains ``has_samples`` (already benchmarked — i.e. its
    tag appears in ``bench_samples.model_tag``) and ``vrram_fit`` (the class
    recorded for it in bench_samples, when known). ``size_bytes``/``size_gb``
    let the GUI apply the offload-pull safety warning before the operator pulls.
    """
    annotated: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, Mapping):
            continue
        name = model.get("name") or model.get("model")
        if not isinstance(name, str) or not name:
            continue
        size = model.get("size")
        if not isinstance(size, (int, float)):
            size = None
        details = model.get("details") or {}
        entry = {
            "name": name,
            "model": model.get("model") or name,
            "resident": True,
            "size_bytes": size,
            "size_gb": round(size / 1e9, 2) if size is not None else None,
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "context_length": details.get("context_length"),
            "capabilities": model.get("capabilities") or [],
            "modified_at": model.get("modified_at"),
            "has_samples": name in sample_tags,
            "vrram_fit": fit_by_tag.get(name),
        }
        annotated.append(entry)
    return annotated


class OllamaClient:
    """Async client over the native ollama ``/api`` surface.

    Injectable: pass an existing ``httpx.AsyncClient`` to avoid creating a new
    connection pool per request, or (in tests) to swap in a fake transport.
    """

    def __init__(self, root: Optional[str] = None, http: Optional[httpx.AsyncClient] = None, timeout: float = 30.0):
        root = (root or ollama_root()).strip().rstrip("/")
        # ``ollama_root()``/``_native_api_root`` always end in ``/api``; hold the
        # host base separately so endpoint paths (``/api/tags``, ``/api/pull``)
        # are never double-suffixed.
        if root.endswith("/api"):
            self.base = root[: -len("/api")].rstrip("/")
        else:
            self.base = root
        self._http = http
        self._timeout = timeout

    @property
    def root(self) -> str:
        """The native API root (base + ``/api``), e.g. ``http://10.0.0.8:11434/api``."""
        return f"{self.base}/api"

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # pragma: no cover - defensive
                pass
            self._http = None

    async def tags(self) -> list[Mapping[str, Any]]:
        """Fetch the resident model list (native GET /api/tags)."""
        try:
            resp = await self.http.get(f"{self.base}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaUnreachable(f"ollama tags request failed: {e}") from e
        data = resp.json()
        models = data.get("models") if isinstance(data, Mapping) else None
        return list(models) if isinstance(models, list) else []

    async def stream_pull(self, tag: str) -> AsyncIterator[Mapping[str, Any]]:
        """Yield per-layer progress objects from a streamed ``POST /api/pull``.

        Raises :class:`OllamaPullError` when ollama streams an ``error`` object
        (manifest-missing, network, disk-full, insufficient-storage — all arrive
        in-band as HTTP 200 + ``{"error": ...}``, per live 0.32.13 verification)
        or when the transport itself fails.
        """
        payload = {"model": tag, "stream": True}
        try:
            async with self.http.stream("POST", f"{self.base}/api/pull", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        logger.debug("ignoring non-JSON ollama pull line: %r", line[:120])
                        continue
                    if isinstance(obj, Mapping) and "error" in obj:
                        err = obj.get("error")
                        raise OllamaPullError(str(err) if err is not None else "unknown pull error")
                    yield obj
        except OllamaPullError:
            raise
        except httpx.HTTPError as e:
            raise OllamaPullError(f"ollama pull transport failure: {e}") from e
