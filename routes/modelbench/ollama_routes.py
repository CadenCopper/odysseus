"""Ollama-native ModelBench runner endpoints.

``GET  /api/modelbench/ollama/models``           — live resident model list
``POST /api/modelbench/ollama/models/{tag}/pull`` — streamed model pull (SSE)

These are the ollama-facing surface the runner GUI uses to list resident models
and pull new ones. They are DISTINCT from ``GET /api/modelbench/models``, which
summarizes the already-benchmarked ``bench_samples`` rows.

Auth/CSP posture (mirrors the runs-API card): neither endpoint is in the
auth-exempt path; both call ``require_authenticated_request`` so an
unauthenticated caller gets 401 when auth is enabled. The pull is a
single-flight slot — one pull at a time, refused while a bench run is active —
and refuses to silently pull a model larger than the VRAM safety threshold
unless the operator explicitly confirms an offload pull.

Pull progress streams back as Server-Sent Events; the frontend (separate card)
renders a live bar from ``status``/``completed``/``total`` and surfaces any
streamed ``error``.
"""

import json
import logging
from collections.abc import Mapping
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.database import BenchSample, SessionLocal
from src.auth_helpers import get_current_user, require_authenticated_request
from services.modelbench.ollama_client import (
    OllamaClient,
    OllamaPullError,
    OllamaUnreachable,
    PullGate,
    annotate_resident_models,
    get_max_pull_size_gb,
)

logger = logging.getLogger(__name__)

# One pull at a time, app-wide. Threading-guarded so it works across asyncio
# loops (TestClient portal vs the app loop) without loop-binding errors.
_pull_gate = PullGate()

_SSE_MEDIA_TYPE = "text/event-stream"


def _sse(obj: Mapping[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _default_client_factory() -> OllamaClient:
    return OllamaClient()


def _sample_context() -> tuple[set[str], dict[str, str]]:
    """Tags already benchmarked + each tag's recorded vrram_fit class.

    Reads ``bench_samples`` (the same table the read-only dashboard summarizes)
    so a resident model can be annotated with ``has_samples`` and the fit class
    observed for it. ``vrram_fit`` here is the *recorded* class (fit/partial/
    offload), not a fresh residency probe — that requires an invocation and
    belongs to the runner engine card.
    """
    sample_tags: set[str] = set()
    fit_counts: dict[str, dict[str, int]] = {}
    db = SessionLocal()
    try:
        rows = db.query(BenchSample.model_tag, BenchSample.vrram_fit).all()
    finally:
        db.close()
    for tag, fit in rows:
        sample_tags.add(tag)
        if fit:
            fit_counts.setdefault(tag, {}).setdefault(fit, 0)
            fit_counts[tag][fit] += 1
    fit_by_tag = {}
    for tag, counts in fit_counts.items():
        fit_by_tag[tag] = max(counts.items(), key=lambda kv: kv[1])[0]
    return sample_tags, fit_by_tag


def setup_ollama_routes(
    ollama_client_factory: Optional[Callable[[], OllamaClient]] = None,
    bench_registry: Optional[Any] = None,
) -> APIRouter:
    """Build the ollama router.

    ``ollama_client_factory`` returns a fresh :class:`OllamaClient` per request
    (default: real client against the resolved native root). ``bench_registry``
    is the app's single-run ``BenchJobRegistry``; when supplied, a pull is
    refused while a bench run is active (serialization safety).
    """
    factory = ollama_client_factory or _default_client_factory
    router = APIRouter(prefix="/api/modelbench/ollama", tags=["modelbench"])

    @router.get("/models")
    async def list_resident_models(request: Request):
        """Live ollama-resident model list, annotated for the runner GUI.

        Each resident model carries ``has_samples`` (bench_sampled) and its
        recorded ``vrram_fit`` where known, plus ``size_bytes``/``size_gb`` for
        the GUI's offload-pull warning.
        """
        require_authenticated_request(request)
        client = factory()
        try:
            models = await client.tags()
        except OllamaUnreachable as e:
            raise HTTPException(502, str(e))
        finally:
            await client.aclose()
        sample_tags, fit_by_tag = _sample_context()
        annotated = annotate_resident_models(models, sample_tags, fit_by_tag)
        return {
            "models": annotated,
            "meta": {
                "ollama_root": client.root,
                "resident_count": len(annotated),
                "bench_sampled_count": len(sample_tags),
                "max_pull_size_gb": get_max_pull_size_gb(),
            },
        }

    @router.post("/models/{tag:path}/pull")
    async def pull_model(tag: str, request: Request):
        """Kick a streamed ollama pull for ``tag`` and stream progress (SSE).

        Auth-gated. Single-flight: 409 while another pull is in flight or a
        bench run is active. A model whose known (resident) size exceeds the
        VRAM safety threshold is refused with 400 unless the operator sends an
        explicit ``confirm`` (offload) — never silently pulls a >VRAM model.
        """
        require_authenticated_request(request)
        initiator = get_current_user(request) or ""

        body: dict[str, Any] = {}
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                body = raw
        except Exception:
            body = {}

        confirm = bool(body.get("confirm")) or bool(body.get("confirmed_offload"))

        # Single-flight pull slot.
        if not _pull_gate.try_acquire():
            raise HTTPException(
                409,
                "A model pull is already in progress; only one pull is allowed at a time.",
            )
        client = None
        try:
            # Serialize with bench runs: no generation while pulling.
            if bench_registry is not None and bench_registry.has_active_run():
                raise HTTPException(
                    409,
                    "A bench run is active; a model pull cannot start while a run is in progress.",
                )

            client = factory()
            try:
                models = await client.tags()
            except OllamaUnreachable as e:
                raise HTTPException(502, str(e))

            size = None
            for m in models:
                name = m.get("name") or m.get("model")
                if name == tag and isinstance(m.get("size"), (int, float)):
                    size = m["size"]
                    break

            max_gb = get_max_pull_size_gb()
            if size is not None and size / 1e9 > max_gb and not confirm:
                raise HTTPException(
                    400,
                    f"Model {tag!r} is {size/1e9:.1f}GB, above the {max_gb:g}GB VRAM "
                    "safety threshold. Refusing to silently pull a >VRAM model; "
                    "send {\"confirm\": true} to proceed with an explicit offload pull.",
                )

            logger.info(
                "modelbench pull tag=%s size_bytes=%s initiator=%s root=%s",
                tag, size, initiator, client.root,
            )

            started = {"event": "started", "model": tag, "size_bytes": size,
                       "max_pull_size_gb": max_gb}

            async def stream():
                try:
                    yield _sse(started)
                    async for ev in client.stream_pull(tag):
                        yield _sse(ev)
                        if isinstance(ev, dict) and ev.get("status") == "success":
                            yield _sse({"event": "done", "ok": True})
                            return
                except OllamaPullError as e:
                    logger.warning("modelbench pull %s failed: %s", tag, e)
                    yield _sse({"event": "error", "error": str(e)})
                    yield _sse({"event": "done", "ok": False})
                finally:
                    await client.aclose()
                    _pull_gate.release()

            return StreamingResponse(
                stream(), media_type=_SSE_MEDIA_TYPE,
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except HTTPException:
            if client is not None:
                await client.aclose()
            _pull_gate.release()
            raise
        except Exception:
            if client is not None:
                await client.aclose()
            _pull_gate.release()
            raise

    return router
