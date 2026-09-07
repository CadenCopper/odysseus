"""FastAPI routes fronting the single-run BenchJobRegistry + runner engine.

``POST   /api/modelbench/runs``                  — validate + start a bench run
``GET    /api/modelbench/runs``                  — list all runs (job-history panel)
``GET    /api/modelbench/runs/{job_id}``         — poll one run's status/progress
``POST   /api/modelbench/runs/{job_id}/cancel``  — cooperative cancel

Auth/CSP posture (mirrors ollama_routes / the runs-API card): every endpoint
calls ``require_authenticated_request`` first; none are in the auth-exempt
path — these are mutating/resource-consuming and must not be reachable
unauthenticated.

"Unknown model_tag" is rejected here against the live ollama resident list
(via an injectable client factory, mirroring ``ollama_routes``) rather than
relying solely on the runner's own live check at dispatch time, so the GUI
gets an immediate 4xx instead of a job that fails after being queued.
"""

import logging
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Request

from services.modelbench.job_registry import ConcurrentJobError
from services.modelbench.ollama_client import OllamaClient, OllamaUnreachable
from services.modelbench.runner import (
    MAX_CTX_SWEEP_PROBES,
    MAX_SAMPLES,
    MIN_SAMPLES,
    effective_ctx_cap,
)
from src.auth_helpers import require_authenticated_request

logger = logging.getLogger(__name__)

# Poll contract (card t_12dab7b4): status/progress/message/run_id/error/
# heartbeat_at are required; job_id + the submitted params are fine to include.
_POLL_FIELDS = (
    "job_id", "status", "progress", "message", "run_id", "error",
    "heartbeat_at", "model_tag", "think", "ctx_target", "ctx_sweep",
    "n_samples",
)


def _default_client_factory() -> OllamaClient:
    return OllamaClient()


def _bad_request(message: str) -> HTTPException:
    return HTTPException(400, message)


async def _resolve_resident_model(factory: Callable[[], OllamaClient], model_tag: str) -> dict:
    """Fetch the live resident model list and return `model_tag`'s entry.

    Raises HTTPException(502) on OllamaUnreachable, HTTPException(404) when
    `model_tag` is not resident.
    """
    client = factory()
    try:
        models = await client.tags()
    except OllamaUnreachable as e:
        raise HTTPException(502, str(e))
    finally:
        await client.aclose()
    for m in models:
        name = m.get("name") or m.get("model")
        if name == model_tag:
            return m
    raise HTTPException(404, f"unknown model_tag {model_tag!r}: not resident in ollama")


def _validate_start_body(body: dict) -> None:
    """Validate the fields that don't require the live resident model list."""
    model_tag = body.get("model_tag")
    if not isinstance(model_tag, str) or not model_tag.strip():
        raise _bad_request("model_tag is required and must be a non-empty string")

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _bad_request("prompt is required and must be a non-empty string")

    think = body.get("think")
    if not isinstance(think, bool):
        raise _bad_request("think is required and must be a boolean")

    n_samples = body.get("n_samples")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise _bad_request("n_samples is required and must be an integer")
    if not (MIN_SAMPLES <= n_samples <= MAX_SAMPLES):
        raise _bad_request(f"n_samples must be between {MIN_SAMPLES} and {MAX_SAMPLES}")

    ctx_target = body.get("ctx_target")
    if ctx_target is not None:
        if not isinstance(ctx_target, int) or isinstance(ctx_target, bool):
            raise _bad_request("ctx_target must be an integer")
        if ctx_target < 1:
            raise _bad_request("ctx_target must be >= 1")

    ctx_sweep = body.get("ctx_sweep")
    if ctx_sweep is not None:
        if not isinstance(ctx_sweep, list):
            raise _bad_request("ctx_sweep must be a list of integers")
        if len(ctx_sweep) > MAX_CTX_SWEEP_PROBES:
            raise _bad_request(
                f"ctx_sweep must have at most {MAX_CTX_SWEEP_PROBES} points"
            )
        for i, point in enumerate(ctx_sweep):
            if not isinstance(point, int) or isinstance(point, bool):
                raise _bad_request(f"ctx_sweep[{i}] must be an integer")
            if point < 1:
                raise _bad_request(f"ctx_sweep[{i}] must be >= 1")


def setup_runs_routes(
    bench_registry: Any,
    ollama_client_factory: Optional[Callable[[], OllamaClient]] = None,
) -> APIRouter:
    """Build the runs router bound to `bench_registry`.

    `ollama_client_factory` returns a fresh :class:`OllamaClient` per request
    (default: a real client); tests inject a fake to validate model_tag
    against a controlled resident list without touching a live ollama.
    """
    factory = ollama_client_factory or _default_client_factory
    router = APIRouter(prefix="/api/modelbench/runs", tags=["modelbench"])

    @router.post("", status_code=201)
    async def start_run(request: Request):
        require_authenticated_request(request)

        body: dict = {}
        try:
            raw = await request.json()
            if isinstance(raw, dict):
                body = raw
        except Exception:
            body = {}

        _validate_start_body(body)

        model_tag = body["model_tag"]
        prompt = body["prompt"]
        think = body["think"]
        n_samples = body["n_samples"]
        ctx_target = body.get("ctx_target")
        ctx_sweep = body.get("ctx_sweep")

        resident = await _resolve_resident_model(factory, model_tag)

        if ctx_target is not None or ctx_sweep is not None:
            details = resident.get("details") or {}
            cap = effective_ctx_cap(model_tag, details.get("context_length"))
            if ctx_target is not None and ctx_target > cap:
                raise _bad_request(
                    f"ctx_target {ctx_target} exceeds the model's safety cap ({cap})"
                )
            # Over-cap sweep points are CLAMPED to the model's effective cap
            # (not rejected) so an explicit series is honoured as far as the
            # model can go. Structural validation (ints >= 1, <= max probes)
            # already ran in _validate_start_body.
            if ctx_sweep is not None:
                ctx_sweep = [min(point, cap) for point in ctx_sweep]

        try:
            job = await bench_registry.submit(
                model_tag=model_tag,
                think=think,
                ctx_target=ctx_target,
                ctx_sweep=ctx_sweep,
                prompt=prompt,
                n_samples=n_samples,
            )
        except ConcurrentJobError as e:
            raise HTTPException(409, str(e))

        return job

    @router.get("")
    async def list_runs(request: Request):
        require_authenticated_request(request)
        return {"runs": await bench_registry.list()}

    @router.get("/{job_id}")
    async def get_run(job_id: str, request: Request):
        require_authenticated_request(request)
        job = await bench_registry.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job_id {job_id!r}")
        return {k: job.get(k) for k in _POLL_FIELDS}

    @router.post("/{job_id}/cancel")
    async def cancel_run(job_id: str, request: Request):
        require_authenticated_request(request)
        job = await bench_registry.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job_id {job_id!r}")
        await bench_registry.cancel(job_id)
        job = await bench_registry.get(job_id)
        return {"job_id": job_id, "status": job["status"]}

    return router
