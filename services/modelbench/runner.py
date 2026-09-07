"""CONTAINER-NATIVE ModelBench runner engine.

The engine attached to services.modelbench.job_registry.BenchJobRegistry via
``attach_runner``. Reads a queued BenchJob row, drives a native ollama /api/chat
token test plus a bounded ctx-length sweep, and inserts the resulting raw
samples into bench_samples (core.database.BenchSample) in a single idempotent
transaction.

Pure helpers (NDJSON stream parsing, metric computation, token estimation,
vrram_fit derivation, parameter-size parsing, ctx sweep point generation) are
module-level functions so they're unit-testable without a network client.
"""

import asyncio
import contextlib
import logging
import math
import os
import re
import time
import uuid

import httpx

from core import database as core_db
from core.database import BenchJob, BenchSample
from services.modelbench.job_registry import JobContext, decode_ctx_sweep

logger = logging.getLogger(__name__)

# The P-file's recorded 10.0.0.188 serves a different, near-empty ollama
# instance (research-verified: only gemma3:4b). The real copper-wsl ollama
# (0.32.13, all 5 Qwen models + gpt-oss) is at 10.0.0.8. The bench container's
# recorded OLLAMA_URL env points at 10.0.0.188 -- a deployment trap -- so we
# NEVER honor that host; an env pointing at the known-empty 10.0.0.188 is
# overridden to the verified 10.0.0.8 with a warning, and any other explicit
# env is respected.
def _resolve_ollama_url() -> str:
    env = os.getenv("OLLAMA_URL")
    if env:
        host = env.split("/")[-1].split(":")[0]
        if host == "10.0.0.188":
            logger.warning(
                "OLLAMA_URL=%s points at the known-empty bench instance "
                "(10.0.0.188 serves only gemma3:4b); overriding to "
                "http://10.0.0.8:11434 (the verified copper-wsl ollama)",
                env,
            )
            return "http://10.0.0.8:11434"
        return env.rstrip("/")
    return "http://10.0.0.8:11434"


OLLAMA_URL = _resolve_ollama_url()

DEFAULT_TIMEOUT_S = float(os.getenv("MB_RUNNER_TIMEOUT_S", "120"))
HEARTBEAT_INTERVAL_S = 15.0

MIN_SAMPLES = 1
MAX_SAMPLES = 20
DEFAULT_CTX_TARGET = 4096
MAX_CTX_SWEEP_PROBES = 12

# Deviation from spec: the exact N formula for num_predict is unspecified;
# a fixed conservative cap keeps token-test generations bounded.
DEFAULT_NUM_PREDICT = 512

# Per-model ctx sweep safety caps (research table). Keys are matched as
# case-insensitive substrings of the model tag. Anything unmatched -- including
# the two 27B tags -- falls back to LOW_CTX_CAP (offload-only, conservative).
MODEL_CTX_CAPS = {
    ":defiant-fable:": 116736,
    "qwen3-14b-ctx": 40960,
    "qwen3:14b": 40960,
}
LOW_CTX_CAP = 8192

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_PARAM_SIZE_RE = re.compile(r"([\d.]+)\s*B", re.IGNORECASE)

_VRRAM_FIT_VALUES = ("fit", "partial", "offload")

# Identity/config fields expected non-null on every successfully-inserted row.
# Deliberately excludes vrram_fit (may be legitimately NULL when ollama's
# /api/ps is unavailable) and the measured metrics (output_tokens etc, which
# describe the sample rather than its provenance).
_PROVENANCE_FIELDS = (
    "run_id", "model_tag", "true_params", "quant", "ctx_len", "think",
    "prompt_bytes", "temperature", "seed", "ollama_version", "created_at",
)


def estimate_tokens(text: str) -> int:
    """Deterministic, pure token-count estimate. Empty text -> 0."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def split_think_tags(content_text: str):
    """Strip <think>...</think> blocks out of content into a thinking string.

    Covers content-templated models (defiant-fable, the 27B pair) that embed
    reasoning inline instead of emitting a separate message.thinking channel.
    Returns (extracted_thinking, remaining_content).
    """
    extracted = []

    def _capture(m):
        extracted.append(m.group(1))
        return ""

    remaining = _THINK_TAG_RE.sub(_capture, content_text)
    return "".join(extracted), remaining


def parse_true_params(parameter_size) -> float | None:
    """Parse details.parameter_size (e.g. "27.3B") -> 27.3. Never derive from tag."""
    if not parameter_size:
        return None
    m = _PARAM_SIZE_RE.search(str(parameter_size))
    if not m:
        return None
    return float(m.group(1))


def vrram_fit_from_ps(ps_json, model_tag: str):
    """Derive vrram_fit from a GET /api/ps payload for `model_tag`.

    Returns (vrram_fit_or_None, provenance_incomplete). /api/ps lists only
    loaded models; a missing/malformed payload or missing model is
    provenance-incomplete (vrram_fit NULL), not an error.
    """
    if not ps_json:
        return None, True
    models = ps_json.get("models") or []
    base_tag = (model_tag or "").split(":")[0]
    match = None
    for m in models:
        name = m.get("name") or m.get("model") or ""
        if name == model_tag or name.split(":")[0] == base_tag:
            match = m
            break
    if match is None:
        return None, True
    size = match.get("size") or 0
    size_vram = match.get("size_vram")
    if size_vram is None or size <= 0:
        return None, True
    ratio = size_vram / size
    if ratio >= 1.0:
        return "fit", False
    if ratio <= 0:
        return "offload", False
    return "partial", False


def parse_ndjson_stream(raw: bytes):
    """Parse a streamed /api/chat NDJSON body.

    Accumulates message.thinking deltas into thinking_text and message.content
    deltas into content_text, then splits any <think>...</think> blocks out of
    content_text into thinking_text (content-templated models). Returns
    (thinking_text, content_text, done_event_or_None).
    """
    import json

    thinking_parts = []
    content_parts = []
    done_event = None
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        message = chunk.get("message") or {}
        thinking_parts.append(message.get("thinking") or "")
        content_parts.append(message.get("content") or "")
        if chunk.get("done"):
            done_event = chunk

    thinking_text = "".join(thinking_parts)
    content_text = "".join(content_parts)
    extra_thinking, content_text = split_think_tags(content_text)
    thinking_text += extra_thinking
    return thinking_text, content_text, done_event


def compute_metrics(done_event: dict, thinking_text: str, content_text: str) -> dict:
    """Compute the per-sample metric set from a done:true event + accumulated text."""
    eval_count = done_event.get("eval_count") or 0
    eval_duration = done_event.get("eval_duration")
    prompt_eval_duration = done_event.get("prompt_eval_duration")
    total_duration = done_event.get("total_duration")

    tokens_per_sec = None
    if eval_duration:
        tokens_per_sec = eval_count / (eval_duration / 1e9)

    ttft_ms = (prompt_eval_duration / 1e6) if prompt_eval_duration is not None else None
    latency_ms = (total_duration / 1e6) if total_duration is not None else None

    return {
        "output_tokens": eval_count,
        "ttft_ms": ttft_ms,
        "latency_ms": latency_ms,
        "tokens_per_sec": tokens_per_sec,
        "thinking_tokens": estimate_tokens(thinking_text),
        "content_tokens": estimate_tokens(content_text),
    }


def default_ctx_cap(model_tag: str) -> int:
    """Per-model ctx sweep safety cap from the research table (substring match)."""
    tag = (model_tag or "").lower()
    for key, cap in MODEL_CTX_CAPS.items():
        if key.lower() in tag:
            return cap
    return LOW_CTX_CAP


def effective_ctx_cap(model_tag: str, details_context_length) -> int:
    """default_ctx_cap bounded by the model's own advertised context_length."""
    cap = default_ctx_cap(model_tag)
    if details_context_length:
        cap = min(cap, details_context_length)
    return cap


def resolve_ctx_target(ctx_target, cap: int) -> int:
    """Clamp a requested ctx_target to the effective cap; default when None."""
    if ctx_target is None:
        return min(DEFAULT_CTX_TARGET, cap)
    return min(ctx_target, cap)


def ctx_sweep_points(cap: int, max_probes: int = MAX_CTX_SWEEP_PROBES) -> list:
    """Bounded doubling sequence (1024, 2048, ...) up to `cap`, capped at max_probes."""
    if cap <= 0:
        return []
    points = []
    v = 1024
    while v < cap and len(points) < max_probes:
        points.append(v)
        v *= 2
    if len(points) < max_probes and (not points or points[-1] != cap):
        points.append(cap)
    return points[:max_probes]


def resolve_sweep_points(sweep_raw, cap: int) -> list:
    """Pick the ctx sweep points for run_bench's sweep loop.

    When the BenchJob row carries a user-controllable ``ctx_sweep`` (JSON list
    of ints), the sweep uses exactly those points -- sorted ascending, deduped,
    each clamped to the model's effective ``cap``, and bounded to
    ``MAX_CTX_SWEEP_PROBES`` entries. Otherwise it falls back to the
    backward-compatible doubling sequence ``ctx_sweep_points(cap)``. Tolerates
    a missing/empty/malformed value by degrading to the fallback.
    """
    sweep = decode_ctx_sweep(sweep_raw)
    if not sweep:
        return ctx_sweep_points(cap)
    if cap and cap > 0:
        norm = sorted({min(int(p), cap) for p in sweep if isinstance(p, int) and p >= 1})
    else:
        norm = sorted({int(p) for p in sweep if isinstance(p, int) and p >= 1})
    norm = norm[:MAX_CTX_SWEEP_PROBES]
    if not norm:
        return ctx_sweep_points(cap)
    return norm


def sample_run_id(base_run_id: str, label: str) -> str:
    """A distinct, deterministic bench_samples.run_id (PK) for one sample.

    bench_samples.run_id is a per-request primary key, not a per-job key --
    the job-level base_run_id (attached via ctx.set_run_id) fans out into one
    distinct row id per sample so re-running the same job is idempotent.
    """
    return f"{base_run_id}:{label}"[:64]


async def _chat_once(client: httpx.AsyncClient, *, model_tag, prompt, ctx_len, think, timeout):
    options = {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": DEFAULT_NUM_PREDICT,
        "num_ctx": ctx_len,
    }
    payload = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": options,
    }
    if think is not None:
        payload["think"] = think
    resp = await client.post("/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.content


async def _fetch_model_info(client: httpx.AsyncClient, model_tag: str):
    """GET /api/tags, find `model_tag`. Returns None if the model is missing."""
    resp = await client.get("/api/tags")
    resp.raise_for_status()
    data = resp.json()
    for m in data.get("models", []) or []:
        if m.get("name") == model_tag:
            details = m.get("details") or {}
            return {
                "true_params": parse_true_params(details.get("parameter_size")),
                "quant": details.get("quantization_level"),
                "ctx_cap": details.get("context_length"),
                "supports_thinking": "thinking" in (m.get("capabilities") or []),
            }
    return None


async def _fetch_vrram_fit(client: httpx.AsyncClient, model_tag: str):
    """GET /api/ps -> vrram_fit_from_ps. Any transport/HTTP failure -> incomplete."""
    try:
        resp = await client.get("/api/ps")
    except Exception:
        return None, True
    if resp.status_code != 200:
        return None, True
    try:
        data = resp.json()
    except Exception:
        return None, True
    return vrram_fit_from_ps(data, model_tag)


async def _fetch_ollama_version(client: httpx.AsyncClient):
    """GET /api/version -> the server version string, or None on any failure.

    Deviation from spec: the spec's researched API contract does not mention
    this endpoint explicitly, but bench_samples.ollama_version needs a source;
    /api/version is ollama's standard version endpoint.
    """
    try:
        resp = await client.get("/api/version")
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json().get("version")
    except Exception:
        return None


async def _run_one_sample(
    client: httpx.AsyncClient,
    *,
    run_id,
    model_tag,
    model_info,
    prompt,
    ctx_len,
    think_payload_value,
    think_active,
    ollama_version,
    timeout,
) -> dict:
    raw = await _chat_once(
        client,
        model_tag=model_tag,
        prompt=prompt,
        ctx_len=ctx_len,
        think=think_payload_value,
        timeout=timeout,
    )
    thinking_text, content_text, done_event = parse_ndjson_stream(raw)
    if done_event is None:
        raise RuntimeError(f"ollama stream for {model_tag!r} never emitted done:true")

    metrics = compute_metrics(done_event, thinking_text, content_text)
    return {
        "run_id": run_id,
        "model_tag": model_tag,
        "true_params": model_info["true_params"],
        "quant": model_info["quant"],
        "ctx_len": ctx_len,
        "think": think_active,
        "prompt_bytes": len(prompt),
        "temperature": 0.0,
        "seed": 42,
        "ollama_version": ollama_version,
        "vrram_fit": None,
        "output_tokens": metrics["output_tokens"],
        "thinking_tokens": metrics["thinking_tokens"],
        "content_tokens": metrics["content_tokens"],
        "tokens_per_sec": metrics["tokens_per_sec"],
        "ttft_ms": metrics["ttft_ms"],
        "latency_ms": metrics["latency_ms"],
        "created_at": core_db.utcnow_naive(),
        "prompt_text": prompt,
        "collector": "gui-runner",
    }


async def insert_samples_idempotent(rows: list) -> int:
    """Insert `rows` (BenchSample field-dicts) in one transaction.

    Pre-queries existing run_ids and inserts only the rows not already
    present (idempotence: re-running the same run_id set inserts nothing on
    the second pass). On any exception the transaction rolls back and the
    exception re-raises, so zero partial rows ever land. Returns the count of
    rows actually inserted.
    """
    if not rows:
        return 0
    run_ids = [r["run_id"] for r in rows]
    db = core_db.SessionLocal()
    try:
        existing = {
            r[0]
            for r in db.query(BenchSample.run_id).filter(BenchSample.run_id.in_(run_ids)).all()
        }
        new_rows = [BenchSample(**r) for r in rows if r["run_id"] not in existing]
        if new_rows:
            db.add_all(new_rows)
            db.commit()
        return len(new_rows)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def _heartbeat_loop(ctx: "JobContext"):
    """Bump the job's heartbeat every HEARTBEAT_INTERVAL_S so a healthy long
    ctx sweep survives the registry's stale-job supervisor (30min timeout)."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        await ctx.heartbeat()


async def run_bench(ctx: "JobContext", *, client: httpx.AsyncClient | None = None) -> None:
    """Engine entrypoint passed to BenchJobRegistry.attach_runner().

    Reads job params off the BenchJob row for ctx.job_id, runs a token test
    plus a bounded ctx sweep against ollama, and inserts the resulting samples
    transactionally. Raises on any failure (unknown model, sample error,
    JobCancelled) so the registry can mark the job failed/cancelled with zero
    partial rows ever committed.
    """
    own_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=DEFAULT_TIMEOUT_S)
    try:
        db = core_db.SessionLocal()
        try:
            job = db.query(BenchJob).filter(BenchJob.id == ctx.job_id).first()
            if job is None:
                return
            model_tag = job.model_tag
            think_pref = job.think
            ctx_target_req = job.ctx_target
            ctx_sweep_req = job.ctx_sweep
            prompt = job.prompt
            n_samples = job.n_samples
        finally:
            db.close()

        n_samples = max(MIN_SAMPLES, min(MAX_SAMPLES, n_samples))

        base_run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        await ctx.set_run_id(base_run_id)

        heartbeat_task = asyncio.ensure_future(_heartbeat_loop(ctx))
        try:
            model_info = await _fetch_model_info(client, model_tag)
            if model_info is None:
                raise ValueError(f"unknown model {model_tag!r}: not present in GET /api/tags")

            cap = effective_ctx_cap(model_tag, model_info["ctx_cap"])
            ctx_target = resolve_ctx_target(ctx_target_req, cap)

            think_requested = True if think_pref is None else bool(think_pref)
            supports_thinking = model_info["supports_thinking"]
            # `think` in the row is the REQUESTED think class (the GUI's toggle),
            # not whether the model natively toggled it -- the real samples.db
            # records think=true rows for content-templated defiant-fable (17/5
            # split). Gating the recorded class on the native `thinking`
            # capability would silently move every defiant-fable think=true
            # sample into the think=false bucket (R3: never conflate classes).
            think_active = think_requested
            # Payload: send top-level `think` only for native-thinking models.
            # For content-templated models (defiant-fable, 27B pair) OMIT it --
            # a genuinely non-thinking model rejects think:true with HTTP 400
            # (research-verified on gemma3:4b). Their templated thinking is
            # split out of content by split_think_tags instead.
            think_payload_value = think_requested if supports_thinking else None

            ollama_version = await _fetch_ollama_version(client)

            rows = []

            # TOKEN TEST
            for i in range(n_samples):
                await ctx.check_cancelled()
                row = await _run_one_sample(
                    client,
                    run_id=sample_run_id(base_run_id, f"tok{i}"),
                    model_tag=model_tag,
                    model_info=model_info,
                    prompt=prompt,
                    ctx_len=ctx_target,
                    think_payload_value=think_payload_value,
                    think_active=think_active,
                    ollama_version=ollama_version,
                    timeout=DEFAULT_TIMEOUT_S,
                )
                rows.append(row)
                await ctx.set_progress(
                    (i + 1) / (n_samples * 2), f"token test {i + 1}/{n_samples}"
                )

            # CTX SWEEP — uses the user-controllable ctx_sweep when the job
            # carries one; otherwise the backward-compatible doubling fallback.
            sweep_points = resolve_sweep_points(ctx_sweep_req, cap)
            baseline_fit = None
            for j, sweep_ctx_len in enumerate(sweep_points):
                await ctx.check_cancelled()
                try:
                    row = await _run_one_sample(
                        client,
                        run_id=sample_run_id(base_run_id, f"ctx{sweep_ctx_len}"),
                        model_tag=model_tag,
                        model_info=model_info,
                        prompt=prompt,
                        ctx_len=sweep_ctx_len,
                        think_payload_value=think_payload_value,
                        think_active=think_active,
                        ollama_version=ollama_version,
                        timeout=DEFAULT_TIMEOUT_S,
                    )
                except Exception as e:
                    logger.warning(
                        f"ctx sweep probe at {sweep_ctx_len} failed, stopping sweep early: {e}"
                    )
                    break
                rows.append(row)

                probe_fit, _ = await _fetch_vrram_fit(client, model_tag)
                if baseline_fit is None:
                    baseline_fit = probe_fit
                elif probe_fit != baseline_fit:
                    break

                await ctx.set_progress(
                    0.5 + (j + 1) / (2 * max(1, len(sweep_points))),
                    f"ctx sweep {j + 1}/{len(sweep_points)}",
                )

            final_vrram_fit, provenance_incomplete = await _fetch_vrram_fit(client, model_tag)
            if provenance_incomplete:
                logger.warning(f"vrram_fit unavailable for {model_tag!r}: provenance incomplete")
            for row in rows:
                row["vrram_fit"] = final_vrram_fit

            await insert_samples_idempotent(rows)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
    finally:
        if own_client:
            await client.aclose()
