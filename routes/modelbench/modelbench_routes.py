"""Read-only ModelBench dashboard backend.

Serves aggregate and raw views over the `bench_samples` table (ORM
`BenchSample` in core/database.py) — one row per ModelBench benchmark
request. GET-only, no per-user gating (bench data is global), mirroring
the read-only surface shape of routes/hwfit_routes.py.

Percentiles are ALWAYS recomputed from the raw rows fetched for the
current request (nearest-rank method) rather than stored or cached, so
the numbers reflect the live table on every call — this table is written
by the ModelBench harness out-of-band and can grow between requests.
"""

import math
from typing import Optional

from fastapi import APIRouter, HTTPException

from core.database import BenchSample, SessionLocal

_VRRAM_FIT_VALUES = {"fit", "partial", "offload"}

# The 11 fields that must all be non-null for a row to count as
# provenance-complete.
_PROVENANCE_FIELDS = (
    "run_id", "model_tag", "true_params", "quant", "ctx_len", "think",
    "prompt_bytes", "temperature", "seed", "ollama_version", "vrram_fit",
)

_SAMPLE_COLUMNS = (
    "run_id", "model_tag", "true_params", "quant", "ctx_len", "think",
    "prompt_bytes", "temperature", "seed", "ollama_version", "vrram_fit",
    "output_tokens", "thinking_tokens", "content_tokens", "tokens_per_sec",
    "ttft_ms", "latency_ms", "created_at",
)


def _nearest_rank_percentile(sorted_values, p):
    """Nearest-rank percentile: sorted ascending, rank = ceil(p/100 * n),
    1-indexed, value at that index."""
    n = len(sorted_values)
    rank = math.ceil(p / 100 * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def _provenance_complete(row) -> bool:
    return all(getattr(row, field) is not None for field in _PROVENANCE_FIELDS)


def _mode(values):
    """Most common non-null value, or None if no non-null values."""
    counts = {}
    for v in values:
        if v is None:
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _fit_split(rows):
    split = {"fit": 0, "partial": 0, "offload": 0}
    for r in rows:
        if r.vrram_fit in split:
            split[r.vrram_fit] += 1
    return split


def _row_to_dict(row):
    d = {}
    for col in _SAMPLE_COLUMNS:
        value = getattr(row, col)
        if col == "created_at" and value is not None:
            value = value.isoformat()
        d[col] = value
    d["provenance_complete"] = _provenance_complete(row)
    return d


def _validate_fit(fit: Optional[str]):
    if fit is not None and fit not in _VRRAM_FIT_VALUES:
        raise HTTPException(400, f"Invalid fit value: {fit!r} (must be one of {sorted(_VRRAM_FIT_VALUES)})")


def _parse_think(think: Optional[str]) -> Optional[bool]:
    if think is None:
        return None
    lowered = think.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise HTTPException(400, f"Invalid think value: {think!r} (must be 'true' or 'false')")


def _parse_percentiles(percentiles: str):
    parts = [p.strip() for p in percentiles.split(",") if p.strip()]
    if not parts:
        raise HTTPException(400, "percentiles must be a non-empty comma list")
    result = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise HTTPException(400, f"Invalid percentile value: {part!r}")
        if not (1 <= value <= 99):
            raise HTTPException(400, f"Percentile out of range (1..99): {value}")
        result.append(value)
    return result


def setup_modelbench_routes():
    router = APIRouter(prefix="/api/modelbench", tags=["modelbench"])

    @router.get("/models")
    def list_models(fit: Optional[str] = None):
        """Per-model_tag summary: sample counts, think/fit splits, ctx sweep,
        and provenance completeness, optionally restricted to one vrram_fit
        class."""
        _validate_fit(fit)
        db = SessionLocal()
        try:
            all_rows = db.query(BenchSample).all()
            zero_null_provenance = all(_provenance_complete(r) for r in all_rows)

            rows = [r for r in all_rows if fit is None or r.vrram_fit == fit]

            by_tag = {}
            for r in rows:
                by_tag.setdefault(r.model_tag, []).append(r)

            models = []
            for tag in sorted(by_tag.keys()):
                group = by_tag[tag]
                think_true = sum(1 for r in group if r.think is True)
                think_false = sum(1 for r in group if r.think is False)
                fit_rows_ctx = [r.ctx_len for r in group if r.vrram_fit == "fit" and r.ctx_len is not None]
                advertised = [r.ctx_len for r in group if r.ctx_len is not None]
                complete = sum(1 for r in group if _provenance_complete(r))
                models.append({
                    "model_tag": tag,
                    "true_params": _mode([r.true_params for r in group]),
                    "quant": _mode([r.quant for r in group]),
                    "sample_count": len(group),
                    "think": {"false": think_false, "true": think_true},
                    "fit_split": _fit_split(group),
                    "ctx": {
                        "advertised": max(advertised) if advertised else None,
                        "achieved": max(fit_rows_ctx) if fit_rows_ctx else None,
                        "achieved_swept": len(fit_rows_ctx) > 0,
                    },
                    "provenance": {"complete": complete, "total": len(group)},
                })

            meta = {
                "total_models": len(models),
                "total_rows": len(rows),
                "filter_fit": fit,
                "zero_null_provenance": zero_null_provenance,
            }
            return {"models": models, "meta": meta}
        finally:
            db.close()

    @router.get("/metrics")
    def get_metrics(model: str, think: Optional[str] = None, fit: Optional[str] = None,
                     percentiles: str = "50,90,95"):
        """Per-think-bucket metric distributions (tokens_per_sec, ttft_ms,
        latency_ms) for one model_tag, with percentiles recomputed from the
        raw rows on every call."""
        _validate_fit(fit)
        think_bool = _parse_think(think)
        percentile_list = _parse_percentiles(percentiles)

        db = SessionLocal()
        try:
            model_exists = db.query(BenchSample).filter(BenchSample.model_tag == model).first() is not None
            if not model_exists:
                raise HTTPException(404, f"No samples found for model: {model!r}")

            query = db.query(BenchSample).filter(BenchSample.model_tag == model)
            if think_bool is not None:
                query = query.filter(BenchSample.think == think_bool)
            if fit is not None:
                query = query.filter(BenchSample.vrram_fit == fit)
            rows = query.all()

            by_think = {}
            for r in rows:
                by_think.setdefault(r.think, []).append(r)

            groups = []
            for think_val in sorted(by_think.keys()):
                grows = by_think[think_val]
                metrics = {}
                for col in ("tokens_per_sec", "ttft_ms", "latency_ms"):
                    values = sorted(v for v in (getattr(r, col) for r in grows) if v is not None)
                    if not values:
                        continue
                    metrics[col] = {
                        "count": len(values),
                        "min": values[0],
                        "max": values[-1],
                        "mean": sum(values) / len(values),
                        "percentiles": {
                            f"p{p}": _nearest_rank_percentile(values, p) for p in percentile_list
                        },
                    }
                groups.append({
                    "think": think_val,
                    "count": len(grows),
                    "fit_split": _fit_split(grows),
                    "metrics": metrics,
                })

            return {
                "model": model,
                "percentiles": percentile_list,
                "filter": {"think": think_bool, "fit": fit},
                "groups": groups,
            }
        finally:
            db.close()

    @router.get("/samples")
    def list_samples(model: Optional[str] = None, think: Optional[str] = None,
                      fit: Optional[str] = None, limit: int = 200, offset: int = 0):
        """Raw bench_samples rows, most recent first."""
        _validate_fit(fit)
        think_bool = _parse_think(think)
        limit = max(0, min(limit, 1000))
        offset = max(0, offset)

        db = SessionLocal()
        try:
            query = db.query(BenchSample)
            if model is not None:
                query = query.filter(BenchSample.model_tag == model)
            if think_bool is not None:
                query = query.filter(BenchSample.think == think_bool)
            if fit is not None:
                query = query.filter(BenchSample.vrram_fit == fit)

            total = query.count()
            rows = (
                query.order_by(BenchSample.created_at.desc(), BenchSample.run_id.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return {
                "rows": [_row_to_dict(r) for r in rows],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        finally:
            db.close()

    @router.get("/samples/{run_id}")
    def get_sample(run_id: str):
        """Single raw sample row (drill-down / re-verification)."""
        db = SessionLocal()
        try:
            row = db.query(BenchSample).filter(BenchSample.run_id == run_id).first()
            if row is None:
                raise HTTPException(404, f"No sample found for run_id: {run_id!r}")
            return _row_to_dict(row)
        finally:
            db.close()

    return router
