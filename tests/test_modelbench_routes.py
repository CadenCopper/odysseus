"""Route-level tests for the read-only ModelBench dashboard backend
(GET /api/modelbench/*).

Seeds a temp file-backed sqlite DB with a realistic mix of ``bench_samples``
rows (a Q4_K_M 9B model swept across three context lengths, both think
states, plus one offload 27B row) and monkeypatches the route module's
``SessionLocal`` to point at it, per the repo's DB-route-test convention
(see tests/test_caldav_writeback_route.py).
"""

import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette.testclient")

from fastapi import FastAPI
from starlette.testclient import TestClient

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

from core.database import Base, BenchSample
import routes.modelbench.modelbench_routes as mbroutes

BASE_TIME = datetime(2026, 1, 1, 0, 0, 0)


def _nearest_rank(sorted_values, p):
    """Independent (test-side) nearest-rank percentile, used to prove the
    route's recomputed percentiles rather than just echoing its own helper."""
    n = len(sorted_values)
    rank = math.ceil(p / 100 * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def _row(run_id, model_tag, true_params, quant, ctx_len, think, vrram_fit,
         tokens_per_sec, ttft_ms, latency_ms, created_offset, seed):
    return BenchSample(
        run_id=run_id,
        model_tag=model_tag,
        true_params=true_params,
        quant=quant,
        ctx_len=ctx_len,
        think=think,
        prompt_bytes=100,
        temperature=0.7,
        seed=seed,
        ollama_version="0.5.1",
        vrram_fit=vrram_fit,
        output_tokens=50,
        thinking_tokens=0,
        content_tokens=50,
        tokens_per_sec=tokens_per_sec,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        created_at=BASE_TIME + timedelta(seconds=created_offset),
    )


@pytest.fixture(scope="module")
def seeded_db():
    SessionLocal, engine, tmpfile = make_temp_sqlite(Base.metadata)
    db = SessionLocal()
    try:
        rows = [
            # defiant-fable:latest, think=False (5 rows) — ctx sweep 116736
            # fit / 131072 partial / 262144 partial.
            _row("df-001", "defiant-fable:latest", 9.0, "Q4_K_M", 116736, False, "fit", 10.0, 100.0, 1000.0, 1, seed=1),
            _row("df-002", "defiant-fable:latest", 9.0, "Q4_K_M", 116736, False, "fit", 20.0, 200.0, 1010.0, 2, seed=2),
            _row("df-003", "defiant-fable:latest", 9.0, "Q4_K_M", 131072, False, "partial", 30.0, 300.0, 1020.0, 3, seed=3),
            # df-004 has a NULL seed -> provenance_complete must be False.
            _row("df-004", "defiant-fable:latest", 9.0, "Q4_K_M", 131072, False, "partial", 40.0, 400.0, 1030.0, 4, seed=None),
            _row("df-005", "defiant-fable:latest", 9.0, "Q4_K_M", 262144, False, "partial", 50.0, 500.0, 1040.0, 5, seed=5),
            # defiant-fable:latest, think=True (5 rows)
            _row("df-006", "defiant-fable:latest", 9.0, "Q4_K_M", 116736, True, "fit", 15.0, 150.0, 1500.0, 6, seed=6),
            _row("df-007", "defiant-fable:latest", 9.0, "Q4_K_M", 131072, True, "partial", 25.0, 250.0, 1510.0, 7, seed=7),
            _row("df-008", "defiant-fable:latest", 9.0, "Q4_K_M", 131072, True, "partial", 35.0, 350.0, 1520.0, 8, seed=8),
            _row("df-009", "defiant-fable:latest", 9.0, "Q4_K_M", 262144, True, "partial", 45.0, 450.0, 1530.0, 9, seed=9),
            _row("df-010", "defiant-fable:latest", 9.0, "Q4_K_M", 262144, True, "partial", 55.0, 550.0, 1540.0, 10, seed=10),
            # grand-oracle:27b — single offload row.
            _row("go-001", "grand-oracle:27b", 27.3, "Q4_K_M", 131072, False, "offload", 5.0, 50.0, 2000.0, 11, seed=11),
        ]
        db.add_all(rows)
        db.commit()
    finally:
        db.close()
    yield SessionLocal
    engine.dispose()
    Path(tmpfile.name).unlink(missing_ok=True)


@pytest.fixture
def client(seeded_db, monkeypatch):
    monkeypatch.setattr(mbroutes, "SessionLocal", seeded_db)
    app = FastAPI()
    app.include_router(mbroutes.setup_modelbench_routes())
    return TestClient(app, raise_server_exceptions=False)


# --- GET /api/modelbench/models -------------------------------------------

def test_models_lists_all_models_sorted_with_expected_aggregates(client):
    r = client.get("/api/modelbench/models")
    assert r.status_code == 200
    body = r.json()

    tags = [m["model_tag"] for m in body["models"]]
    assert tags == ["defiant-fable:latest", "grand-oracle:27b"]

    fable = body["models"][0]
    assert fable["true_params"] == 9.0
    assert fable["quant"] == "Q4_K_M"
    assert fable["sample_count"] == 10
    assert fable["think"] == {"false": 5, "true": 5}
    assert fable["fit_split"] == {"fit": 3, "partial": 7, "offload": 0}
    assert fable["ctx"] == {"advertised": 262144, "achieved": 116736, "achieved_swept": True}
    assert fable["provenance"] == {"complete": 9, "total": 10}

    oracle = body["models"][1]
    assert oracle["true_params"] == 27.3
    assert oracle["sample_count"] == 1
    assert oracle["ctx"]["achieved"] is None
    assert oracle["ctx"]["achieved_swept"] is False
    assert oracle["provenance"] == {"complete": 1, "total": 1}

    assert body["meta"] == {
        "total_models": 2, "total_rows": 11, "filter_fit": None, "zero_null_provenance": False,
    }


def test_models_fit_filter_restricts_counts_and_meta(client):
    r = client.get("/api/modelbench/models", params={"fit": "fit"})
    assert r.status_code == 200
    body = r.json()

    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["model_tag"] == "defiant-fable:latest"
    assert m["sample_count"] == 3
    assert m["think"] == {"false": 2, "true": 1}
    assert m["fit_split"] == {"fit": 3, "partial": 0, "offload": 0}
    assert m["ctx"] == {"advertised": 116736, "achieved": 116736, "achieved_swept": True}

    assert body["meta"]["filter_fit"] == "fit"
    assert body["meta"]["total_rows"] == 3
    assert body["meta"]["total_models"] == 1


def test_models_invalid_fit_returns_400(client):
    r = client.get("/api/modelbench/models", params={"fit": "bogus"})
    assert r.status_code == 400


# --- GET /api/modelbench/metrics -------------------------------------------

def test_metrics_default_percentiles_recomputed_and_grouped_by_think(client):
    r = client.get("/api/modelbench/metrics", params={"model": "defiant-fable:latest"})
    assert r.status_code == 200
    body = r.json()

    assert body["model"] == "defiant-fable:latest"
    assert body["percentiles"] == [50, 90, 95]
    assert body["filter"] == {"think": None, "fit": None}
    assert len(body["groups"]) == 2

    false_group = next(g for g in body["groups"] if g["think"] is False)
    true_group = next(g for g in body["groups"] if g["think"] is True)

    false_values = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    expected_false = {f"p{p}": _nearest_rank(false_values, p) for p in (50, 90, 95)}
    assert false_group["count"] == 5
    assert false_group["fit_split"] == {"fit": 2, "partial": 3, "offload": 0}
    tps = false_group["metrics"]["tokens_per_sec"]
    assert tps["count"] == 5
    assert tps["min"] == 10.0
    assert tps["max"] == 50.0
    assert tps["mean"] == sum(false_values) / 5
    assert tps["percentiles"] == expected_false

    true_values = sorted([15.0, 25.0, 35.0, 45.0, 55.0])
    expected_true = {f"p{p}": _nearest_rank(true_values, p) for p in (50, 90, 95)}
    assert true_group["count"] == 5
    assert true_group["fit_split"] == {"fit": 1, "partial": 4, "offload": 0}
    assert true_group["metrics"]["tokens_per_sec"]["percentiles"] == expected_true


def test_metrics_think_filter_returns_single_group(client):
    r = client.get("/api/modelbench/metrics", params={"model": "defiant-fable:latest", "think": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["filter"] == {"think": True, "fit": None}
    assert len(body["groups"]) == 1
    assert body["groups"][0]["think"] is True
    assert body["groups"][0]["count"] == 5


def test_metrics_fit_filter_excluding_all_rows_returns_empty_groups(client):
    r = client.get("/api/modelbench/metrics", params={"model": "defiant-fable:latest", "fit": "offload"})
    assert r.status_code == 200
    body = r.json()
    assert body["groups"] == []


def test_metrics_unknown_model_returns_404(client):
    r = client.get("/api/modelbench/metrics", params={"model": "does-not-exist:latest"})
    assert r.status_code == 404


def test_metrics_invalid_fit_returns_400(client):
    r = client.get("/api/modelbench/metrics", params={"model": "defiant-fable:latest", "fit": "bogus"})
    assert r.status_code == 400


def test_metrics_invalid_percentiles_returns_400(client):
    r = client.get(
        "/api/modelbench/metrics",
        params={"model": "defiant-fable:latest", "percentiles": "0,150"},
    )
    assert r.status_code == 400


# --- GET /api/modelbench/samples -------------------------------------------

def test_samples_list_filters_and_paginates(client):
    r = client.get(
        "/api/modelbench/samples",
        params={"model": "defiant-fable:latest", "limit": 3, "offset": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 10
    assert body["limit"] == 3
    assert body["offset"] == 2
    # Sorted created_at desc: df-010 .. df-001; offset 2 skips df-010, df-009.
    assert [row["run_id"] for row in body["rows"]] == ["df-008", "df-007", "df-006"]


def test_samples_list_surfaces_provenance_complete_per_row(client):
    r = client.get("/api/modelbench/samples", params={"model": "defiant-fable:latest", "limit": 200})
    body = r.json()
    df004 = next(row for row in body["rows"] if row["run_id"] == "df-004")
    assert df004["seed"] is None
    assert df004["provenance_complete"] is False
    df003 = next(row for row in body["rows"] if row["run_id"] == "df-003")
    assert df003["provenance_complete"] is True


def test_samples_list_think_and_fit_filters(client):
    r = client.get(
        "/api/modelbench/samples",
        params={"model": "defiant-fable:latest", "think": "false", "fit": "partial"},
    )
    body = r.json()
    assert body["total"] == 3
    assert {row["run_id"] for row in body["rows"]} == {"df-003", "df-004", "df-005"}


# --- GET /api/modelbench/samples/{run_id} -----------------------------------

def test_sample_detail_returns_full_row_with_provenance_flag(client):
    r = client.get("/api/modelbench/samples/df-004")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "df-004"
    assert body["model_tag"] == "defiant-fable:latest"
    assert body["seed"] is None
    assert body["provenance_complete"] is False
    assert body["created_at"] == (BASE_TIME + timedelta(seconds=4)).isoformat()


def test_sample_detail_unknown_run_id_returns_404(client):
    r = client.get("/api/modelbench/samples/does-not-exist")
    assert r.status_code == 404


# --- read-only proof ---------------------------------------------------

def test_router_exposes_only_get_routes():
    router = mbroutes.setup_modelbench_routes()
    assert len(router.routes) > 0
    for route in router.routes:
        methods = getattr(route, "methods", set()) or set()
        assert not (methods & {"POST", "PUT", "DELETE", "PATCH"}), (
            f"mutating method found on {getattr(route, 'path', route)}: {methods}"
        )
