"""Tests for scripts/import_modelbench_samples.py.

Self-contained: builds a temp sqlite3 source `samples` table matching the
real ModelBench schema, imports it into a temp SQLAlchemy target via the
importer's own run_import(), and asserts on the target contents. No network,
no real app database — the isolated target is a throwaway tempfile engine.
"""
import sqlite3
import tempfile
from pathlib import Path

from tests.helpers.import_state import clear_fake_database_modules

clear_fake_database_modules()

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.database import BenchSample
from scripts.import_modelbench_samples import run_import

SOURCE_COLUMNS = [
    "run_id", "model_tag", "true_params", "quant", "ctx_len", "think",
    "prompt_bytes", "temperature", "seed", "ollama_version", "vrram_fit",
    "output_tokens", "thinking_tokens", "content_tokens", "tokens_per_sec",
    "ttft_ms", "latency_ms", "created_at",
]

# Source rows, in insertion order. run-008 has an invalid `think` value (2)
# and must be rejected without aborting the rest of the batch.
SOURCE_ROWS = [
    # run-001: think=True, vrram_fit already canonical, tokens sum exactly.
    ("run-001", "hf.co/models/model-a:Q4_K_M", 8.0, "Q4_K_M", 8192, 1,
     512, 0.7, 42, "0.4.0", "fit",
     120, 40, 80, 35.2, 120.5, 900.0, "2026-09-04T19:05:21.375Z"),
    # run-002: think=False, vrram_fit=offload, thinking_tokens=0.
    ("run-002", "hf.co/models/model-b:Q4_K_M", 14.0, "Q4_K_M", 4096, 0,
     256, 0.2, 7, "0.4.0", "offload",
     95, 0, 95, 12.4, 300.1, 2500.0, "2026-09-04T19:06:00Z"),
    # run-003: "defiant" row — output_tokens = thinking + content + 2.
    # Must NOT be rejected (no strict-equality enforcement).
    ("run-003", "hf.co/models/model-c:Q5_K_M", 27.0, "Q5_K_M", 8192, 1,
     1024, 0.9, 1, "0.4.1", "partial",
     92, 30, 60, 18.9, 500.0, 4000.0, "2026-09-04T19:07:11.500Z"),
    # run-004: vrram_fit alias "vram" -> should normalize to "fit".
    ("run-004", "hf.co/models/model-d:Q4_K_M", 8.0, "Q4_K_M", 8192, 1,
     512, 0.7, 42, "0.4.0", "vram",
     130, 45, 85, 33.0, 140.0, 950.0, "2026-09-04T19:08:00.100Z"),
    # run-005: true_params (27.3) must never be re-derived from the tag,
    # which itself looks like a (misleading) "3.8B" model.
    ("run-005", "hf.co/x/Qwen3.8-27B-OBLITERATED:Q4_K_M", 27.3, "Q4_K_M", 8192, 0,
     512, 0.7, 42, "0.4.0", "offload",
     88, 0, 88, 20.1, 200.0, 1800.0, "2026-09-04T19:09:00Z"),
    # run-006 / run-007: share model_tag/content, distinct run_ids.
    ("run-006", "hf.co/models/shared-tag:Q4_K_M", 7.0, "Q4_K_M", 4096, 0,
     256, 0.5, 3, "0.4.0", "fit",
     60, 0, 60, 40.0, 90.0, 700.0, "2026-09-04T19:10:00Z"),
    ("run-007", "hf.co/models/shared-tag:Q4_K_M", 7.0, "Q4_K_M", 4096, 0,
     256, 0.5, 3, "0.4.0", "fit",
     60, 0, 60, 40.0, 90.0, 700.0, "2026-09-04T19:10:05Z"),
    # run-008: invalid think value -> must be rejected, batch continues.
    ("run-008", "hf.co/models/model-e:Q4_K_M", 9.0, "Q4_K_M", 8192, 2,
     512, 0.7, 42, "0.4.0", "fit",
     100, 30, 70, 25.0, 150.0, 1000.0, "2026-09-04T19:11:00Z"),
]

VALID_ROW_COUNT = len(SOURCE_ROWS) - 1  # run-008 is rejected


def _make_source_db(tmp_path):
    path = str(tmp_path / "samples.db")
    conn = sqlite3.connect(path)
    cols_sql = ", ".join(SOURCE_COLUMNS)
    placeholders = ", ".join("?" for _ in SOURCE_COLUMNS)
    conn.execute(f"CREATE TABLE samples ({cols_sql})")
    conn.executemany(f"INSERT INTO samples ({cols_sql}) VALUES ({placeholders})", SOURCE_ROWS)
    conn.commit()
    conn.close()
    return path


def _make_target_engine(tmp_path):
    target_path = str(tmp_path / "target.db")
    return target_path, create_engine(f"sqlite:///{target_path}")


def _fetch_all(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        return {row.run_id: row for row in session.execute(select(BenchSample)).scalars()}
    finally:
        session.close()


def test_import_lands_rows_and_rejects_bad_think(tmp_path):
    source_path = _make_source_db(tmp_path)
    target_path, _ = _make_target_engine(tmp_path)

    counts = run_import(source_path, target_path)

    assert counts["source_read"] == len(SOURCE_ROWS)
    assert counts["inserted"] == VALID_ROW_COUNT
    assert counts["skipped_already_present"] == 0
    assert counts["failed"] == 1

    engine = create_engine(f"sqlite:///{target_path}")
    try:
        rows = _fetch_all(engine)
        assert len(rows) == VALID_ROW_COUNT
        assert "run-008" not in rows  # bad think value, rejected
        assert "run-007" in rows  # batch continued past the bad row
    finally:
        engine.dispose()


def test_import_is_idempotent(tmp_path):
    source_path = _make_source_db(tmp_path)
    target_path, _ = _make_target_engine(tmp_path)

    first = run_import(source_path, target_path)
    assert first["inserted"] == VALID_ROW_COUNT

    second = run_import(source_path, target_path)
    assert second["inserted"] == 0
    assert second["skipped_already_present"] == VALID_ROW_COUNT

    engine = create_engine(f"sqlite:///{target_path}")
    try:
        rows = _fetch_all(engine)
        assert len(rows) == VALID_ROW_COUNT  # no duplicates
    finally:
        engine.dispose()


def test_true_params_never_derived_from_tag(tmp_path):
    source_path = _make_source_db(tmp_path)
    target_path, _ = _make_target_engine(tmp_path)

    run_import(source_path, target_path)

    engine = create_engine(f"sqlite:///{target_path}")
    try:
        rows = _fetch_all(engine)
        row = rows["run-005"]
        assert row.model_tag == "hf.co/x/Qwen3.8-27B-OBLITERATED:Q4_K_M"
        # The tag looks like it encodes "3.8B" — true_params must stay the
        # authoritative source value (27.3), never re-derived from the tag.
        assert row.true_params == 27.3
    finally:
        engine.dispose()


def test_vrram_fit_normalization_and_bad_row_rejection(tmp_path):
    source_path = _make_source_db(tmp_path)
    target_path, _ = _make_target_engine(tmp_path)

    counts = run_import(source_path, target_path)
    assert counts["failed"] == 1

    engine = create_engine(f"sqlite:///{target_path}")
    try:
        rows = _fetch_all(engine)
        assert rows["run-004"].vrram_fit == "fit"  # alias "vram" -> "fit"
        assert rows["run-002"].vrram_fit == "offload"
        assert rows["run-003"].vrram_fit == "partial"
        assert "run-008" not in rows
    finally:
        engine.dispose()


def test_token_columns_stay_separate_never_summed(tmp_path):
    source_path = _make_source_db(tmp_path)
    target_path, _ = _make_target_engine(tmp_path)

    run_import(source_path, target_path)

    # No summed/total token column exists on the model at all.
    column_names = {c.name for c in BenchSample.__table__.columns}
    assert "total_tokens" not in column_names
    assert {"output_tokens", "thinking_tokens", "content_tokens"} <= column_names

    engine = create_engine(f"sqlite:///{target_path}")
    try:
        rows = _fetch_all(engine)

        think_false = rows["run-002"]
        assert think_false.think is False
        assert think_false.thinking_tokens == 0
        assert think_false.content_tokens == 95
        assert think_false.output_tokens == 95

        think_true = rows["run-001"]
        assert think_true.think is True
        assert think_true.thinking_tokens == 40
        assert think_true.content_tokens == 80
        assert think_true.output_tokens == 120

        # Defiant row: output_tokens deliberately != thinking + content,
        # and it must be stored verbatim (not recomputed/rejected).
        defiant = rows["run-003"]
        assert defiant.output_tokens == 92
        assert defiant.thinking_tokens == 30
        assert defiant.content_tokens == 60
        assert defiant.output_tokens != defiant.thinking_tokens + defiant.content_tokens
    finally:
        engine.dispose()
