#!/usr/bin/env python3
"""
import_modelbench_samples.py — standalone SQLite->SQLite importer for
ModelBench raw samples into the isolated `bench_samples` table
(core.database.BenchSample).

Reads a source `samples.db` (table `samples`, 18 columns, one row per
benchmark request) and copies rows into a --target SQLite database's
`bench_samples` table, normalizing a handful of source-specific encodings
along the way: `think` as 0/1 -> bool, `vrram_fit` aliases -> the canonical
fit/partial/offload enum, and the ISO-8601 `created_at` string -> a naive UTC
datetime. All other columns copy verbatim; `true_params` is never derived
from the model tag — it is read as-is from the source.

Isolation: this script NEVER reads or writes the app's global DATABASE_URL,
core.database.engine, or core.database.SessionLocal. `core.database` is
imported only to obtain the BenchSample model/metadata. A dedicated
SQLAlchemy engine is created for --target, and only the bench_samples table
is created/touched there.

Idempotent: bench_samples' primary key is run_id. Existing run_ids in
--target are pre-queried and only new rows are inserted, so re-running the
same source against the same target inserts 0 rows the second time.

Usage:
    python3 scripts/import_modelbench_samples.py --source /path/to/samples.db --target /path/to/target.db
    python3 scripts/import_modelbench_samples.py --source samples.db --target sqlite:///./bench.db --dry-run
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, select

from core.database import BenchSample

SOURCE_COLUMNS = [
    "run_id", "model_tag", "true_params", "quant", "ctx_len", "think",
    "prompt_bytes", "temperature", "seed", "ollama_version", "vrram_fit",
    "output_tokens", "thinking_tokens", "content_tokens", "tokens_per_sec",
    "ttft_ms", "latency_ms", "created_at",
]

REQUIRED_KEYS = ("run_id", "model_tag", "true_params", "tokens_per_sec")

_VRRAM_FIT_VALID = {"fit", "partial", "offload"}
_VRRAM_FIT_ALIASES = {
    "vram": "fit", "true": "fit", "1": "fit",
    "false": "offload", "0": "offload",
}


def _target_url(target: str) -> str:
    return target if target.startswith("sqlite://") else f"sqlite:///{target}"


def _parse_created_at(value):
    """Parse 'YYYY-MM-DDTHH:MM:SS(.ffffff)?Z' -> naive UTC datetime, or None."""
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    body = value[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(body, fmt)
        except ValueError:
            continue
    return None


def _normalize_think(value):
    """0/1 -> False/True. Anything else -> None (caller rejects the row)."""
    if value == 0:
        return False
    if value == 1:
        return True
    return None


def _normalize_vrram_fit(value):
    """Returns (normalized_value, ok). ok=False means reject the row.

    NULL passes through untouched (the column is nullable); it is not an
    "anything else" case — only unrecognized non-null strings are rejected.
    """
    if value is None:
        return None, True
    v = str(value).strip().lower()
    if v in _VRRAM_FIT_VALID:
        return v, True
    if v in _VRRAM_FIT_ALIASES:
        return _VRRAM_FIT_ALIASES[v], True
    return None, False


def _transform_row(raw, row_num, warnings):
    """Returns (row_dict, None) on success, or (None, error_message) to reject."""
    row = dict(zip(SOURCE_COLUMNS, raw))
    run_id = row.get("run_id")

    for key in REQUIRED_KEYS:
        if row[key] is None:
            return None, f"row {row_num} (run_id={run_id!r}): missing required field {key!r}"

    think = _normalize_think(row["think"])
    if think is None:
        return None, f"row {row_num} (run_id={run_id!r}): invalid think value {row['think']!r}"
    row["think"] = think

    vrram_fit, ok = _normalize_vrram_fit(row["vrram_fit"])
    if not ok:
        return None, f"row {row_num} (run_id={run_id!r}): invalid vrram_fit value {row['vrram_fit']!r}"
    row["vrram_fit"] = vrram_fit

    created_at = _parse_created_at(row["created_at"])
    if created_at is None:
        return None, f"row {row_num} (run_id={run_id!r}): unparseable created_at {row['created_at']!r}"
    row["created_at"] = created_at

    # R3 caveat: output_tokens == thinking_tokens + content_tokens does NOT
    # hold exactly on real data (observed +1..+3 drift) — never reject on it.
    # Warn instead, on the token counts actually looking wrong for `think`.
    thinking_tokens = row["thinking_tokens"]
    content_tokens = row["content_tokens"]
    if think:
        if thinking_tokens is None or content_tokens is None or thinking_tokens <= 0 or content_tokens <= 0:
            warnings.append(
                f"row {row_num} (run_id={run_id!r}): think=True but "
                f"thinking_tokens={thinking_tokens!r} content_tokens={content_tokens!r}"
            )
    else:
        if thinking_tokens not in (0, None):
            warnings.append(
                f"row {row_num} (run_id={run_id!r}): think=False but thinking_tokens={thinking_tokens!r}"
            )

    return row, None


def _read_source_rows(source_path):
    conn = sqlite3.connect(source_path)
    try:
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
        missing = [c for c in SOURCE_COLUMNS if c not in existing_cols]
        if missing:
            raise ValueError(f"source table `samples` is missing expected columns: {missing}")
        cols_sql = ", ".join(SOURCE_COLUMNS)
        return conn.execute(f"SELECT {cols_sql} FROM samples ORDER BY run_id").fetchall()
    finally:
        conn.close()


def run_import(source, target, dry_run=False):
    """Import `source` samples.db into `target`. Returns a counts dict."""
    raw_rows = _read_source_rows(source)
    source_read = len(raw_rows)

    warnings = []
    failed = 0
    good_rows = []
    for i, raw in enumerate(raw_rows, start=1):
        row, error = _transform_row(raw, i, warnings)
        if error:
            failed += 1
            print(f"REJECTED: {error}", file=sys.stderr)
            continue
        good_rows.append(row)

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    target_engine = create_engine(_target_url(target))
    try:
        BenchSample.__table__.create(bind=target_engine, checkfirst=True)

        with target_engine.connect() as conn:
            existing_ids = {
                r[0] for r in conn.execute(select(BenchSample.__table__.c.run_id))
            }

        new_rows = [r for r in good_rows if r["run_id"] not in existing_ids]
        skipped_already_present = len(good_rows) - len(new_rows)

        inserted = 0
        if not dry_run and new_rows:
            with target_engine.begin() as conn:
                conn.execute(BenchSample.__table__.insert(), new_rows)
            inserted = len(new_rows)

        counts = {
            "source_read": source_read,
            "inserted": inserted,
            "skipped_already_present": skipped_already_present,
            "failed": failed,
        }
        print(
            f"source_read={counts['source_read']} inserted={counts['inserted']} "
            f"skipped_already_present={counts['skipped_already_present']} failed={counts['failed']}"
        )
        if dry_run:
            print(f"Dry run — no rows written ({len(new_rows)} would be inserted).")
        return counts
    finally:
        target_engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Path to source samples.db (table `samples`).")
    parser.add_argument("--target", required=True, help="Filesystem path or sqlite:// URL for the isolated target DB.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing to --target.")
    args = parser.parse_args(argv)

    run_import(args.source, args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
