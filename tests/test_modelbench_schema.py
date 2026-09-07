from pathlib import Path

import subprocess
import sys

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

from sqlalchemy import inspect

import core.database
from core.database import Base, BenchSample

EXPECTED_COLUMNS = {
    "run_id",
    "model_tag",
    "true_params",
    "quant",
    "ctx_len",
    "think",
    "prompt_bytes",
    "temperature",
    "seed",
    "ollama_version",
    "vrram_fit",
    "output_tokens",
    "thinking_tokens",
    "content_tokens",
    "tokens_per_sec",
    "ttft_ms",
    "latency_ms",
    "created_at",
    "prompt_text",
    "collector",
}


def test_bench_samples_table_created_with_expected_columns():
    SessionLocal, engine, tmpfile = make_temp_sqlite(Base.metadata)
    try:
        inspector = inspect(engine)
        assert "bench_samples" in inspector.get_table_names()
        columns = {col["name"] for col in inspector.get_columns("bench_samples")}
        assert columns == EXPECTED_COLUMNS
    finally:
        engine.dispose()
        Path(tmpfile.name).unlink(missing_ok=True)


def test_create_all_is_idempotent():
    SessionLocal, engine, tmpfile = make_temp_sqlite(Base.metadata)
    try:
        Base.metadata.create_all(bind=engine)
    finally:
        engine.dispose()
        Path(tmpfile.name).unlink(missing_ok=True)


def test_bench_sample_registered_on_base_metadata():
    assert "bench_samples" in core.database.Base.metadata.tables


def test_update_database_script_untouched():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "scripts/update_database.py"],
        cwd=repo_root,
    )
    assert result.returncode == 0


def test_init_db_creates_bench_samples_on_fresh_db_and_is_re_runnable():
    """Acceptance (1)+(3): a fresh SQLite DB initialized via init_db() has the
    modelbench table, and init_db() is re-runnable/idempotent (run twice)."""
    repo_root = Path(__file__).resolve().parent.parent
    probe = (
        "import os, sys, tempfile, sqlite3;"
        "tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False); tmp.close();"
        "os.environ['DATABASE_URL'] = 'sqlite:///' + tmp.name;"
        "sys.path.insert(0, %r);"
        "import core.database as db;"
        "db.init_db(); db.init_db();"  # run twice -> idempotent
        "con = sqlite3.connect(tmp.name);"
        "cur = con.cursor();"
        "cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='bench_samples'\");"
        "assert cur.fetchone(), 'bench_samples missing after init_db()';"
        "cols = {r[1] for r in cur.execute('PRAGMA table_info(bench_samples)')};"
        "assert %r <= cols, 'missing columns: ' + repr(%r - cols);"
        "con.close(); os.unlink(tmp.name);"
        "print('OK')" % (str(repo_root), EXPECTED_COLUMNS, EXPECTED_COLUMNS)
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"init_db idempotency probe failed:\n{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
