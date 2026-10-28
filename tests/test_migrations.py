"""Migrations render to SQL and match the models."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upgrade_head_renders_sql_offline():
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    sql = result.stdout
    assert 'CREATE TABLE "user"' in sql
    assert "CREATE TABLE session" in sql
    assert "CREATE UNIQUE INDEX ix_user_email" in sql
    assert "DROP TABLE IF EXISTS thread" in sql


def test_models_define_the_migrated_tables():
    from sqlmodel import SQLModel

    from src.data.models.session import Session  # noqa: F401
    from src.data.models.user import User  # noqa: F401

    assert {"user", "session"} <= set(SQLModel.metadata.tables)
    assert "thread" not in SQLModel.metadata.tables
