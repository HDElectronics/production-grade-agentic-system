"""Initial schema: user and session tables.

Databases created by the old SQLModel.metadata.create_all() already have these
tables; they are left as they are, so this revision adopts them.

Revision ID: 0001
Revises:
Create Date: 2026-10-29
"""

from typing import (
    Sequence,
    Union,
)

import sqlalchemy as sa
from alembic import (
    context,
    op,
)

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _missing(table: str) -> bool:
    """True if the table does not exist yet (always True when generating offline SQL)."""
    if context.is_offline_mode():
        return True
    return not sa.inspect(op.get_bind()).has_table(table)


def upgrade() -> None:
    """Create the user and session tables if they are not there yet."""
    if _missing("user"):
        op.create_table(
            "user",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("hashed_password", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_user_email", "user", ["email"], unique=True)

    if _missing("session"):
        op.create_table(
            "session",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
            sa.Column("name", sa.String(), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_session_user_id", "session", ["user_id"])


def downgrade() -> None:
    """Drop the session and user tables."""
    op.drop_table("session")
    op.drop_table("user")
