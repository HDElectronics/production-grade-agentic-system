"""Drop the unused thread table.

The Thread model was defined but nothing ever wrote to it; conversation history
lives in LangGraph's checkpoint tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-29
"""

from typing import (
    Sequence,
    Union,
)

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop the table if an older create_all() made it."""
    op.execute("DROP TABLE IF EXISTS thread")


def downgrade() -> None:
    """Recreate the (empty) table."""
    op.create_table(
        "thread",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
