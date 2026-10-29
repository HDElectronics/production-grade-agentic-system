"""Alembic environment: runs migrations against the database from POSTGRES_* settings."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from src.config.settings import settings
from src.data.models.session import Session  # noqa: F401  (registers the table)
from src.data.models.user import User  # noqa: F401  (registers the table)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = SQLModel.metadata

# Same database as the app, through the synchronous psycopg 3 driver
SYNC_URL = settings.POSTGRES_ASYNC_URL


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting (alembic upgrade head --sql)."""
    context.configure(url=SYNC_URL, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations."""
    engine = create_engine(SYNC_URL)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
