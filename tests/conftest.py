"""Shared fixtures. Environment is set before any src module is imported."""

import os
import tempfile

os.environ.update(
    {
        "APP_ENV": "test",
        "JWT_SECRET_KEY": "test-secret-key-that-is-long-enough-0123456789",
        "OPENAI_API_KEY": "sk-test",
        "RATE_LIMIT_STORAGE_URI": "memory://",
        "LOG_DIR": tempfile.mkdtemp(prefix="agent-logs-"),
        "LANGFUSE_TRACING_ENABLED": "false",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "",
    }
)

import fakeredis  # noqa: E402
import pytest  # noqa: E402

from src.system import cache  # noqa: E402


@pytest.fixture(autouse=True)
def fake_redis():
    """Give every test an empty in-memory Redis."""
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    cache.set_redis(client)
    yield client
    cache.set_redis(None)


class FakeDatabase:
    """In-memory stand-in for DatabaseService with the methods the routes use."""

    def __init__(self):
        """Start empty."""
        from src.data.models.session import Session
        from src.data.models.user import User

        self._user_cls, self._session_cls = User, Session
        self.users: dict[int, object] = {}
        self.sessions: dict[str, object] = {}

    async def create_user(self, email, password):
        """Store a user."""
        user = self._user_cls(id=len(self.users) + 1, email=email, hashed_password=password)
        self.users[user.id] = user
        return user

    async def get_user(self, user_id):
        """Look a user up by id."""
        return self.users.get(user_id)

    async def get_user_by_email(self, email):
        """Look a user up by email."""
        return next((u for u in self.users.values() if u.email == email), None)

    async def create_session(self, session_id, user_id, name=""):
        """Store a session."""
        session = self._session_cls(id=session_id, user_id=user_id, name=name)
        self.sessions[session_id] = session
        return session

    async def get_session(self, session_id):
        """Look a session up by id."""
        return self.sessions.get(session_id)

    async def delete_session(self, session_id):
        """Delete a session."""
        return self.sessions.pop(session_id, None) is not None

    async def get_user_sessions(self, user_id):
        """List a user's sessions."""
        return [s for s in self.sessions.values() if s.user_id == user_id]

    async def update_session_name(self, session_id, name):
        """Rename a session."""
        self.sessions[session_id].name = name
        return self.sessions[session_id]


@pytest.fixture
def fake_db(monkeypatch):
    """Swap the auth routes' database for an in-memory one."""
    from src.interface import auth

    db = FakeDatabase()
    monkeypatch.setattr(auth, "db_service", db)
    return db


@pytest.fixture
def client(fake_db):
    """A TestClient for the real app, without running the lifespan (no Postgres needed)."""
    from fastapi.testclient import TestClient

    from src.main import app

    return TestClient(app)
