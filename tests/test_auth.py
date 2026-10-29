"""Auth: token kinds, login with special characters, logout, rate-limited session routes."""

import pytest

from src.utils.auth import (
    create_access_token,
    is_token_revoked,
    revoke_token,
    verify_token,
)

PASSWORD = 'Str0ng&"<Pass>'  # every character html.escape would change


def register_and_login(client, email="Alice@Example.com", password=PASSWORD):
    """Register, then log in with the same (mixed-case) email; return the user token."""
    r = client.post("/api/auth/register", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    r = client.post("/api/auth/login", data={"username": email, "password": password, "grant_type": "password"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_login_accepts_password_with_html_special_characters(client):
    assert register_and_login(client)


def test_login_rejects_wrong_password(client):
    register_and_login(client)
    r = client.post("/api/auth/login", data={"username": "alice@example.com", "password": "Wrong&Pass1"})
    assert r.status_code == 401


def test_token_kinds_are_not_interchangeable():
    user_token = create_access_token("1", "user").access_token
    session_token = create_access_token("abc", "session").access_token
    assert verify_token(user_token, "user")["sub"] == "1"
    assert verify_token(user_token, "session") is None
    assert verify_token(session_token, "user") is None
    assert verify_token(session_token, "session")["typ"] == "session"


def test_user_token_cannot_open_chat_routes(client):
    user_token = register_and_login(client)
    r = client.get("/api/chatbot/messages", headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code == 401


def test_session_flow_and_logout(client):
    user_token = register_and_login(client)
    auth = {"Authorization": f"Bearer {user_token}"}
    r = client.post("/api/auth/session", headers=auth)
    assert r.status_code == 200, r.text
    assert client.get("/api/auth/sessions", headers=auth).status_code == 200

    assert client.post("/api/auth/logout", headers=auth).status_code == 200
    r = client.get("/api/auth/sessions", headers=auth)
    assert r.status_code == 401
    assert r.json()["detail"] == "Token has been revoked"


@pytest.mark.asyncio
async def test_revocation_is_stored_until_expiry(fake_redis):
    claims = verify_token(create_access_token("7", "user").access_token)
    assert not await is_token_revoked(claims)
    await revoke_token(claims)
    assert await is_token_revoked(claims)
    assert 0 < await fake_redis.ttl(f"revoked_jti:{claims['jti']}") <= 30 * 24 * 3600


@pytest.mark.asyncio
async def test_revocation_check_fails_open_when_redis_is_down(monkeypatch):
    from src.utils import auth as auth_utils

    class Down:
        async def exists(self, *_):
            raise ConnectionError("redis down")

    monkeypatch.setattr(auth_utils, "get_redis", lambda: Down())
    claims = verify_token(create_access_token("7", "user").access_token)
    assert await is_token_revoked(claims) is False


def test_session_routes_are_rate_limited():
    from src.system.rate_limit import limiter

    limited = {name.rsplit(".", 1)[-1] for name in limiter._route_limits}
    assert {"create_session", "update_session_name", "delete_session", "get_user_sessions", "logout"} <= limited


def test_delete_session_clears_its_checkpoints(client, monkeypatch):
    from src.interface import auth

    cleared = []

    async def fake_clear(session_id):
        cleared.append(session_id)

    monkeypatch.setattr(auth.langgraph_agent, "clear_chat_history", fake_clear)
    user_token = register_and_login(client)
    session = client.post("/api/auth/session", headers={"Authorization": f"Bearer {user_token}"}).json()
    session_auth = {"Authorization": f"Bearer {session['token']['access_token']}"}
    r = client.delete(f"/api/auth/session/{session['session_id']}", headers=session_auth)
    assert r.status_code == 200, r.text
    assert cleared == [session["session_id"]]
