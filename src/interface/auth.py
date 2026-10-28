"""Authentication and authorization endpoints for the API.

This module provides endpoints for user registration, login, session management,
logout and token verification.

Token kinds (see src/utils/auth.py): register and login return a *user* token;
POST /session returns a *session* token. Session management routes require a user
token; the chatbot routes and the per-session routes require a session token.
"""

import uuid
from typing import (
    Any,
    Dict,
    List,
)

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from src.agent.workflow import langgraph_agent
from src.config.settings import settings
from src.data.db_manager import db_manager
from src.data.models.session import Session
from src.data.models.user import User
from src.data.schemas.auth import (
    SessionResponse,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from src.system.logs import (
    bind_context,
    logger,
)
from src.system.rate_limit import limiter
from src.utils.auth import (
    TokenType,
    create_access_token,
    is_token_revoked,
    revoke_token,
    verify_token,
)
from src.utils.sanitization import (
    sanitize_email,
    sanitize_string,
    validate_password_strength,
)

router = APIRouter()
security = HTTPBearer()
db_service = db_manager

_UNAUTHORIZED = {"status_code": 401, "headers": {"WWW-Authenticate": "Bearer"}}


async def _claims_from(credentials: HTTPAuthorizationCredentials, expected_type: TokenType | None) -> Dict[str, Any]:
    """Verify a bearer token's signature, type and revocation status.

    Args:
        credentials: The bearer credentials from the Authorization header.
        expected_type: The token kind the route requires, or None for either.

    Returns:
        Dict[str, Any]: The token's claims.

    Raises:
        HTTPException: 401 if invalid, of the wrong kind or revoked; 422 if not a JWT.
    """
    try:
        claims = verify_token(credentials.credentials, expected_type)
    except ValueError as ve:
        logger.error("token_validation_failed", error=str(ve))
        raise HTTPException(status_code=422, detail="Invalid token format", headers={"WWW-Authenticate": "Bearer"})

    if claims is None:
        raise HTTPException(detail="Invalid authentication credentials", **_UNAUTHORIZED)
    if await is_token_revoked(claims):
        raise HTTPException(detail="Token has been revoked", **_UNAUTHORIZED)
    return claims


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get the current user from a user token.

    Args:
        credentials: The HTTP authorization credentials containing the JWT token.

    Returns:
        User: The user extracted from the token.

    Raises:
        HTTPException: If the token is invalid, not a user token, revoked, or the user is gone.
    """
    claims = await _claims_from(credentials, "user")
    try:
        user_id = int(claims["sub"])
    except ValueError:
        raise HTTPException(detail="Invalid authentication credentials", **_UNAUTHORIZED)

    user = await db_service.get_user(user_id)
    if user is None:
        logger.error("user_not_found", user_id=user_id)
        raise HTTPException(status_code=404, detail="User not found", headers={"WWW-Authenticate": "Bearer"})

    # Bind user_id to logging context for all subsequent logs in this request
    bind_context(user_id=user_id)
    return user


async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Session:
    """Get the current chat session from a session token.

    Args:
        credentials: The HTTP authorization credentials containing the JWT token.

    Returns:
        Session: The session extracted from the token.

    Raises:
        HTTPException: If the token is invalid, not a session token, revoked, or the session is gone.
    """
    claims = await _claims_from(credentials, "session")
    session_id = claims["sub"]

    session = await db_service.get_session(session_id)
    if session is None:
        logger.error("session_not_found", session_id=session_id)
        raise HTTPException(status_code=404, detail="Session not found", headers={"WWW-Authenticate": "Bearer"})

    # Bind ids to logging context for all subsequent logs in this request
    bind_context(user_id=session.user_id, session_id=session.id)
    return session


async def get_token_claims(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Get the verified claims of a user or session token.

    Args:
        credentials: The HTTP authorization credentials containing the JWT token.

    Returns:
        Dict[str, Any]: The token's claims.
    """
    return await _claims_from(credentials, None)


@router.post("/register", response_model=UserResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["register"][0])
async def register_user(request: Request, user_data: UserCreate):
    """Register a new user.

    Args:
        request: The FastAPI request object for rate limiting.
        user_data: User registration data

    Returns:
        UserResponse: The created user info, with a user token
    """
    try:
        # Sanitize email (validates format and lowercases it)
        sanitized_email = sanitize_email(user_data.email)

        # Passwords are never escaped or altered: they are only hashed
        password = user_data.password.get_secret_value()
        validate_password_strength(password)

        # Check if user exists
        if await db_service.get_user_by_email(sanitized_email):
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        user = await db_service.create_user(email=sanitized_email, password=User.hash_password(password))

        # Create access token
        token = create_access_token(str(user.id), "user")

        return UserResponse(id=user.id, email=user.email, token=token)
    except ValueError as ve:
        logger.error("user_registration_validation_failed", error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["login"][0])
async def login(
    request: Request, username: str = Form(...), password: str = Form(...), grant_type: str = Form(default="password")
):
    """Login a user.

    Args:
        request: The FastAPI request object for rate limiting.
        username: User's email
        password: User's password
        grant_type: Must be "password"

    Returns:
        TokenResponse: A user token

    Raises:
        HTTPException: If credentials are invalid
    """
    if grant_type != "password":
        raise HTTPException(status_code=400, detail="Unsupported grant type. Must be 'password'")

    try:
        # Same normalization as registration (lowercase), so the stored email matches
        email = sanitize_email(username)
    except ValueError:
        email = None

    # The password is compared exactly as typed: it was hashed unescaped at registration
    user = await db_service.get_user_by_email(email) if email else None
    if not user or not user.verify_password(password):
        raise HTTPException(detail="Incorrect email or password", **_UNAUTHORIZED)

    token = create_access_token(str(user.id), "user")
    return TokenResponse(access_token=token.access_token, token_type="bearer", expires_at=token.expires_at)


@router.post("/logout")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["logout"][0])
async def logout(request: Request, claims: Dict[str, Any] = Depends(get_token_claims)):
    """Revoke the token used for this request (user or session token).

    Args:
        request: The FastAPI request object for rate limiting.
        claims: The verified claims of the presented token.

    Returns:
        dict: Confirmation message.
    """
    try:
        await revoke_token(claims)
    except Exception as e:
        logger.error("logout_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Logout is temporarily unavailable")
    return {"message": "Logged out"}


@router.post("/session", response_model=SessionResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["auth_session"][0])
async def create_session(request: Request, user: User = Depends(get_current_user)):
    """Create a new chat session for the authenticated user.

    Args:
        request: The FastAPI request object for rate limiting.
        user: The authenticated user

    Returns:
        SessionResponse: The session ID, name, and a session token
    """
    session_id = str(uuid.uuid4())
    session = await db_service.create_session(session_id, user.id)
    token = create_access_token(session_id, "session")

    logger.info(
        "session_created",
        session_id=session_id,
        user_id=user.id,
        name=session.name,
        expires_at=token.expires_at.isoformat(),
    )
    return SessionResponse(session_id=session_id, name=session.name, token=token)


@router.patch("/session/{session_id}/name", response_model=SessionResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["auth_session"][0])
async def update_session_name(
    request: Request, session_id: str, name: str = Form(...), current_session: Session = Depends(get_current_session)
):
    """Update a session's name.

    Args:
        request: The FastAPI request object for rate limiting.
        session_id: The ID of the session to update
        name: The new name for the session
        current_session: The current session from auth

    Returns:
        SessionResponse: The updated session information
    """
    # Verify the session ID matches the authenticated session
    if session_id != current_session.id:
        raise HTTPException(status_code=403, detail="Cannot modify other sessions")

    session = await db_service.update_session_name(session_id, sanitize_string(name))

    # Create a new token (not strictly necessary but maintains consistency)
    token = create_access_token(session_id, "session")
    return SessionResponse(session_id=session_id, name=session.name, token=token)


@router.delete("/session/{session_id}")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["auth_session"][0])
async def delete_session(request: Request, session_id: str, current_session: Session = Depends(get_current_session)):
    """Delete a session and its conversation history.

    Long-term memories are per user, not per session, so they are kept.

    Args:
        request: The FastAPI request object for rate limiting.
        session_id: The ID of the session to delete
        current_session: The current session from auth

    Returns:
        dict: Confirmation message.
    """
    # Verify the session ID matches the authenticated session
    if session_id != current_session.id:
        raise HTTPException(status_code=403, detail="Cannot delete other sessions")

    await langgraph_agent.clear_chat_history(session_id)
    await db_service.delete_session(session_id)

    logger.info("session_deleted", session_id=session_id, user_id=current_session.user_id)
    return {"message": "Session deleted"}


@router.get("/sessions", response_model=List[SessionResponse])
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["auth_session"][0])
async def get_user_sessions(request: Request, user: User = Depends(get_current_user)):
    """Get all sessions for the authenticated user.

    Args:
        request: The FastAPI request object for rate limiting.
        user: The authenticated user

    Returns:
        List[SessionResponse]: The user's sessions, each with a fresh session token
    """
    sessions = await db_service.get_user_sessions(user.id)
    return [
        SessionResponse(
            session_id=session.id,
            name=session.name,
            token=create_access_token(session.id, "session"),
        )
        for session in sessions
    ]
