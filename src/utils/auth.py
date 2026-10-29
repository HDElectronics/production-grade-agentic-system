"""This file contains the authentication utilities for the application.

Two kinds of token share one format and are told apart by the `typ` claim:
- "user":    sub = user id, issued by register and login, used to manage sessions
- "session": sub = chat session id, issued by POST /auth/session, used by the chatbot routes

Logged-out tokens are kept in Redis under their `jti` until they would have expired.
"""

import re
import uuid
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
)

from jose import (
    JWTError,
    jwt,
)

from src.config.settings import settings
from src.data.schemas.auth import Token
from src.system.cache import get_redis
from src.system.logs import logger

TokenType = Literal["user", "session"]

# JWT tokens consist of 3 base64url-encoded segments separated by dots
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_REVOKED_PREFIX = "revoked_jti:"


def create_access_token(subject: str, token_type: TokenType, expires_delta: Optional[timedelta] = None) -> Token:
    """Create a new access token.

    Args:
        subject: The user id (token_type "user") or chat session id (token_type "session").
        token_type: Which kind of token to mint.
        expires_delta: Optional expiration time delta.

    Returns:
        Token: The generated access token.
    """
    now = datetime.now(UTC)
    expire = now + (expires_delta or timedelta(days=settings.JWT_ACCESS_TOKEN_EXPIRE_DAYS))

    to_encode = {
        "sub": subject,
        "typ": token_type,
        "exp": expire,
        "iat": now,
        "jti": uuid.uuid4().hex,  # unique token id, used for revocation
    }

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    logger.info("token_created", subject=subject, token_type=token_type, expires_at=expire.isoformat())

    return Token(access_token=encoded_jwt, expires_at=expire)


def verify_token(token: str, expected_type: Optional[TokenType] = None) -> Optional[Dict[str, Any]]:
    """Verify a JWT and return its claims.

    Args:
        token: The JWT token to verify.
        expected_type: Reject tokens of any other `typ`. None accepts both kinds.

    Returns:
        Optional[Dict[str, Any]]: The claims (sub, typ, exp, iat, jti) if the token is valid
        and of the expected type, None otherwise.

    Raises:
        ValueError: If the token is not shaped like a JWT.
    """
    if not token or not isinstance(token, str):
        logger.warning("token_invalid_format")
        raise ValueError("Token must be a non-empty string")

    if not _JWT_RE.match(token):
        logger.warning("token_suspicious_format")
        raise ValueError("Token format is invalid - expected JWT format")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as e:
        logger.warning("token_verification_failed", error=str(e))
        return None

    if not payload.get("sub") or not payload.get("jti"):
        logger.warning("token_missing_claims")
        return None

    if expected_type is not None and payload.get("typ") != expected_type:
        logger.warning("token_wrong_type", expected=expected_type, got=payload.get("typ"))
        return None

    return payload


async def revoke_token(claims: Dict[str, Any]) -> None:
    """Add a token to the revoked list until it would have expired anyway.

    Args:
        claims: The verified claims of the token to revoke.
    """
    ttl = int(claims["exp"] - datetime.now(UTC).timestamp())
    if ttl > 0:
        await get_redis().set(f"{_REVOKED_PREFIX}{claims['jti']}", "1", ex=ttl)
    logger.info("token_revoked", jti=claims["jti"], token_type=claims.get("typ"))


async def is_token_revoked(claims: Dict[str, Any]) -> bool:
    """Check the revoked list.

    Fails open: if Redis is unreachable the token is treated as not revoked and an
    error is logged, so a Redis outage degrades logout instead of blocking all traffic.

    Args:
        claims: The verified claims of the token.

    Returns:
        bool: True if the token was revoked.
    """
    try:
        return bool(await get_redis().exists(f"{_REVOKED_PREFIX}{claims['jti']}"))
    except Exception as e:
        logger.error("token_revocation_check_failed", error=str(e))
        return False
