"""Shared Redis client.

Redis holds state that every API worker must agree on: rate-limit counters
(through slowapi, see src/system/rate_limit.py) and the list of revoked JWTs
(see src/utils/auth.py).
"""

from typing import Optional

from redis.asyncio import Redis

from src.config.settings import settings

_redis: Optional[Redis] = None


def get_redis() -> Redis:
    """Return the process-wide async Redis client, creating it on first use.

    Returns:
        Redis: A client with a connection pool; connections open lazily.
    """
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
    return _redis


def set_redis(client: Optional[Redis]) -> None:
    """Replace the shared client (used by tests to inject a fake).

    Args:
        client: The client to use, or None to recreate from settings on next use.
    """
    global _redis
    _redis = client


async def redis_health_check() -> bool:
    """Ping Redis.

    Returns:
        bool: True if Redis answered, False otherwise.
    """
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False


async def close_redis() -> None:
    """Close the shared client's connection pool."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
