"""Rate limiting configuration for the application.

This module configures rate limiting using slowapi. Counters live in Redis
(RATE_LIMIT_STORAGE_URI), so every worker and every replica shares the same
budget. If Redis is unreachable, slowapi falls back to per-process memory
until it comes back.

Limits are keyed by client IP. Behind a reverse proxy, set FORWARDED_ALLOW_IPS
to the proxy's address so uvicorn takes the client IP from X-Forwarded-For
instead of seeing every request as coming from the proxy.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config.settings import settings

# Initialize rate limiter. Routes with @limiter.limit use their own limit; every other
# route gets RATE_LIMIT_DEFAULT through SlowAPIASGIMiddleware (added in src/main.py).
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=settings.RATE_LIMIT_DEFAULT,
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
    in_memory_fallback_enabled=True,
    key_prefix="ratelimit",
)
