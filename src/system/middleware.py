"""Custom middleware for tracking metrics and other cross-cutting concerns.

Both middlewares are plain ASGI callables rather than BaseHTTPMiddleware:
they run in the same task as the route (so context set by dependencies is
visible to later logs) and they see every body chunk of a streaming response,
so durations cover the whole stream, not just the headers.
"""

import re
import time
import uuid
from typing import Optional

from jose import (
    JWTError,
    jwt,
)
from opentelemetry import trace
from starlette.datastructures import (
    Headers,
    MutableHeaders,
)
from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)

from src.config.settings import settings
from src.system.logs import (
    new_request_context,
    reset_request_context,
)
from src.system.telemetry import (
    http_request_duration_seconds,
    http_requests_total,
)


# Accept a caller-supplied request id only if it is short and plain; otherwise mint one.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _route_template(scope: Scope) -> str:
    """Return the matched route's path template, or a fixed label for unmatched paths.

    FastAPI stores the matched APIRoute in scope["route"] during routing; the scope dict
    is shared with outer middleware, so it is readable once the app has returned.
    """
    route = scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class MetricsMiddleware:
    """Count and time every HTTP request by method, route template and status."""

    def __init__(self, app: ASGIApp):
        """Wrap the next ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Track metrics for each request, until the last body chunk has been sent."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start_time
            endpoint = _route_template(scope)
            method = scope.get("method", "")
            http_requests_total.labels(method=method, endpoint=endpoint, status=status_code).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


class LoggingContextMiddleware:
    """Give every request its own logging context.

    Binds a request_id (from X-Request-ID or a new UUID, echoed in the response),
    the request's trace_id (when tracing is on), and the session_id or user_id from
    the Bearer token. The token is decoded only
    to label logs; authentication itself happens in the route dependencies.
    """

    def __init__(self, app: ASGIApp):
        """Wrap the next ASGI app."""
        self.app = app

    @staticmethod
    def _identity(authorization: Optional[str]) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            return {}
        try:
            payload = jwt.decode(authorization[7:], settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        except JWTError:
            return {}  # invalid token: the auth dependency rejects it, nothing to label here
        subject = payload.get("sub")
        if not subject:
            return {}
        return {"session_id": subject} if payload.get("typ") == "session" else {"user_id": subject}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind request context, call the app, and always restore the previous context."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        incoming = headers.get("x-request-id")
        request_id = incoming if incoming and _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        context = {"request_id": request_id, **self._identity(headers.get("authorization"))}
        # The OTel middleware runs outside this one, so the request span is current here.
        # Binding its trace id keeps it on every log line even if a library later swaps
        # the active span (Langfuse's LangChain handler attaches its own spans).
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            context["trace_id"] = format(span_context.trace_id, "032x")
        token = new_request_context(**context)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_request_context(token)
