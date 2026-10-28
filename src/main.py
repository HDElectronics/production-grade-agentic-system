"""This file contains the main application entry point."""

from contextlib import asynccontextmanager
from datetime import (
    UTC,
    datetime,
)
from typing import (
    Any,
    Dict,
)

from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware

from src.agent.workflow import langgraph_agent
from src.config.settings import settings
from src.data.db_manager import db_manager
from src.interface.router import api_router
from src.system.cache import (
    close_redis,
    redis_health_check,
)
from src.system.logs import logger
from src.system.middleware import (
    LoggingContextMiddleware,
    MetricsMiddleware,
)
from src.system.rate_limit import limiter
from src.system.telemetry import setup_metrics
from src.system.tracing import (
    init_langfuse,
    setup_tracing,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    settings.validate_runtime()
    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        api_prefix=settings.API_V1_STR,
    )
    yield
    logger.info("application_shutdown")
    await langgraph_agent.shutdown()
    await db_manager.dispose()
    await close_redis()
    langfuse.flush()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Tracing first: Langfuse attaches to the TracerProvider that exists when it starts
setup_tracing(app, db_manager.engine)
langfuse = init_langfuse()

# Set up Prometheus metrics endpoint
setup_metrics(app)

# Set up rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Add validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data.

    Args:
        request: The request that caused the validation error
        exc: The validation error

    Returns:
        JSONResponse: A formatted error response
    """
    # Log the validation error
    logger.error(
        "validation_error",
        client_host=request.client.host if request.client else "unknown",
        path=request.url.path,
        errors=str(exc.errors()),
    )

    # Format the errors to be more user-friendly
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join([str(loc_part) for loc_part in error["loc"] if loc_part != "body"])
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


# Middleware. Starlette runs the LAST added middleware FIRST, so the request passes
# through them in the reverse order of these calls:
#   CORS -> LoggingContext -> Metrics -> SlowAPI (default rate limits) -> routes
app.add_middleware(SlowAPIASGIMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    # Auth uses bearer tokens, not cookies; never combine credentials with a wildcard origin
    allow_credentials="*" not in settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["root"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health/live")
@limiter.exempt
async def liveness(request: Request) -> Dict[str, str]:
    """Liveness probe: the process is up and serving requests."""
    return {"status": "alive"}


@app.get("/health")
@app.get("/health/ready")
@limiter.exempt
async def health_check(request: Request) -> JSONResponse:
    """Readiness probe with dependency status.

    Postgres is required (503 when down). Redis is optional: without it rate limits
    fall back to per-process memory and logout is unavailable, so it reports degraded.

    Returns:
        JSONResponse: Health status information
    """
    db_healthy = await db_manager.health_check()
    redis_healthy = await redis_health_check()

    response: Dict[str, Any] = {
        "status": "healthy" if db_healthy and redis_healthy else ("degraded" if db_healthy else "unhealthy"),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {
            "api": "healthy",
            "database": "healthy" if db_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy",
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response, status_code=status_code)
