"""OpenTelemetry tracing and Langfuse client setup.

Two tracing backends, one trace:
- OpenTelemetry (when OTEL_EXPORTER_OTLP_ENDPOINT is set) records every request,
  SQL query, Redis command and outbound HTTP call, and ships them to the OTel
  collector, which forwards them to Grafana Tempo.
- Langfuse records the LLM side (graph nodes, prompts, completions, token usage).

Langfuse 3.x is itself built on OpenTelemetry: it attaches its span processor to
the global TracerProvider when one exists. So the app's provider is installed
first, and infrastructure scopes are blocked from Langfuse; Tempo then sees the
whole request including the LLM spans, while Langfuse only receives LLM spans.
"""

from typing import (
    List,
    Optional,
)

from fastapi import FastAPI
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config.settings import settings
from src.system.logs import logger

# Instrumentation scopes that belong in Tempo but not in Langfuse.
INFRA_SCOPES: List[str] = [
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.instrumentation.asgi",
    "opentelemetry.instrumentation.sqlalchemy",
    "opentelemetry.instrumentation.redis",
    "opentelemetry.instrumentation.httpx",
]


def setup_tracing(app: FastAPI, engine: Optional[AsyncEngine] = None) -> bool:
    """Install the global TracerProvider and instrument FastAPI, SQLAlchemy, Redis and HTTPX.

    Args:
        app: The FastAPI application.
        engine: The async SQLAlchemy engine to instrument.

    Returns:
        bool: True if tracing was enabled, False if no OTLP endpoint is configured.
    """
    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    if not endpoint:
        logger.info("otel_tracing_disabled", reason="OTEL_EXPORTER_OTLP_ENDPOINT not set")
        return False

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.OTEL_SERVICE_NAME,
                "service.version": settings.VERSION,
                "deployment.environment": settings.ENVIRONMENT.value,
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider, excluded_urls="health,metrics")
    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine, tracer_provider=provider)
    RedisInstrumentor().instrument(tracer_provider=provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)

    logger.info("otel_tracing_enabled", endpoint=endpoint, service_name=settings.OTEL_SERVICE_NAME)
    return True


def init_langfuse() -> Langfuse:
    """Create the process-wide Langfuse client.

    Must run after setup_tracing, so Langfuse reuses the app's TracerProvider.
    CallbackHandler() instances created later pick this client up via get_client().

    Returns:
        Langfuse: The client (a no-op if LANGFUSE_* keys are missing).
    """
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY or None,
        secret_key=settings.LANGFUSE_SECRET_KEY or None,
        host=settings.LANGFUSE_HOST,
        environment=settings.ENVIRONMENT.value,
        blocked_instrumentation_scopes=INFRA_SCOPES,
    )
