"""Prometheus metrics configuration for the application.

This module defines the application's metrics and serves them on /metrics.

With several uvicorn workers (WEB_CONCURRENCY > 1) each worker is its own process with
its own counters. Set PROMETHEUS_MULTIPROC_DIR (the Docker image does) and every worker
writes its samples there, and /metrics aggregates all of them; otherwise a scrape would
only see whichever worker answered it.
"""

import os

from fastapi import (
    FastAPI,
    Request,
    Response,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

from src.system.rate_limit import limiter

# Request metrics. `endpoint` is the route template (/api/v1/auth/session/{session_id}),
# never the raw path, so ids don't create a new time series each.
http_requests_total = Counter("http_requests_total", "Total number of HTTP requests", ["method", "endpoint", "status"])

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds, until the last body byte is sent",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# LLM metrics. Reasoning models routinely take longer than a few seconds, so buckets go to minutes.
llm_inference_duration_seconds = Histogram(
    "llm_inference_duration_seconds",
    "Time spent in one successful LLM call, labelled with the model that answered",
    ["model"],
    buckets=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)

llm_stream_duration_seconds = Histogram(
    "llm_stream_duration_seconds",
    "Time spent streaming one full chat response",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0],
)

llm_fallbacks_total = Counter(
    "llm_fallbacks_total",
    "LLM calls that gave up on a model and moved to the next one",
    ["model", "error"],
)


@limiter.exempt
async def metrics(request: Request) -> Response:
    """Serve all metrics in the Prometheus text format, aggregated across workers."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_metrics(app: FastAPI) -> None:
    """Register the /metrics endpoint.

    Request counting is done by MetricsMiddleware (src/system/middleware.py).

    Args:
        app: FastAPI application instance
    """
    app.add_api_route("/metrics", metrics, methods=["GET"], include_in_schema=False)
