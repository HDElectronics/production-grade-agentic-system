"""Observability: middleware order, metric labels, stream timing, log files, trace scopes."""

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.system.middleware import (
    LoggingContextMiddleware,
    MetricsMiddleware,
)


def _sample(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_middleware_runs_in_the_documented_order():
    from src.main import app

    order = [m.cls.__name__ for m in app.user_middleware]  # outermost first
    assert order == ["CORSMiddleware", "LoggingContextMiddleware", "MetricsMiddleware", "SlowAPIASGIMiddleware"]


def test_metrics_use_route_templates_and_time_whole_streams():
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/items/{item_id}")
    async def item(item_id: str):
        return {"id": item_id}

    @app.get("/stream")
    async def stream():
        async def body():
            for _ in range(3):
                await asyncio.sleep(0.05)
                yield b"x"

        return StreamingResponse(body())

    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/items/{item_id}", "status": "200"}
    before = _sample("http_requests_total", labels)
    client.get("/items/1")
    client.get("/items/2")
    assert _sample("http_requests_total", labels) == before + 2
    assert _sample("http_requests_total", {"method": "GET", "endpoint": "/items/1", "status": "200"}) == 0

    stream_labels = {"method": "GET", "endpoint": "/stream"}
    sum_before = _sample("http_request_duration_seconds_sum", stream_labels)
    client.get("/stream")
    assert _sample("http_request_duration_seconds_sum", stream_labels) - sum_before >= 0.15


def test_request_id_is_echoed_and_bound_to_logs():
    from src.system.logs import get_context

    seen = {}
    app = FastAPI()
    app.add_middleware(LoggingContextMiddleware)

    @app.get("/")
    async def root():
        seen.update(get_context())
        return {}

    response = TestClient(app).get("/", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert seen["request_id"] == "abc-123"
    assert get_context() == {}  # reset after the request

    response = TestClient(app).get("/", headers={"X-Request-ID": "not ok\nheader-injection"})
    assert response.headers["X-Request-ID"] != "not ok\nheader-injection"


def test_log_file_lines_are_flat_json_with_context():
    from src.system.logs import (
        bind_context,
        get_log_file_path,
        logger,
        new_request_context,
        reset_request_context,
    )

    token = new_request_context(request_id="req-1")
    bind_context(user_id=5)
    logger.info("test_event", answer=42)
    logging.getLogger("third.party").warning("stdlib message")
    reset_request_context(token)

    lines = [json.loads(line) for line in get_log_file_path().read_text().splitlines()]
    event = next(line for line in lines if line.get("event") == "test_event")
    assert event["answer"] == 42 and event["request_id"] == "req-1" and event["user_id"] == 5
    assert any(line.get("event") == "stdlib message" for line in lines)


def test_infra_span_scopes_are_blocked_from_langfuse():
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from sqlalchemy import (
        create_engine,
        text,
    )

    from src.system.tracing import INFRA_SCOPES

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {}

    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    engine = create_engine("sqlite://")
    SQLAlchemyInstrumentor().instrument(engine=engine, tracer_provider=provider)
    try:
        TestClient(app).get("/ping")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        SQLAlchemyInstrumentor().uninstrument()
        FastAPIInstrumentor.uninstrument_app(app)

    scopes = {span.instrumentation_scope.name for span in exporter.get_finished_spans()}
    assert scopes, "no spans recorded"
    assert scopes <= set(INFRA_SCOPES), scopes - set(INFRA_SCOPES)


@pytest.mark.asyncio
async def test_metrics_endpoint_is_exempt_from_rate_limits(client):
    for _ in range(30):
        assert client.get("/metrics").status_code == 200


@pytest.mark.asyncio
async def test_request_trace_id_is_bound_for_the_whole_request():
    from opentelemetry.sdk.trace import TracerProvider

    from src.system.logs import get_context

    tracer = TracerProvider().get_tracer("test")
    seen = {}

    async def inner(scope, receive, send):
        seen.update(get_context())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b""}

    sent = []

    async def send(message):
        sent.append(message)

    with tracer.start_as_current_span("request") as span:
        await LoggingContextMiddleware(inner)({"type": "http", "headers": []}, receive, send)

    assert seen["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert dict(sent[0]["headers"])[b"x-request-id"]


def test_metrics_aggregate_across_worker_processes(tmp_path):
    """Two processes write samples; one /metrics response must contain both."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}
    worker = (
        "from prometheus_client import Counter; "
        "Counter('worker_hits_total', 'hits', ['w']).labels(w='{}').inc()"
    )
    for name in ("a", "b"):
        subprocess.run([sys.executable, "-c", worker.format(name)], env=env, check=True)

    reader = (
        "import asyncio, os; from src.system.telemetry import metrics; "
        "print(asyncio.run(metrics(None)).body.decode())"
    )
    out = subprocess.run([sys.executable, "-c", reader], env=env, check=True, capture_output=True, text=True).stdout
    assert 'worker_hits_total{w="a"} 1.0' in out and 'worker_hits_total{w="b"} 1.0' in out
