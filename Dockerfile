# syntax=docker/dockerfile:1

# ---- build: resolve dependencies from uv.lock into /app/.venv ----
FROM python:3.13.2-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime: no compilers, non-root ----
FROM python:3.13.2-slim

ARG APP_ENV=production
ENV APP_ENV=${APP_ENV} \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-multiproc

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY --from=build --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
COPY --chown=appuser:appuser src ./src

RUN chmod +x /app/scripts/docker-entrypoint.sh && mkdir -p /app/logs && chown appuser:appuser /app/logs

USER appuser
EXPOSE 8000

# python:slim has no curl; probe with the standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
# Workers: WEB_CONCURRENCY. Behind a proxy: FORWARDED_ALLOW_IPS=<proxy address> (both read by uvicorn).
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
