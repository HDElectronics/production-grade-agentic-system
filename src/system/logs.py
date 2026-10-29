"""Logging configuration and setup for the application.

This module provides structured logging configuration using structlog,
with environment-specific formatters and handlers. It supports both
console-friendly development logging and JSON-formatted production logging.

Every event carries the request context (request_id, session_id, user_id) and,
when OpenTelemetry tracing is active, the current trace_id and span_id, so a log
line in Loki links straight to its trace in Tempo.
"""

import logging
import sys
import threading
from contextvars import (
    ContextVar,
    Token,
)
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import structlog
from opentelemetry import trace

from src.config.settings import (
    Environment,
    settings,
)

# Ensure log directory exists
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Context variable for request-specific data. Defaults to None (not a shared mutable dict).
_request_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("request_context", default=None)


def new_request_context(**kwargs: Any) -> Token:
    """Start a fresh logging context for one request.

    Args:
        **kwargs: Initial key-value pairs (for example request_id).

    Returns:
        Token: Pass it to reset_request_context when the request ends.
    """
    return _request_context.set(dict(kwargs))


def reset_request_context(token: Token) -> None:
    """Restore the logging context that was active before new_request_context.

    Args:
        token: The token returned by new_request_context.
    """
    _request_context.reset(token)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to the current request.

    Args:
        **kwargs: Key-value pairs to bind to the logging context
    """
    current = _request_context.get() or {}
    _request_context.set({**current, **kwargs})


def clear_context() -> None:
    """Clear all context variables for the current request."""
    _request_context.set({})


def get_context() -> Dict[str, Any]:
    """Get the current logging context.

    Returns:
        Dict[str, Any]: Current context dictionary
    """
    return _request_context.get() or {}


def add_context_to_event_dict(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add context variables to the event dictionary.

    This processor adds any bound context variables to each log event.

    Args:
        logger: The logger instance
        method_name: The name of the logging method
        event_dict: The event dictionary to modify

    Returns:
        Dict[str, Any]: Modified event dictionary with context variables
    """
    context = get_context()
    if context:
        event_dict.update(context)
    return event_dict


def add_trace_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add the active OpenTelemetry trace and span ids, when there is one.

    Args:
        logger: The logger instance
        method_name: The name of the logging method
        event_dict: The event dictionary to modify

    Returns:
        Dict[str, Any]: Event dictionary with trace_id and span_id when a span is recording
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
    return event_dict


def add_environment(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add the deployment environment to every event.

    Args:
        logger: The logger instance
        method_name: The name of the logging method
        event_dict: The event dictionary to modify

    Returns:
        Dict[str, Any]: Event dictionary with the environment name
    """
    event_dict["environment"] = settings.ENVIRONMENT.value
    return event_dict


def get_log_file_path(day: Optional[datetime] = None) -> Path:
    """Get the log file path for a given day (today by default).

    Args:
        day: The day to build the file name for.

    Returns:
        Path: The path to the log file
    """
    day = day or datetime.now(UTC)
    return settings.LOG_DIR / f"{settings.ENVIRONMENT.value}-{day.strftime('%Y-%m-%d')}.jsonl"


class DailyJsonlFileHandler(logging.Handler):
    """Write one JSON object per line to a file named after the current UTC day.

    The file name is computed on every write, so a long-running process rolls over to a
    new file at midnight instead of writing to the day it started on.
    """

    def __init__(self):
        """Initialize the handler with a lock for concurrent writers in the same process."""
        super().__init__()
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a formatted record to today's JSONL file."""
        try:
            line = self.format(record)
            with self._lock, open(get_log_file_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            self.handleError(record)


def get_structlog_processors(include_file_info: bool = True) -> List[Any]:
    """Get the structlog processors shared by structlog and stdlib loggers.

    Args:
        include_file_info: Whether to include file information in the logs

    Returns:
        List[Any]: List of structlog processors
    """
    processors: List[Any] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        # Request context (request_id, session_id, user_id) and trace ids on every event
        add_context_to_event_dict,
        add_trace_context,
        add_environment,
    ]

    if include_file_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.PATHNAME,
                }
            )
        )

    return processors


def setup_logging() -> None:
    """Configure structlog and the stdlib root logger.

    Both handlers render through structlog's ProcessorFormatter, so library logs
    (uvicorn, httpx, langfuse) get the same structure as application logs:
    - stdout: pretty console output when LOG_FORMAT=console, JSON otherwise
    - logs/<env>-YYYY-MM-DD.jsonl: always JSON (shipped to Loki by the OTel collector)
    """
    log_level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    shared_processors = get_structlog_processors(
        # Include detailed file info only in development and test
        include_file_info=settings.ENVIRONMENT in [Environment.DEVELOPMENT, Environment.TEST]
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    console_formatter = (
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(),
            ],
        )
        if settings.LOG_FORMAT == "console"
        else json_formatter
    )

    file_handler = DailyJsonlFileHandler()
    file_handler.setFormatter(json_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    root = logging.getLogger()
    root.handlers = [file_handler, console_handler]
    root.setLevel(log_level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize logging
setup_logging()

# Create logger instance
logger = structlog.get_logger()
logger.info(
    "logging_initialized",
    environment=settings.ENVIRONMENT.value,
    log_level=settings.LOG_LEVEL,
    log_format=settings.LOG_FORMAT,
    debug=settings.DEBUG,
)
