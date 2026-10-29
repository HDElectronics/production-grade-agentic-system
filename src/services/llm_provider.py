"""LLM service for managing LLM calls with retries and fallback mechanisms."""

import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import (
    Environment,
    settings,
)
from src.system.logs import logger
from src.system.telemetry import (
    llm_fallbacks_total,
    llm_inference_duration_seconds,
)

# Worth retrying on the same model: the request was fine, the service was not.
# (APITimeoutError is a subclass of APIConnectionError.)
TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)

# Worth trying the next model: transient errors that outlived their retries, or this
# model being unavailable to this key. Anything else (400 bad request, 401 bad key,
# 422) fails the same way on every model, so it is raised immediately.
FALLBACK_ERRORS = TRANSIENT_ERRORS + (NotFoundError, PermissionDeniedError)


class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances.

    This class maintains a list of LLM configurations and provides
    methods to retrieve them by name with optional argument overrides.
    The list order is the fallback order.
    """

    # Class-level variable containing all available LLM models
    LLMS: List[Dict[str, Any]] = [
        {
            "name": "gpt-5-mini",
            "llm": ChatOpenAI(
                model="gpt-5-mini",
                api_key=settings.OPENAI_API_KEY,
                max_tokens=settings.MAX_TOKENS,
                reasoning={"effort": "low"},
            ),
        },
        {
            "name": "gpt-5",
            "llm": ChatOpenAI(
                model="gpt-5",
                api_key=settings.OPENAI_API_KEY,
                max_tokens=settings.MAX_TOKENS,
                reasoning={"effort": "medium"},
            ),
        },
        {
            "name": "gpt-5-nano",
            "llm": ChatOpenAI(
                model="gpt-5-nano",
                api_key=settings.OPENAI_API_KEY,
                max_tokens=settings.MAX_TOKENS,
                reasoning={"effort": "minimal"},
            ),
        },
        {
            "name": "gpt-4o",
            "llm": ChatOpenAI(
                model="gpt-4o",
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=settings.MAX_TOKENS,
                top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
                presence_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
                frequency_penalty=0.1 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.0,
            ),
        },
        {
            "name": "gpt-4o-mini",
            "llm": ChatOpenAI(
                model="gpt-4o-mini",
                temperature=settings.DEFAULT_LLM_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=settings.MAX_TOKENS,
                top_p=0.9 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
            ),
        },
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name with optional argument overrides.

        Args:
            model_name: Name of the model to retrieve
            **kwargs: Optional arguments to override default model configuration

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If model_name is not found in LLMS
        """
        entry = next((e for e in cls.LLMS if e["name"] == model_name), None)
        if entry is None:
            raise ValueError(
                f"model '{model_name}' not found in registry. available models: {', '.join(cls.get_all_names())}"
            )

        # If user provides kwargs, create a new instance with those args
        if kwargs:
            logger.debug("creating_llm_with_custom_args", model_name=model_name, custom_args=list(kwargs.keys()))
            return ChatOpenAI(model=model_name, api_key=settings.OPENAI_API_KEY, **kwargs)

        return entry["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get all registered LLM names in order.

        Returns:
            List of LLM names
        """
        return [entry["name"] for entry in cls.LLMS]

    @classmethod
    def index_of(cls, model_name: str) -> int:
        """Get the registry index of a model.

        Args:
            model_name: Name of the model

        Returns:
            Index of the model in LLMS

        Raises:
            ValueError: If model_name is not registered
        """
        return cls.get_all_names().index(model_name)


def _log_retry(retry_state: RetryCallState) -> None:
    """Log a retry attempt through structlog (tenacity's helper expects a stdlib logger)."""
    error = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "llm_call_retrying",
        attempt=retry_state.attempt_number,
        error_type=type(error).__name__ if error else None,
        error=str(error) if error else None,
        sleep_seconds=retry_state.next_action.sleep if retry_state.next_action else None,
    )


class LLMService:
    """Service for managing LLM calls with retries and circular fallback.

    Each call starts from the default model and walks the registry in order. Nothing
    about a failed call is remembered: one request's failover never changes which model
    the next request uses. Tools are bound to every model, so a fallback model can
    still call them.
    """

    def __init__(self):
        """Initialize the LLM service."""
        all_names = LLMRegistry.get_all_names()
        try:
            self._default_index = LLMRegistry.index_of(settings.DEFAULT_LLM_MODEL)
        except ValueError:
            self._default_index = 0
            logger.warning(
                "default_model_not_found_using_first",
                requested=settings.DEFAULT_LLM_MODEL,
                using=all_names[0] if all_names else "none",
            )
        self._tools: List[Any] = []
        self._bound: Dict[str, Runnable] = {}
        # Exposed so tests can swap in tenacity.wait_none()
        self.retry_wait = wait_exponential(multiplier=1, min=2, max=10)
        logger.info(
            "llm_service_initialized",
            default_model=all_names[self._default_index],
            fallback_order=all_names[self._default_index :] + all_names[: self._default_index],
            environment=settings.ENVIRONMENT.value,
        )

    @property
    def default_model_name(self) -> str:
        """Name of the model every call starts with."""
        return LLMRegistry.LLMS[self._default_index]["name"]

    def get_llm(self) -> BaseChatModel:
        """Get the default model, without tools (used for token counting and metric labels).

        Returns:
            The default BaseChatModel instance
        """
        return LLMRegistry.LLMS[self._default_index]["llm"]

    def bind_tools(self, tools: Sequence[Any]) -> "LLMService":
        """Bind tools to every registered model.

        Binding happens lazily per model and is cached, so the fallback models carry the
        same tools as the default one.

        Args:
            tools: List of tools to bind

        Returns:
            Self for method chaining
        """
        self._tools = list(tools)
        self._bound.clear()
        logger.debug("tools_registered", tool_count=len(self._tools))
        return self

    def _model(self, index: int) -> Runnable:
        entry = LLMRegistry.LLMS[index]
        name = entry["name"]
        if name not in self._bound:
            self._bound[name] = entry["llm"].bind_tools(self._tools) if self._tools else entry["llm"]
        return self._bound[name]

    async def _call_with_retry(self, index: int, messages: List[BaseMessage]) -> BaseMessage:
        """Call one model, retrying transient errors with exponential backoff.

        Args:
            index: Registry index of the model
            messages: List of messages to send to the LLM

        Returns:
            BaseMessage response from the LLM
        """
        name = LLMRegistry.LLMS[index]["name"]
        model = self._model(index)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
            wait=self.retry_wait,
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            before_sleep=_log_retry,
            reraise=True,
        ):
            with attempt:
                started = time.perf_counter()
                response = await model.ainvoke(messages)
                llm_inference_duration_seconds.labels(model=name).observe(time.perf_counter() - started)
                logger.debug("llm_call_successful", model=name, message_count=len(messages))
                return response
        raise RuntimeError("unreachable: tenacity re-raises the last error")

    async def call(self, messages: List[BaseMessage], model_name: Optional[str] = None) -> BaseMessage:
        """Call the LLM, falling back through the registry on availability errors.

        Args:
            messages: List of messages to send to the LLM
            model_name: Optional model to start with instead of the default.

        Returns:
            BaseMessage response from the LLM

        Raises:
            ValueError: If model_name is not registered.
            openai.OpenAIError: For errors no other model can fix (bad request, bad key).
            RuntimeError: If every model failed.
        """
        start = LLMRegistry.index_of(model_name) if model_name else self._default_index
        total = len(LLMRegistry.LLMS)
        last_error: Optional[Exception] = None

        for offset in range(total):
            index = (start + offset) % total
            name = LLMRegistry.LLMS[index]["name"]
            try:
                return await self._call_with_retry(index, messages)
            except FALLBACK_ERRORS as e:
                last_error = e
                llm_fallbacks_total.labels(model=name, error=type(e).__name__).inc()
                logger.error(
                    "llm_call_failed_trying_next_model",
                    model=name,
                    models_tried=offset + 1,
                    total_models=total,
                    error_type=type(e).__name__,
                    error=str(e),
                )

        logger.error("all_models_failed", models_tried=total, starting_model=LLMRegistry.LLMS[start]["name"])
        raise RuntimeError(f"failed to get response from llm after trying {total} models") from last_error


# Create global LLM service instance
llm_service = LLMService()
