"""LLM service: per-request fallback, tools on every model, no retries for bad requests."""

import httpx
import openai
import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from tenacity import wait_none

from src.services import llm_provider
from src.services.llm_provider import (
    LLMRegistry,
    LLMService,
)


def _error(cls, status):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


class FakeModel:
    """A chat model double that records tool binding and replays scripted outcomes."""

    def __init__(self, name, outcomes):
        """Script the results of successive calls (exceptions are raised)."""
        self.name, self.outcomes, self.calls, self.bound_tools = name, list(outcomes), 0, None

    def bind_tools(self, tools):
        """Record the tools and return self, like a RunnableBinding would wrap it."""
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        """Return or raise the next scripted outcome."""
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else AIMessage(content=f"from {self.name}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def registry(monkeypatch):
    """Two fake models: the first always rate-limited, the second healthy."""
    first = FakeModel("first", [_error(openai.RateLimitError, 429)] * 10)
    second = FakeModel("second", [])
    monkeypatch.setattr(LLMRegistry, "LLMS", [{"name": "first", "llm": first}, {"name": "second", "llm": second}])
    monkeypatch.setattr(llm_provider.settings, "DEFAULT_LLM_MODEL", "first")
    service = LLMService().bind_tools(["search_tool"])
    service.retry_wait = wait_none()
    return service, first, second


@pytest.mark.asyncio
async def test_fallback_model_keeps_tools_and_does_not_stick(registry):
    service, first, second = registry
    reply = await service.call([HumanMessage("hi")])
    assert reply.content == "from second"
    assert first.calls == llm_provider.settings.MAX_LLM_CALL_RETRIES  # retried, then fell back
    assert second.bound_tools == ["search_tool"]  # the fallback model can still call tools
    assert service.default_model_name == "first"  # the next request starts from the default again


@pytest.mark.asyncio
async def test_bad_request_is_not_retried_or_failed_over(monkeypatch):
    bad = FakeModel("bad", [_error(openai.BadRequestError, 400)])
    other = FakeModel("other", [])
    monkeypatch.setattr(LLMRegistry, "LLMS", [{"name": "bad", "llm": bad}, {"name": "other", "llm": other}])
    monkeypatch.setattr(llm_provider.settings, "DEFAULT_LLM_MODEL", "bad")
    service = LLMService()
    service.retry_wait = wait_none()
    with pytest.raises(openai.BadRequestError):
        await service.call([HumanMessage("hi")])
    assert bad.calls == 1 and other.calls == 0


@pytest.mark.asyncio
async def test_all_models_failing_raises_runtime_error(monkeypatch):
    models = [FakeModel(n, [_error(openai.InternalServerError, 500)] * 10) for n in ("a", "b")]
    monkeypatch.setattr(LLMRegistry, "LLMS", [{"name": m.name, "llm": m} for m in models])
    monkeypatch.setattr(llm_provider.settings, "DEFAULT_LLM_MODEL", "a")
    service = LLMService()
    service.retry_wait = wait_none()
    with pytest.raises(RuntimeError, match="after trying 2 models"):
        await service.call([HumanMessage("hi")])
