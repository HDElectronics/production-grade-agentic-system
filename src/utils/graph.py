"""This file contains the graph utilities for the application."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.messages import trim_messages as _trim_messages

from src.config.settings import settings
from src.data.schemas import ChatMessage
from src.system.logs import logger


def dump_messages(messages: list[ChatMessage]) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages (list[ChatMessage]): The messages to dump.

    Returns:
        list[dict]: The dumped messages.
    """
    return [message.model_dump() for message in messages]


def extract_text(content: Any) -> str:
    """Return the visible text of a message's content.

    GPT-5 models (Responses API) return content as a list of blocks like
    [{'type': 'reasoning', ...}, {'type': 'text', 'text': 'actual response'}];
    streamed chunks carry the same blocks in pieces. Reasoning blocks are dropped.

    Args:
        content: A message's `content` (str, list of blocks, or None).

    Returns:
        str: The concatenated text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def process_llm_response(response: BaseMessage) -> BaseMessage:
    """Process LLM response to handle structured content blocks (e.g., from GPT-5 models).

    Args:
        response: The raw response from the LLM

    Returns:
        BaseMessage with processed content
    """
    if isinstance(response.content, list):
        block_count = len(response.content)
        reasoning_blocks = sum(1 for b in response.content if isinstance(b, dict) and b.get("type") == "reasoning")
        response.content = extract_text(response.content)
        logger.debug(
            "processed_structured_content",
            block_count=block_count,
            reasoning_blocks=reasoning_blocks,
            extracted_length=len(response.content),
        )

    return response


def prepare_messages(messages: list[BaseMessage], llm: BaseChatModel, system_prompt: str) -> list[BaseMessage]:
    """Prepare the messages for the LLM.

    Keeps the most recent messages that fit in MAX_CONTEXT_TOKENS (a separate budget
    from MAX_TOKENS, which caps the reply), then puts the system prompt first.

    Args:
        messages (list[BaseMessage]): The thread's messages from the graph state.
        llm (BaseChatModel): The LLM to use.
        system_prompt (str): The system prompt to use.

    Returns:
        list[BaseMessage]: The system prompt followed by the trimmed history.
    """
    try:
        trimmed_messages = _trim_messages(
            messages,
            strategy="last",
            token_counter=llm,
            max_tokens=settings.MAX_CONTEXT_TOKENS,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
    except ValueError as e:
        # Handle unrecognized content blocks (e.g., reasoning blocks from GPT-5)
        if "Unrecognized content block type" in str(e):
            logger.warning(
                "token_counting_failed_skipping_trim",
                error=str(e),
                message_count=len(messages),
            )
            # Skip trimming and return all messages
            trimmed_messages = messages
        else:
            raise

    return [SystemMessage(content=system_prompt)] + list(trimmed_messages)
