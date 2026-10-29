"""This file contains the chat schema for the application."""

import re
from typing import (
    List,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class Message(BaseModel):
    """A message returned by the API (replies and stored history).

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message. No length cap: replies can be long.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")


class ChatMessage(BaseModel):
    """A message sent by the client.

    Clients cannot send "system" messages: the system prompt is built on the server.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate the message content.

        Args:
            v: The content to validate

        Returns:
            str: The validated content

        Raises:
            ValueError: If the content contains disallowed patterns
        """
        # Check for potentially harmful content
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")

        # Check for null bytes
        if "\0" in v:
            raise ValueError("Content contains null bytes")

        return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        messages: The new turn. The thread's history is stored on the server
            (LangGraph checkpointer), so send only messages it hasn't seen.
    """

    messages: List[ChatMessage] = Field(
        ...,
        description="New messages for this turn, usually one user message. History is kept server-side.",
        min_length=1,
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: List[Message] = Field(..., description="List of messages in the conversation")


class StreamResponse(BaseModel):
    """Response model for streaming chat endpoint.

    Attributes:
        content: The content of the current chunk.
        done: Whether the stream is complete.
        error: Set on the final frame when the stream failed.
    """

    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")
    error: Optional[str] = Field(default=None, description="Error message if the stream failed")
