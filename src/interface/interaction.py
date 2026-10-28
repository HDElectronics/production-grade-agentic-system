"""Chatbot API endpoints for handling chat interactions.

This module provides endpoints for chat interactions, including regular chat,
streaming chat, message history management, and chat history clearing.
"""

import json
import time

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse

from src.agent.workflow import langgraph_agent as agent
from src.config.settings import settings
from src.data.models.session import Session
from src.data.schemas.chat import (
    ChatRequest,
    ChatResponse,
    StreamResponse,
)
from src.interface.auth import get_current_session
from src.system.logs import logger
from src.system.rate_limit import limiter
from src.system.telemetry import llm_stream_duration_seconds

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def chat(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The new messages for this turn.
        session: The current session from the auth token.

    Returns:
        ChatResponse: The thread's user and assistant messages after this turn.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    logger.info("chat_request_received", session_id=session.id, message_count=len(chat_request.messages))
    try:
        result = await agent.get_response(chat_request.messages, session.id, user_id=session.user_id)
    except Exception as e:
        # Details go to the logs (with the request id); clients get a stable message
        logger.error("chat_request_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat request")

    logger.info("chat_request_processed", session_id=session.id)
    return ChatResponse(messages=result)


@router.post("/chat/stream")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat_stream"][0])
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    session: Session = Depends(get_current_session),
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The new messages for this turn.
        session: The current session from the auth token.

    Returns:
        StreamingResponse: Server-sent events, one frame per text chunk, then a done frame.
    """
    logger.info("stream_chat_request_received", session_id=session.id, message_count=len(chat_request.messages))

    async def event_generator():
        """Generate streaming events.

        Yields:
            str: Server-sent events in JSON format.
        """
        started = time.perf_counter()
        try:
            async for chunk in agent.get_stream_response(chat_request.messages, session.id, user_id=session.user_id):
                yield f"data: {StreamResponse(content=chunk, done=False).model_dump_json()}\n\n"

            yield f"data: {StreamResponse(content='', done=True).model_dump_json()}\n\n"
        except Exception as e:
            logger.error("stream_chat_request_failed", session_id=session.id, error=str(e), exc_info=True)
            error_frame = StreamResponse(content="", done=True, error="Failed to process chat request")
            yield f"data: {error_frame.model_dump_json()}\n\n"
        finally:
            llm_stream_duration_seconds.labels(model=agent.llm_service.default_model_name).observe(
                time.perf_counter() - started
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        # Tell proxies (nginx, Cloudflare) not to buffer or cache the stream
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/messages", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def get_session_messages(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Get all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        ChatResponse: All messages in the session.

    Raises:
        HTTPException: If there's an error retrieving the messages.
    """
    try:
        messages = await agent.get_chat_history(session.id)
        return ChatResponse(messages=messages)
    except Exception as e:
        logger.error("get_messages_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load messages")


@router.delete("/messages")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["messages"][0])
async def clear_chat_history(
    request: Request,
    session: Session = Depends(get_current_session),
):
    """Clear all messages for a session.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        dict: A message indicating the chat history was cleared.
    """
    try:
        await agent.clear_chat_history(session.id)
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error("clear_chat_history_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear chat history")
