"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from typing import (
    Any,
    AsyncGenerator,
    Optional,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RunnableConfig,
    StateSnapshot,
)
from mem0 import AsyncMemory
from psycopg_pool import AsyncConnectionPool

from src.agent.prompts import load_system_prompt
from src.agent.tools import tools
from src.config.settings import settings
from src.data.schemas import (
    ChatMessage,
    GraphState,
    Message,
)
from src.services.llm_provider import llm_service
from src.system.logs import logger
from src.utils import (
    dump_messages,
    prepare_messages,
    process_llm_response,
)
from src.utils.graph import extract_text

# The graph node whose model output is streamed to clients
CHAT_NODE = "chat"


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Tools are bound to every model in the registry, so fallbacks can call them too
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self._graph_lock = asyncio.Lock()
        self.memory: Optional[AsyncMemory] = None
        self._memory_lock = asyncio.Lock()
        # Strong references to fire-and-forget tasks, so they aren't garbage-collected mid-flight
        self._background_tasks: set[asyncio.Task] = set()
        logger.info(
            "langgraph_agent_initialized",
            model=self.llm_service.default_model_name,
            environment=settings.ENVIRONMENT.value,
        )

    async def _long_term_memory(self) -> AsyncMemory:
        """Initialize the long term memory (once per process)."""
        if self.memory is None:
            async with self._memory_lock:
                if self.memory is None:
                    self.memory = await AsyncMemory.from_config(
                        config_dict={
                            "vector_store": {
                                "provider": "pgvector",
                                "config": {
                                    "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                                    "dbname": settings.POSTGRES_DB,
                                    "user": settings.POSTGRES_USER,
                                    "password": settings.POSTGRES_PASSWORD,
                                    "host": settings.POSTGRES_HOST,
                                    "port": settings.POSTGRES_PORT,
                                },
                            },
                            "llm": {
                                "provider": "openai",
                                "config": {"model": settings.LONG_TERM_MEMORY_MODEL},
                            },
                            "embedder": {
                                "provider": "openai",
                                "config": {"model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL},
                            },
                        }
                    )
        return self.memory

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get the PostgreSQL connection pool used by the checkpointer.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.

        Raises:
            Exception: If the pool cannot be opened. Without it there is no conversation
                history, so the error is surfaced instead of silently degrading.
        """
        if self._connection_pool is None:
            pool = AsyncConnectionPool(
                settings.POSTGRES_DSN,
                open=False,
                max_size=settings.POSTGRES_POOL_SIZE,
                kwargs={
                    "autocommit": True,
                    "connect_timeout": 5,
                    "prepare_threshold": None,
                },
            )
            try:
                await pool.open(wait=True, timeout=10)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                raise
            self._connection_pool = pool
            logger.info(
                "connection_pool_created", max_size=settings.POSTGRES_POOL_SIZE, environment=settings.ENVIRONMENT.value
            )
        return self._connection_pool

    async def _get_relevant_memory(self, user_id: str, query: str) -> str:
        """Get the relevant memory for the user and query.

        Args:
            user_id (str): The user ID.
            query (str): The query to search for.

        Returns:
            str: The relevant memory, one bullet per fact, or "" if none or on error.
        """
        try:
            memory = await self._long_term_memory()
            results = await memory.search(user_id=str(user_id), query=query)
            facts = [result["memory"] for result in results.get("results", [])]
            logger.debug("long_term_memory_recalled", user_id=user_id, fact_count=len(facts))
            return "\n".join(f"* {fact}" for fact in facts)
        except Exception as e:
            logger.error("failed_to_get_relevant_memory", error=str(e), user_id=user_id)
            return ""

    async def _update_long_term_memory(self, user_id: str, messages: list[dict], metadata: dict = None) -> None:
        """Update the long term memory.

        Args:
            user_id (str): The user ID.
            messages (list[dict]): The new turn (user message(s) and the final reply).
            metadata (dict): Optional metadata to include.
        """
        try:
            memory = await self._long_term_memory()
            await memory.add(messages, user_id=str(user_id), metadata=metadata)
            logger.info("long_term_memory_updated_successfully", user_id=user_id, message_count=len(messages))
        except Exception as e:
            logger.exception(
                "failed_to_update_long_term_memory",
                user_id=user_id,
                error=str(e),
            )

    def _schedule_memory_update(
        self, user_id: Optional[str], sent: list[ChatMessage], final_messages: list[BaseMessage], metadata: dict
    ) -> None:
        """Send the new turn to mem0 in the background, keeping a reference to the task.

        Only the messages of this turn are sent, not the whole thread, so the cost of a
        memory update stays flat as the conversation grows.

        Args:
            user_id: The user ID; nothing is stored without one.
            sent: The messages the client sent this turn.
            final_messages: The thread's messages after the run.
            metadata: Metadata stored with the memories.
        """
        if not user_id:
            return
        turn = [{"role": m.role, "content": m.content} for m in sent if m.role == "user"]
        reply = final_messages[-1] if final_messages else None
        if isinstance(reply, AIMessage):
            text = extract_text(reply.content)
            if text:
                turn.append({"role": "assistant", "content": text})
        if not turn:
            return
        task = asyncio.create_task(self._update_long_term_memory(user_id, turn, metadata))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.
            config (RunnableConfig): The run config (thread id, callbacks).

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        system_prompt = load_system_prompt(long_term_memory=state.long_term_memory)

        # System prompt + the most recent history that fits in MAX_CONTEXT_TOKENS
        messages = prepare_messages(state.messages, self.llm_service.get_llm(), system_prompt)

        try:
            # Retries, then fallback through the model registry; latency is recorded per model
            response_message = await self.llm_service.call(messages)
        except Exception as e:
            logger.error(
                "llm_call_failed",
                session_id=config["configurable"]["thread_id"],
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise

        # Process response to handle structured content blocks
        response_message = process_llm_response(response_message)

        logger.info(
            "llm_response_generated",
            session_id=config["configurable"]["thread_id"],
            tool_calls=len(getattr(response_message, "tool_calls", []) or []),
        )

        # Determine next node based on whether there are tool calls
        goto = "tool_call" if getattr(response_message, "tool_calls", None) else END
        return Command(update={"messages": [response_message]}, goto=goto)

    async def _tool_call(self, state: GraphState) -> Command:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            tool = self.tools_by_name.get(tool_call["name"])
            if tool is None:
                logger.warning("unknown_tool_requested", tool=tool_call["name"])
                result = f"Error: unknown tool '{tool_call['name']}'"
            else:
                result = await tool.ainvoke(tool_call["args"])
            outputs.append(ToolMessage(content=result, name=tool_call["name"], tool_call_id=tool_call["id"]))
        return Command(update={"messages": outputs}, goto=CHAT_NODE)

    async def create_graph(self) -> CompiledStateGraph:
        """Create and configure the LangGraph workflow (once per process).

        Returns:
            CompiledStateGraph: The configured LangGraph instance.

        Raises:
            Exception: If the checkpointer cannot be set up.
        """
        if self._graph is not None:
            return self._graph
        async with self._graph_lock:
            if self._graph is not None:
                return self._graph

            graph_builder = StateGraph(GraphState)
            graph_builder.add_node(CHAT_NODE, self._chat, ends=["tool_call", END])
            graph_builder.add_node("tool_call", self._tool_call, ends=[CHAT_NODE])
            graph_builder.set_entry_point(CHAT_NODE)
            graph_builder.set_finish_point(CHAT_NODE)

            connection_pool = await self._get_connection_pool()
            checkpointer = AsyncPostgresSaver(connection_pool)
            await checkpointer.setup()

            self._graph = graph_builder.compile(
                checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
            )
            logger.info(
                "graph_created", graph_name=f"{settings.PROJECT_NAME} Agent", environment=settings.ENVIRONMENT.value
            )
        return self._graph

    def _build_config(self, session_id: str, user_id: Optional[str]) -> RunnableConfig:
        """Build the run config: checkpoint thread, Langfuse callback and trace attributes.

        Langfuse 3.x reads user, session and tags from these `langfuse_*` metadata keys.

        Args:
            session_id: The chat session id, used as the checkpoint thread id.
            user_id: The user id.

        Returns:
            RunnableConfig: The config for one graph run.
        """
        return {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "langfuse_user_id": str(user_id) if user_id is not None else None,
                "langfuse_session_id": session_id,
                "langfuse_tags": [settings.ENVIRONMENT.value],
                "environment": settings.ENVIRONMENT.value,
            },
        }

    async def get_response(
        self,
        messages: list[ChatMessage],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[Message]:
        """Get a response from the LLM.

        Args:
            messages (list[ChatMessage]): The new messages for this turn.
            session_id (str): The session ID (checkpoint thread and Langfuse session).
            user_id (Optional[str]): The user ID for memory and Langfuse.

        Returns:
            list[Message]: The thread's user and assistant messages after this turn.

        Raises:
            Exception: Any failure of the graph run, after logging it.
        """
        graph = await self.create_graph()
        config = self._build_config(session_id, user_id)
        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."
        try:
            # Run in a child task: callback handlers that attach OTel context during the run
            # (Langfuse does) then can't leave a stale span in this request's context.
            response = await asyncio.create_task(
                graph.ainvoke(
                    input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                    config=config,
                )
            )
        except Exception as e:
            logger.error("graph_run_failed", session_id=session_id, error=str(e))
            raise
        self._schedule_memory_update(user_id, messages, response["messages"], config["metadata"])
        return self.__process_messages(response["messages"])

    @staticmethod
    def _stream_text(chunk: Any, metadata: dict) -> str:
        """Return the text to send to the client for one item of the messages stream.

        Only model output from the chat node is forwarded: tool results (ToolMessages
        from the tool_call node) and reasoning blocks never reach the client.

        Args:
            chunk: The message or message chunk LangGraph emitted.
            metadata: LangGraph's metadata for it (includes `langgraph_node`).

        Returns:
            str: Text to forward, possibly empty.
        """
        if metadata.get("langgraph_node") != CHAT_NODE or not isinstance(chunk, AIMessage):
            return ""
        return extract_text(chunk.content)

    async def get_stream_response(
        self, messages: list[ChatMessage], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[ChatMessage]): The new messages for this turn.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Text chunks of the chat node's response.
        """
        graph = await self.create_graph()
        config = self._build_config(session_id, user_id)
        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."

        try:
            async for chunk, metadata in graph.astream(
                {"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config,
                stream_mode="messages",
            ):
                text = self._stream_text(chunk, metadata)
                if text:
                    yield text

            # After streaming completes, read the final state and update memory in background
            state: StateSnapshot = await graph.aget_state(config)
            if state.values and "messages" in state.values:
                self._schedule_memory_update(user_id, messages, state.values["messages"], config["metadata"])
        except Exception as stream_error:
            logger.error("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        graph = await self.create_graph()
        state: StateSnapshot = await graph.aget_state(config={"configurable": {"thread_id": session_id}})
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        """Keep the user and assistant messages that have text, as API messages."""
        openai_style_messages = convert_to_openai_messages(messages)
        result = []
        for message in openai_style_messages:
            if message["role"] not in ("assistant", "user"):
                continue
            text = extract_text(message["content"])
            if text:
                result.append(Message(role=message["role"], content=text))
        return result

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        conn_pool = await self._get_connection_pool()
        async with conn_pool.connection() as conn:
            for table in settings.CHECKPOINT_TABLES:
                try:
                    await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                    logger.info("checkpoint_table_cleared", table=table, session_id=session_id)
                except Exception as e:
                    logger.error("checkpoint_table_clear_failed", table=table, error=str(e))
                    raise

    async def shutdown(self) -> None:
        """Wait briefly for pending memory updates, then close the checkpointer pool."""
        if self._background_tasks:
            await asyncio.wait(self._background_tasks, timeout=10)
        if self._connection_pool is not None:
            await self._connection_pool.close()
            self._connection_pool = None


# Process-wide agent shared by the chatbot and auth routes
langgraph_agent = LangGraphAgent()
