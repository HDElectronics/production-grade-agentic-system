"""This file contains the database service for the application."""

from typing import (
    List,
    Optional,
)

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import (
    col,
    select,
)
from sqlmodel.ext.asyncio.session import AsyncSession

from src.config.settings import settings
from src.data.models.session import Session as ChatSession
from src.data.models.user import User
from src.system.logs import logger


class DatabaseService:
    """Service class for database operations.

    This class handles all database operations for Users and Sessions on an async
    SQLAlchemy engine (psycopg 3), so queries never block the event loop. The schema
    is owned by Alembic migrations (see migrations/), not created here.
    """

    def __init__(self):
        """Create the async engine. No connection is opened until the first query."""
        self.engine: AsyncEngine = create_async_engine(
            settings.POSTGRES_ASYNC_URL,
            pool_pre_ping=True,
            pool_size=settings.POSTGRES_POOL_SIZE,
            max_overflow=settings.POSTGRES_MAX_OVERFLOW,
            pool_timeout=30,  # Connection timeout (seconds)
            pool_recycle=1800,  # Recycle connections after 30 minutes
        )
        self._sessionmaker = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        logger.info(
            "database_engine_created",
            environment=settings.ENVIRONMENT.value,
            pool_size=settings.POSTGRES_POOL_SIZE,
            max_overflow=settings.POSTGRES_MAX_OVERFLOW,
        )

    async def create_user(self, email: str, password: str) -> User:
        """Create a new user.

        Args:
            email: User's email address
            password: Hashed password

        Returns:
            User: The created user
        """
        async with self._sessionmaker() as session:
            user = User(email=email, hashed_password=password)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info("user_created", user_id=user.id)
            return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get a user by ID.

        Args:
            user_id: The ID of the user to retrieve

        Returns:
            Optional[User]: The user if found, None otherwise
        """
        async with self._sessionmaker() as session:
            return await session.get(User, user_id)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email.

        Args:
            email: The email of the user to retrieve

        Returns:
            Optional[User]: The user if found, None otherwise
        """
        async with self._sessionmaker() as session:
            result = await session.exec(select(User).where(User.email == email))
            return result.first()

    async def delete_user_by_email(self, email: str) -> bool:
        """Delete a user by email.

        Args:
            email: The email of the user to delete

        Returns:
            bool: True if deletion was successful, False if user not found
        """
        async with self._sessionmaker() as session:
            user = (await session.exec(select(User).where(User.email == email))).first()
            if not user:
                return False

            await session.delete(user)
            await session.commit()
            logger.info("user_deleted", user_id=user.id)
            return True

    async def create_session(self, session_id: str, user_id: int, name: str = "") -> ChatSession:
        """Create a new chat session.

        Args:
            session_id: The ID for the new session
            user_id: The ID of the user who owns the session
            name: Optional name for the session (defaults to empty string)

        Returns:
            ChatSession: The created session
        """
        async with self._sessionmaker() as session:
            chat_session = ChatSession(id=session_id, user_id=user_id, name=name)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            logger.info("session_created", session_id=session_id, user_id=user_id, name=name)
            return chat_session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: The ID of the session to delete

        Returns:
            bool: True if deletion was successful, False if session not found
        """
        async with self._sessionmaker() as session:
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                return False

            await session.delete(chat_session)
            await session.commit()
            logger.info("session_deleted", session_id=session_id)
            return True

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID.

        Args:
            session_id: The ID of the session to retrieve

        Returns:
            Optional[ChatSession]: The session if found, None otherwise
        """
        async with self._sessionmaker() as session:
            return await session.get(ChatSession, session_id)

    async def get_user_sessions(self, user_id: int) -> List[ChatSession]:
        """Get all sessions for a user.

        Args:
            user_id: The ID of the user

        Returns:
            List[ChatSession]: List of user's sessions
        """
        async with self._sessionmaker() as session:
            statement = select(ChatSession).where(ChatSession.user_id == user_id).order_by(col(ChatSession.created_at))
            return list((await session.exec(statement)).all())

    async def update_session_name(self, session_id: str, name: str) -> ChatSession:
        """Update a session's name.

        Args:
            session_id: The ID of the session to update
            name: The new name for the session

        Returns:
            ChatSession: The updated session

        Raises:
            HTTPException: If session is not found
        """
        async with self._sessionmaker() as session:
            chat_session = await session.get(ChatSession, session_id)
            if not chat_session:
                raise HTTPException(status_code=404, detail="Session not found")

            chat_session.name = name
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            logger.info("session_name_updated", session_id=session_id, name=name)
            return chat_session

    async def health_check(self) -> bool:
        """Check database connection health.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return False

    async def dispose(self) -> None:
        """Close every pooled connection (called on shutdown)."""
        await self.engine.dispose()


# The one instance used by the whole process (routes, auth dependencies, /health)
db_manager = DatabaseService()
