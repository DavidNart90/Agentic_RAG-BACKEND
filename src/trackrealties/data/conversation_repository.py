"""
Data repository for conversation-related database operations.
"""

import json
import logging
from typing import Optional
from uuid import UUID

from asyncpg import Connection

from src.trackrealties.models.session import ChatSession

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Encapsulates database logic for user sessions and conversation messages.
    """

    def __init__(self, db_connection: Connection):
        self.db = db_connection

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = "buyer",
        session_data: Optional[dict] = None,
    ) -> "ChatSession":
        """
        Get an existing session or create a new one.
        """
        if session_id:
            session_record = await self.db.fetchrow("SELECT * FROM user_sessions WHERE id = $1", session_id)
            if session_record:
                return ChatSession(**session_record)

        result = await self.db.fetchrow(
            """
            INSERT INTO user_sessions (user_role, user_id)
            VALUES ($1, $2)
            RETURNING *
            """,
            user_role,
            user_id,
        )
        return ChatSession(**result)

    async def log_message(self, session_id: UUID, role: str, content: str) -> None:
        """
        Logs a new message to the conversation_messages table.

        Args:
            session_id: The ID of the session to which the message belongs.
            role: The role of the message author (e.g., 'user', 'assistant').
            content: The text content of the message.
        """
        await self.db.execute(
            """
            INSERT INTO conversation_messages (session_id, role, content)
            VALUES ($1, $2, $3)
            """,
            session_id,
            role,
            content,
        )
        logger.info(f"Logged message for session {session_id} from role '{role}'.")

    async def get_session_role(self, session_id: UUID) -> Optional[str]:
        """
        Retrieves the user role for a given session.

        Args:
            session_id: The ID of the session.

        Returns:
            The user role as a string, or None if the session is not found.
        """
        role = await self.db.fetchval("SELECT user_role FROM user_sessions WHERE id = $1", session_id)
        if role:
            logger.info(f"Retrieved role '{role}' for session {session_id}.")
        else:
            logger.warning(f"No session found with ID: {session_id}")
        return role

    async def get_conversation_history(self, session_id: UUID, limit: int = 50, offset: int = 0):
        """
        Retrieves conversation history for a given session.

        Args:
            session_id: The ID of the session.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip.

        Returns:
            ConversationHistory object with session and messages.
        """
        from ..models.conversation import (ConversationHistory,
                                           ConversationMessageResponse,
                                           SessionResponse)

        # Get session info
        session_record = await self.db.fetchrow("SELECT * FROM user_sessions WHERE id = $1", session_id)

        if not session_record:
            raise ValueError(f"Session {session_id} not found")

        # Convert to SessionResponse model
        session_data = dict(session_record)

        # Handle JSONB fields that might be strings
        if isinstance(session_data.get("session_data"), str):
            session_data["session_data"] = json.loads(session_data["session_data"])
        elif session_data.get("session_data") is None:
            session_data["session_data"] = {}

        session = SessionResponse(**session_data)

        # Get messages for this session
        messages_query = """
            SELECT * FROM conversation_messages 
            WHERE session_id = $1 
            ORDER BY created_at ASC 
            LIMIT $2 OFFSET $3
        """

        message_records = await self.db.fetch(messages_query, session_id, limit, offset)

        # Convert to response models
        messages = []
        for record in message_records:
            # Convert record to dict and handle JSONB fields properly
            message_data = dict(record)

            # Handle JSONB fields that might be strings
            if isinstance(message_data.get("tools_used"), str):
                message_data["tools_used"] = json.loads(message_data["tools_used"])
            elif message_data.get("tools_used") is None:
                message_data["tools_used"] = []

            if isinstance(message_data.get("message_metadata"), str):
                message_data["message_metadata"] = json.loads(message_data["message_metadata"])
            elif message_data.get("message_metadata") is None:
                message_data["message_metadata"] = {}

            if isinstance(message_data.get("validation_result"), str):
                message_data["validation_result"] = json.loads(message_data["validation_result"])

            messages.append(ConversationMessageResponse(**message_data))

        # Get total count for pagination
        total_count = await self.db.fetchval(
            "SELECT COUNT(*) FROM conversation_messages WHERE session_id = $1", session_id
        )

        has_more = (offset + len(messages)) < total_count

        return ConversationHistory(session=session, messages=messages, total_messages=total_count, has_more=has_more)
