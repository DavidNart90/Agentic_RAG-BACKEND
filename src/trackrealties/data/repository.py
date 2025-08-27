"""
Data access layer for all database operations.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from asyncpg import Connection

from ..models.db import ConversationMessage, MessageRole, Session, UserRole

logger = logging.getLogger(__name__)


class SessionRepository:
    def __init__(self, conn: Connection):
        self.conn = conn

    async def create_user_session(
        self,
        role: UserRole,
        user_id: Optional[str] = None,
        session_data: Optional[Dict[str, Any]] = None,
        timeout_minutes: int = 60,
    ) -> Session:
        """Creates a new user session and returns the session object."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)

        row = await self.conn.fetchrow(
            """
            INSERT INTO user_sessions (user_id, user_role, session_data, expires_at)
            VALUES ($1, $2, $3, $4)
            RETURNING id, user_id, user_role, session_data, created_at, updated_at, expires_at, is_active
            """,
            user_id,
            role.value if isinstance(role, Enum) else role,
            json.dumps(session_data or {}),
            expires_at,
        )
        data = dict(row)
        data["session_data"] = json.loads(data["session_data"])
        return Session(**data)

    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """Retrieves a session by its ID."""
        row = await self.conn.fetchrow(
            """
            SELECT id, user_id, user_role, session_data, created_at, updated_at, expires_at, is_active
            FROM user_sessions
            WHERE id = $1 AND is_active = TRUE AND expires_at > NOW()
            """,
            session_id,
        )
        if row:
            data = dict(row)
            data["session_data"] = json.loads(data["session_data"])
            return Session(**data)
        return None


class MessageRepository:
    def __init__(self, conn: Connection):
        self.conn = conn

    async def add_conversation_message(
        self,
        session_id: UUID,
        role: MessageRole,
        content: str,
        tools_used: Optional[List[Dict[str, Any]]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
        token_count: Optional[int] = None,
        message_metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Adds a message to the conversation_messages table."""
        row = await self.conn.fetchrow(
            """
            INSERT INTO conversation_messages (
                session_id, role, content, tools_used, validation_result,
                confidence_score, processing_time_ms, token_count, message_metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id, session_id, role, content, tools_used, validation_result, confidence_score,
                      processing_time_ms, token_count, message_metadata, created_at
            """,
            session_id,
            role.value if isinstance(role, Enum) else role,
            content,
            json.dumps(tools_used or []),
            json.dumps(validation_result) if validation_result else None,
            confidence_score,
            processing_time_ms,
            token_count,
            json.dumps(message_metadata or {}),
        )
        data = dict(row)
        data["tools_used"] = json.loads(data["tools_used"])
        if data.get("message_metadata"):
            data["message_metadata"] = json.loads(data["message_metadata"])
        return ConversationMessage(**data)

    async def get_conversation_history(self, session_id: UUID, limit: int = 10) -> List[ConversationMessage]:
        """Retrieves the recent conversation history for a session."""
        rows = await self.conn.fetch(
            """
            SELECT id, session_id, role, content, created_at, message_metadata, tools_used
            FROM conversation_messages
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
        return [ConversationMessage(**self._process_message_row(row)) for row in reversed(rows)]

    def _process_message_row(self, row: dict) -> dict:
        data = dict(row)
        if data.get("tools_used") and isinstance(data["tools_used"], str):
            data["tools_used"] = json.loads(data["tools_used"])
        if data.get("message_metadata") and isinstance(data["message_metadata"], str):
            data["message_metadata"] = json.loads(data["message_metadata"])
        return data
