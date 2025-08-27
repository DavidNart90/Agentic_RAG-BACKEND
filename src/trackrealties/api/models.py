"""
Pydantic models for the TrackRealties API.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class UserSession(BaseModel):
    """
    Pydantic model for a user session.
    """

    id: UUID = Field(default_factory=uuid4)
    user_role: str
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class ConversationMessage(BaseModel):
    """
    Pydantic model for a conversation message.
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class ChatRequest(BaseModel):
    """
    Pydantic model for a chat request.
    """

    query: str
    session_id: Optional[UUID] = None


class ChatResponse(BaseModel):
    """
    Pydantic model for a chat response.
    """

    message: str
    session_id: UUID
