"""
Pydantic models representing database tables.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    INVESTOR = "investor"
    DEVELOPER = "developer"
    BUYER = "buyer"
    AGENT = "agent"
    GENERAL = "general"


class Session(BaseModel):
    """Represents the 'user_sessions' table."""

    id: UUID = Field(default_factory=uuid.uuid4)
    user_id: Optional[str] = None
    user_role: UserRole
    session_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    expires_at: datetime
    is_active: bool = True

    class Config:
        orm_mode = True


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Represents the 'conversation_messages' table."""

    id: UUID = Field(default_factory=uuid.uuid4)
    session_id: UUID
    role: MessageRole
    content: str
    tools_used: List[Dict[str, Any]] = Field(default_factory=list)
    validation_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    processing_time_ms: Optional[int] = None
    token_count: Optional[int] = None
    message_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        orm_mode = True
