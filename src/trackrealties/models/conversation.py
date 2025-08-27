import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (CheckConstraint, Column, DateTime, ForeignKey, Integer,
                        Numeric, String, Text, text)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.sql import func

from src.trackrealties.models.base import Base

from .session import ChatSession


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("user_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tools_used = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    validation_result = Column(JSONB, nullable=True)
    confidence_score = Column(Numeric(3, 2), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    message_metadata = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="conversation_messages_role_check"),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1", name="conversation_messages_confidence_score_check"
        ),
    )


# Pydantic Models for API Interaction


class MessageCreateRequest(BaseModel):
    session_id: uuid.UUID
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    tools_used: list[dict[str, Any]] | None = Field(default_factory=list)
    validation_result: dict[str, Any] | None = None
    confidence_score: float | None = Field(None, ge=0, le=1)
    processing_time_ms: int | None = None
    token_count: int | None = None
    message_metadata: dict[str, Any] | None = Field(default_factory=dict)


class MessageUpdateRequest(BaseModel):
    content: str | None = None
    tools_used: list[dict[str, Any]] | None = None
    validation_result: dict[str, Any] | None = None
    confidence_score: float | None = Field(None, ge=0, le=1)
    message_metadata: dict[str, Any] | None = None


class ConversationMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    tools_used: list[dict[str, Any]]
    validation_result: dict[str, Any] | None
    confidence_score: float | None
    processing_time_ms: int | None
    token_count: int | None
    message_metadata: dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    id: uuid.UUID
    user_id: str | None
    user_role: str
    session_data: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class ConversationHistory(BaseModel):
    session: SessionResponse
    messages: list[ConversationMessageResponse]
    total_messages: int
    has_more: bool

    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    session_id: uuid.UUID
    user_id: str
    title: str
    last_message_at: datetime
    message_count: int
    topics: list[str]
    preview: str


class ConversationFilter(BaseModel):
    user_id: uuid.UUID | None = None
    role_filter: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    contains_text: str | None = None
    property_reference: str | None = None
    market_reference: str | None = None
    limit: int = 50
    offset: int = 0
