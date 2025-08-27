import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from src.trackrealties.models.base import Base


class ChatSession(Base):
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(200), nullable=True)
    user_role = Column(String(50), nullable=False)
    session_data = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), server_default=text("(now() + interval '1 hour')"))
    is_active = Column(Boolean, nullable=False, server_default=text("true"))

    __table_args__ = (
        CheckConstraint(
            "user_role IN ('investor', 'developer', 'buyer', 'agent', 'general')", name="user_sessions_user_role_check"
        ),
    )
