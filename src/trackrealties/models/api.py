"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    INVESTOR = "investor"
    DEVELOPER = "developer"
    BUYER = "buyer"
    AGENT = "agent"
    GENERAL = "general"


class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    role: UserRole


class SessionCreateResponse(BaseModel):
    session_id: str
    role: UserRole
    created_at: datetime


class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None


class ToolCall(BaseModel):
    tool_name: str
    args: Dict[str, Any]


class ChatResponse(BaseModel):
    message: str
    session_id: str
    assistant_message_id: str
    tools_used: List[ToolCall] = []
    metadata: Dict[str, Any] = {}


# Streaming response models
class StreamDelta(BaseModel):
    """Stream delta for Server-Sent Events."""

    type: str = Field(..., description="Type of stream event: 'session', 'text', 'error', 'end'")
    content: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="Content of the stream event. String for text, dict for structured data"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the stream event")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthStatus(BaseModel):
    status: str
    database: bool
    graph_database: bool
    llm_connection: bool
    version: str
    timestamp: datetime
