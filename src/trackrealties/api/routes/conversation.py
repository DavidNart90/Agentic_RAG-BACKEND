"""
API routes for conversation management.

This module provides API endpoints for managing conversations, including
creating, retrieving, and updating conversation messages and sessions.
"""

from typing import List, Optional
from uuid import UUID

from asyncpg import Connection
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from src.trackrealties.agents.context import ContextManager
from src.trackrealties.api.dependencies import (get_current_session,
                                                get_db_connection)
from src.trackrealties.data.conversation_repository import \
    ConversationRepository
from src.trackrealties.models.conversation import (ConversationFilter,
                                                   ConversationHistory,
                                                   ConversationMessage,
                                                   ConversationMessageResponse,
                                                   ConversationSummary,
                                                   MessageCreateRequest,
                                                   MessageUpdateRequest)
from src.trackrealties.models.session import ChatSession

router = APIRouter(prefix="/conversations", tags=["conversations"])

# In-memory manager for conversation contexts
context_manager = ContextManager()


@router.post("/messages", response_model=ConversationMessageResponse)
async def create_message(
    message: MessageCreateRequest,
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Create a new conversation message.

    Args:
        message: Message creation request
        session: Current user session
        db: Database connection

    Returns:
        Created conversation message
    """
    repo = ConversationRepository(db)

    try:
        # The session_id is now part of the MessageCreateRequest
        # and user_id is part of the session, so we pass the whole message object.
        return await repo.create_message(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create message: {str(e)}")


@router.get("/messages/{message_id}", response_model=ConversationMessageResponse)
async def get_message(
    message_id: UUID = Path(..., description="Message ID"),
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Get a message by ID.

    Args:
        message_id: Message ID
        user_id: Current user ID
        db: Database connection

    Returns:
        Conversation message
    """
    repo = ConversationRepository(db)

    message = await repo.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    return message


@router.patch("/messages/{message_id}", response_model=ConversationMessageResponse)
async def update_message(
    update: MessageUpdateRequest,
    message_id: UUID = Path(..., description="Message ID"),
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Update an existing message.

    Args:
        update: Message update request
        message_id: Message ID
        user_id: Current user ID
        db: Database connection

    Returns:
        Updated conversation message
    """
    repo = ConversationRepository(db)

    message = await repo.update_message(message_id, update)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    return message


@router.delete("/messages/{message_id}", response_model=dict)
async def delete_message(
    message_id: UUID = Path(..., description="Message ID"),
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Delete a message.

    Args:
        message_id: Message ID
        user_id: Current user ID
        db: Database connection

    Returns:
        Success message
    """
    repo = ConversationRepository(db)

    success = await repo.delete_message(message_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")

    return {"success": True, "message": "Message deleted successfully"}


@router.get("/sessions/{target_session_id}", response_model=ConversationHistory)
async def get_conversation_history(
    target_session_id: UUID = Path(..., description="Target session ID to get history for"),
    limit: int = Query(50, description="Maximum number of messages to return"),
    offset: int = Query(0, description="Number of messages to skip"),
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Get conversation history for a session.

    Args:
        target_session_id: Target session ID to get history for
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        session: Current user session
        db: Database connection

    Returns:
        Conversation history
    """
    repo = ConversationRepository(db)

    try:
        return await repo.get_conversation_history(target_session_id, limit, offset)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")


@router.get("/summaries", response_model=List[ConversationSummary])
async def get_conversation_summaries(
    user_id: Optional[UUID] = Query(None, description="Filter by user ID"),
    role: Optional[str] = Query(None, description="Filter by user role"),
    contains: Optional[str] = Query(None, description="Filter by text content"),
    property_ref: Optional[str] = Query(None, description="Filter by property reference"),
    market_ref: Optional[str] = Query(None, description="Filter by market reference"),
    limit: int = Query(50, description="Maximum number of summaries to return"),
    offset: int = Query(0, description="Number of summaries to skip"),
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Get conversation summaries based on filter criteria.

    Args:
        user_id: Filter by user ID
        role: Filter by user role
        contains: Filter by text content
        property_ref: Filter by property reference
        market_ref: Filter by market reference
        limit: Maximum number of summaries to return
        offset: Number of summaries to skip
        current_user: Current user ID
        db: Database connection

    Returns:
        List of conversation summaries
    """
    repo = ConversationRepository(db)

    # Create filter parameters
    filter_params = ConversationFilter(
        user_id=user_id,
        role_filter=role,
        contains_text=contains,
        property_reference=property_ref,
        market_reference=market_ref,
        limit=limit,
        offset=offset,
    )

    try:
        return await repo.get_conversation_summaries(filter_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation summaries: {str(e)}")


@router.delete("/sessions/{target_session_id}", response_model=dict)
async def delete_conversation(
    target_session_id: UUID = Path(..., description="Target session ID to delete"),
    session: ChatSession = Depends(get_current_session),
    db: Connection = Depends(get_db_connection),
):
    """
    Delete an entire conversation.

    Args:
        target_session_id: Target session ID to delete
        session: Current user session
        db: Database connection

    Returns:
        Success message
    """
    repo = ConversationRepository(db)

    success = await repo.delete_conversation(target_session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "message": "Conversation deleted successfully"}


@router.get("/context/{session_id}")
async def get_context(session_id: UUID):
    """Retrieve stored conversation context."""
    context = context_manager.get_context(str(session_id))
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")
    return JSONResponse(content=context.dict())


@router.delete("/context/{session_id}")
async def clear_context(session_id: UUID):
    """Clear stored conversation context."""
    context_manager.clear_context(str(session_id))
    return {"success": True, "message": "Context cleared"}
