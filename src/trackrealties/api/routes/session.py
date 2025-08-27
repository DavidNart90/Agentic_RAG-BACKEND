"""
API route for session management.
"""

import logging
from uuid import UUID

from asyncpg import Connection
from fastapi import APIRouter, Depends, HTTPException

from ...data.repository import SessionRepository
from ...models.api import SessionCreateRequest, SessionCreateResponse
from ..dependencies import get_db_connection

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest, conn: Connection = Depends(get_db_connection)):
    """Creates a new user session."""
    try:
        repo = SessionRepository(conn)
        session = await repo.create_user_session(role=request.role, user_id=request.user_id)
        return SessionCreateResponse(session_id=str(session.id), role=session.user_role, created_at=session.created_at)
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session.")


@router.get("/{session_id}")
async def get_session(session_id: UUID, conn: Connection = Depends(get_db_connection)):
    """Gets session details."""
    try:
        repo = SessionRepository(conn)
        session = await repo.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or has expired.")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve session.")
