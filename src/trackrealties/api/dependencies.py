"""
API dependencies.
"""

from typing import AsyncGenerator, Optional
from uuid import UUID

from asyncpg import Connection
from fastapi import Depends, Header, HTTPException
from starlette import status

from ..analytics.cma_engine import ComparativeMarketAnalysis
from ..core.database import db_pool
from ..data.repository import SessionRepository
from ..models.session import ChatSession


async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """
    Dependency to get a database connection from the pool.
    """
    async with db_pool.acquire() as connection:
        yield connection


def get_cma_engine(db: Connection = Depends(get_db_connection)) -> ComparativeMarketAnalysis:
    """
    Dependency to get a CMA engine instance.
    """
    return ComparativeMarketAnalysis(db)


async def get_current_session(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"), db: Connection = Depends(get_db_connection)
) -> ChatSession:
    """
    Dependency to get the current user session from the request headers.
    """
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session ID not provided in X-Session-ID header",
        )

    try:
        session_uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format.",
        )

    repo = SessionRepository(db)
    session = await repo.get_session(session_uuid)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or has expired.",
        )

    return session
