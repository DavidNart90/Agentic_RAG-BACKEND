"""
Main FastAPI application file.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..core.config import settings
from ..core.database import db_pool, test_connection
from ..core.graph import graph_manager
from .routes import (agents, analytics, chat, conversation, health, market,
                     property, rag, session)

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    logger.info("Starting up TrackRealties AI API...")
    try:
        await db_pool.initialize()
        db_ok = await test_connection()
        if not db_ok:
            logger.error("Database connection failed on startup.")

        await graph_manager.initialize()
        graph_ok = await graph_manager.test_connection()
        if not graph_ok:
            logger.error("Graph database connection failed on startup.")

        # Initialize the RAG pipeline
        await rag.pipeline.initialize()

        logger.info("TrackRealties AI API startup complete.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    logger.info("Shutting down TrackRealties AI API...")
    await db_pool.close()
    await graph_manager.close()
    logger.info("Connections closed.")


app = FastAPI(
    title="TrackRealties AI",
    description="A RAG application for real estate analysis.",
    version="0.1.0",
    lifespan=lifespan,
)


# Placeholder for root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to TrackRealties AI"}


# Include the API routers

app.include_router(health.router, prefix="/health", tags=["Health"])  # Health check endpoint
app.include_router(session.router, prefix="/session", tags=["Session"])  # Session management
app.include_router(chat.router, prefix="/chat", tags=["Chat"])  # Chat interface
app.include_router(agents.router, prefix="/agents", tags=["Agents"])  # Agent management
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])  # Analytics endpoints
app.include_router(conversation.router, tags=["Conversation"])  # Conversation management
app.include_router(market.router, prefix="/market", tags=["Market"])  # Market data endpoints
app.include_router(property.router, prefix="/property", tags=["Property"])  # Property listing endpoints
app.include_router(rag.router, prefix="/rag", tags=["RAG"])  # RAG pipeline endpoints
