"""
Core graph database utilities.
"""

import logging
from typing import Optional

from .config import settings as global_settings

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncDriver = None

logger = logging.getLogger(__name__)


class GraphManager:
    """Manages Neo4j graph database connections and operations."""

    def __init__(self, settings: Optional[dict] = None):
        if settings is None:
            
            settings = global_settings
        self.settings = settings
        from typing import Any

        self._driver: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the Neo4j driver."""
        if self._driver is not None:
            return
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available. Graph functionality will be limited.")
            return

        # Access Neo4j settings with proper attribute names
        neo4j_uri = getattr(self.settings, "NEO4J_URI", None) or getattr(self.settings, "neo4j_uri", None)
        neo4j_user = getattr(self.settings, "NEO4J_USER", None) or getattr(self.settings, "neo4j_user", None)
        neo4j_password = getattr(self.settings, "NEO4J_PASSWORD", None) or getattr(
            self.settings, "neo4j_password", None
        )

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            logger.error(
                f"Neo4j connection details not configured. URI: {neo4j_uri}, User: {neo4j_user}, Password: {'***' if neo4j_password else None}"
            )
            return

        try:
            self._driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info(f"Neo4j driver initialized for {neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            self._driver = None

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    def get_session(self, database: Optional[str] = None):
        """Get a Neo4j session context manager."""
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized. Call initialize() first.")

        # Access database name with proper attribute handling
        db_name = (
            database or getattr(self.settings, "NEO4J_DATABASE", None) or getattr(self.settings, "neo4j_database", "neo4j")
        )
        return self._driver.session(database=db_name)

    async def test_connection(self) -> bool:
        """Test the Neo4j connection."""
        if not self._driver:
            await self.initialize()

        if not self._driver:
            logger.error("Neo4j driver not initialized, cannot test connection.")
            return False

        try:
            # Access database name with proper attribute handling
            db_name = getattr(self.settings, "NEO4J_DATABASE", None) or getattr(
                self.settings, "neo4j_database", "neo4j"
            )
            async with self._driver.session(database=db_name) as session:
                logger.info("Testing Neo4j connection...")
                result = await session.run("RETURN 1")
                await result.single()
                logger.info("Neo4j connection test successful.")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}", exc_info=True)
            return False


# Global graph manager instance
graph_manager = GraphManager()


async def test_graph_connection() -> bool:
    """Test the graph database connection."""
    return await graph_manager.test_connection()
