"""
Database Integration Module for TrackRealties AI Platform.

This module provides functionality to save chunks and embeddings to the database
efficiently, with transaction support and error handling.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import asyncpg

from ..core.config import get_settings
from ..core.database import DatabaseManager, db_manager
from ..core.exceptions import DatabaseError
from .chunking.chunk import Chunk

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseIntegration:
    """
    Database integration for saving chunks and embeddings to PostgreSQL.

    This class provides methods to save market data and property listings to the database,
    along with their chunks and embeddings, using transactions for atomicity.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the DatabaseIntegration.

        Args:
            db_manager: Database manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager

    async def initialize(self) -> None:
        """Initialize the database manager if not provided."""
        if not self.db_manager:
            self.db_manager = db_manager
            self.logger.info("Database manager initialized")

    async def save_market_data_to_database(
        self, market_data: Dict[str, Any], chunks: List[Chunk], conn=None, test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Save market data and its chunks to the database.

        Args:
            market_data: Market data dictionary
            chunks: List of chunks generated from the market data
            conn: Optional database connection (for testing)
            test_mode: Whether to run in test mode (bypass DB connection)

        Returns:
            Dictionary with operation results
        """
        if not self.db_manager and not test_mode:
            await self.initialize()

        try:
            if test_mode:
                # For testing, use the mocked methods directly
                market_data_id = await self._save_market_data(conn, market_data)
                chunk_ids = await self._save_market_chunks(conn, market_data_id, chunks)

                self.logger.info(f"[TEST MODE] Saved market data with ID {market_data_id} and {len(chunk_ids)} chunks")

                return {
                    "market_data_id": market_data_id,
                    "chunks_saved": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                    "success": True,
                }
            else:
                # Start a transaction
                async with self.db_manager.get_connection() as conn:
                    # Begin transaction explicitly for better control
                    async with conn.transaction():
                        try:
                            # Insert market data
                            market_data_id = await self._save_market_data(conn, market_data)
                            self.logger.debug(f"Saved market data with ID {market_data_id}")

                            # Insert chunks
                            chunk_ids = await self._save_market_chunks(conn, market_data_id, chunks)
                            self.logger.debug(f"Saved {len(chunk_ids)} market chunks")

                            self.logger.info(
                                f"Successfully completed transaction: saved market data with ID {market_data_id} "
                                f"and {len(chunk_ids)} chunks"
                            )

                            return {
                                "market_data_id": market_data_id,
                                "chunks_saved": len(chunk_ids),
                                "chunk_ids": chunk_ids,
                                "success": True,
                            }
                        except Exception as inner_e:
                            # Log the specific error but let the outer transaction handle the rollback
                            self.logger.error(f"Error during transaction: {inner_e}")
                            raise inner_e

        except Exception as e:
            # Use the error handler to classify and log the error
            region_id_for_log = market_data.get("region_id") or "unknown"
            self.logger.error(f"Failed to save market data {region_id_for_log}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def save_property_to_database(
        self, property_data: Dict[str, Any], chunks: List[Chunk], conn=None, test_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Save property listing and its chunks to the database.

        Args:
            property_data: Property listing dictionary
            chunks: List of chunks generated from the property listing
            conn: Optional database connection (for testing)
            test_mode: Whether to run in test mode (bypass DB connection)

        Returns:
            Dictionary with operation results
        """
        if not self.db_manager and not test_mode:
            await self.initialize()

        try:
            if test_mode:
                # For testing, use the mocked methods directly
                property_id = await self._save_property_listing(conn, property_data)
                chunk_ids = await self._save_property_chunks(conn, property_id, chunks)

                self.logger.info(
                    f"[TEST MODE] Saved property listing with ID {property_id} and {len(chunk_ids)} chunks"
                )

                return {
                    "property_id": property_id,
                    "chunks_saved": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                    "success": True,
                }
            else:
                async with self.db_manager.get_connection() as conn:
                    # Begin transaction explicitly for better control
                    async with conn.transaction():
                        try:
                            # Insert property listing
                            property_id = await self._save_property_listing(conn, property_data)
                            self.logger.debug(f"Saved property listing with ID {property_id}")

                            # Insert chunks
                            chunk_ids = await self._save_property_chunks(conn, property_id, chunks)
                            self.logger.debug(f"Saved {len(chunk_ids)} property chunks")

                            self.logger.info(
                                f"Successfully saved property listing with ID {property_id} "
                                f"and {len(chunk_ids)} chunks"
                            )

                            return {
                                "property_id": property_id,
                                "chunks_saved": len(chunk_ids),
                                "chunk_ids": chunk_ids,
                                "success": True,
                            }
                        except Exception as inner_e:
                            # Log the specific error but let the outer transaction handle the rollback
                            self.logger.error(f"Error during transaction: {inner_e}")
                            raise inner_e

        except Exception as e:
            # Fixed: proper logging without extra kwargs
            property_id_for_log = property_data.get("id") or property_data.get("formattedAddress") or "unknown"
            self.logger.error(f"Failed to save property listing {property_id_for_log}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _save_market_data(self, conn, market_data: Dict[str, Any]) -> UUID:
        """
        Save market data to the database.

        Args:
            conn: Database connection
            market_data: Market data dictionary

        Returns:
            UUID of the inserted market data record
        """
        # Extract market data fields
        location = market_data.get("location")
        date_str = market_data.get("date")
        date = datetime.fromisoformat(date_str) if date_str else None
        median_price = market_data.get("median_price")
        inventory_count = market_data.get("inventory_count")
        sales_volume = market_data.get("sales_volume")
        new_listings = market_data.get("new_listings")
        days_on_market = market_data.get("days_on_market")
        months_supply = market_data.get("months_supply")
        price_per_sqft = market_data.get("price_per_sqft")
        source = market_data.get("source")
        region_type = market_data.get("region_type")
        region_id = market_data.get("region_id")
        duration = market_data.get("duration")
        last_updated = market_data.get("last_updated")
        city = market_data.get("city")
        state = market_data.get("state")

        try:
            # Insert market data
            query = """
            INSERT INTO market_data (
                location, date, median_price, inventory_count, sales_volume,
                new_listings, days_on_market, months_supply, price_per_sqft,
                source, region_type, region_id, duration, last_updated, city, state
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING id
            """

            result = await conn.fetchval(
                query,
                location,
                date,
                median_price,
                inventory_count,
                sales_volume,
                new_listings,
                days_on_market,
                months_supply,
                price_per_sqft,
                source,
                region_type,
                region_id,
                duration,
                datetime.fromisoformat(last_updated.replace("Z", "+00:00")) if last_updated else None,
                city,
                state,
            )

            return result

        except Exception as e:
            # Handle specific Neon DB errors
            if "duplicate key value violates unique constraint" in str(e):
                self.logger.warning(f"Market data with region_id {region_id} already exists: {e}")

                # Try to get the existing record ID
                try:
                    query = "SELECT id FROM market_data WHERE region_id = $1 AND date = $2 LIMIT 1"
                    existing_id = await conn.fetchval(query, region_id, date)

                    if existing_id:
                        self.logger.info(f"Using existing market data record: {existing_id}")
                        return existing_id
                except Exception as inner_e:
                    self.logger.error(f"Failed to get existing market data record: {inner_e}")

            # Re-raise the exception for the caller to handle
            raise

    async def _save_property_listing(self, conn, property_data: Dict[str, Any]) -> UUID:
        """
        Save property listing to the database.

        Args:
            conn: Database connection
            property_data: Property listing dictionary

        Returns:
            UUID of the inserted property listing record
        """
        # Generate property ID from formatted address if 'id' field is missing
        property_string_id = property_data.get("id") or property_data.get("property_id")
        if not property_string_id:
            # Use formatted address as fallback ID
            formatted_address = property_data.get("formattedAddress")
            if formatted_address:
                property_string_id = formatted_address
            else:
                raise ValueError("Property must have either 'id' or 'formattedAddress' field")

        # Convert to UUID (keep as string for asyncpg)
        property_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(property_string_id)))

        # Extract address components
        formatted_address = property_data.get("formattedAddress")
        if not formatted_address:
            raise ValueError("Formatted address is required")

        # Get listing type from the most recent history event
        listing_type = property_data.get("listingType")
        history = property_data.get("history", {})
        if history:
            latest_event_date = max(history.keys())
            latest_event = history[latest_event_date]
            listing_type = latest_event.get("event", listing_type)

        try:
            # Insert property listing
            query = """
            INSERT INTO property_listings (
                id, formatted_address, address_line1, address_line2,
                city, state, zip_code, county, latitude, longitude,
                property_type, bedrooms, bathrooms, square_footage, lot_size,
                year_built, status, price, listing_type, listed_date,
                removed_date, created_date, last_seen_date, days_on_market, mls_name, mls_number,
                listing_agent, listing_office, history
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29
            )
            RETURNING id
            """

            result = await conn.fetchval(
                query,
                property_id,
                property_data.get("formattedAddress"),
                property_data.get("addressLine1"),
                property_data.get("addressLine2"),
                property_data.get("city"),
                property_data.get("state"),
                property_data.get("zipCode"),
                property_data.get("county"),
                property_data.get("latitude"),
                property_data.get("longitude"),
                property_data.get("propertyType"),
                property_data.get("bedrooms"),
                property_data.get("bathrooms"),
                property_data.get("squareFootage"),
                property_data.get("lotSize"),
                property_data.get("yearBuilt"),
                property_data.get("status"),
                property_data.get("price"),
                listing_type,
                (
                    datetime.fromisoformat(property_data.get("listedDate").replace("Z", "+00:00"))
                    if property_data.get("listedDate")
                    else None
                ),
                (
                    datetime.fromisoformat(property_data.get("removedDate").replace("Z", "+00:00"))
                    if property_data.get("removedDate")
                    else None
                ),
                (
                    datetime.fromisoformat(property_data.get("createdDate").replace("Z", "+00:00"))
                    if property_data.get("createdDate")
                    else None
                ),
                (
                    datetime.fromisoformat(property_data.get("lastSeenDate").replace("Z", "+00:00"))
                    if property_data.get("lastSeenDate")
                    else None
                ),
                property_data.get("daysOnMarket"),
                property_data.get("mlsName"),
                property_data.get("mlsNumber"),
                json.dumps(property_data.get("listingAgent", {})),
                json.dumps(property_data.get("listingOffice", {})),
                json.dumps(property_data.get("history", {})),
            )

            return result

        except Exception as e:
            # Handle specific Neon DB errors
            if "duplicate key value violates unique constraint" in str(e):
                self.logger.warning(f"Property listing with property_id {property_string_id} already exists: {e}")

                # Try to get the existing record ID
                try:
                    query = "SELECT id FROM property_listings WHERE id = $1 LIMIT 1"
                    existing_id = await conn.fetchval(query, str(property_id))

                    if existing_id:
                        self.logger.info(f"Using existing property listing record: {existing_id}")
                        return existing_id
                except Exception as inner_e:
                    self.logger.error(f"Failed to get existing property listing record: {inner_e}")
                    # Don't use bare raise here - re-raise the original exception
                    raise e

            # Re-raise the exception for the caller to handle
            raise

    async def _save_market_chunks(self, conn, market_data_id: UUID, chunks: List[Chunk]) -> List[UUID]:
        """
        Save market data chunks to the database with all enhanced fields.
        """
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Extract additional fields from the original market data
            market_region = None
            data_source = None
            report_date = datetime.now().date()  # Current date of ingestion

            # Try to get market_region from chunk metadata or content
            if chunk.metadata:
                # Extract from parent context or derive from content
                parent_context = chunk.metadata.get("parent_context", "")
                if parent_context:
                    market_region = str(parent_context)

                # Get data source
                data_source = chunk.metadata.get("source", "unknown")

            # Extract entities and entity types from metadata
            extracted_entities = chunk.metadata.get("extracted_entities", {})
            entity_types = chunk.metadata.get("entity_types", [])

            # Get chunk_type and semantic_score from metadata
            chunk_type = chunk.metadata.get("chunk_type", "unknown")
            semantic_score = chunk.metadata.get("semantic_score", 0.0)

            query = """
            INSERT INTO market_chunks_enhanced (
                market_data_id, content, chunk_index, token_count, embedding, 
                metadata, market_region, data_source, report_date, 
                extracted_entities, entity_types, chunk_type, semantic_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """

            chunk_id = await conn.fetchval(
                query,
                market_data_id,
                chunk.content,
                i,
                chunk.metadata.get("token_count", 0),
                str(chunk.embedding) if chunk.embedding else None,
                json.dumps(chunk.metadata or {}),
                market_region,
                data_source,
                report_date,
                json.dumps(extracted_entities),
                entity_types,
                chunk_type,
                float(semantic_score),
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    async def _save_property_chunks(self, conn, property_id: str, chunks: List[Chunk]) -> List[UUID]:
        """
        Save property listing chunks to the database with all enhanced fields.

        Args:
            conn: Database connection
            property_id: ID of the parent property listing record
            chunks: List of chunks to save

        Returns:
            List of UUIDs of the inserted chunk records
        """
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Initialize all variables with defaults
            property_region = None
            data_source = "property_listing"
            listing_date = None
            extracted_entities = {}
            entity_types = []
            chunk_type = "unknown"
            semantic_score = 0.0
            content_density = 0.0
            semantic_level = "unknown"

            # Extract fields from chunk metadata if available
            if chunk.metadata:
                # Extract from parent context or metadata
                parent_context = chunk.metadata.get("parent_context", "")
                if parent_context:
                    property_region == str(parent_context)

                # Get data source
                data_source == chunk.metadata.get("source", "property_listing")

                # Get listing date if available
                if "extracted_entities" in chunk.metadata:
                    extracted_entities = chunk.metadata.get("extracted_entities", {})
                    listing_date == extracted_entities.get("listed_date")

                # Get entity types from metadata
                entity_types = chunk.metadata.get("entity_types", [])

                # Get chunk_type and scores from metadata
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                semantic_score = chunk.metadata.get("semantic_score", 0.0)
                content_density = chunk.metadata.get("content_density", 0.0)

                # Get semantic level
                semantic_level = chunk.metadata.get("semantic_level", "unknown")

            query = """
            INSERT INTO property_chunks_enhanced (
                property_listing_id, content, chunk_index, token_count, embedding, 
                metadata, chunk_type, semantic_score, content_density,
                extracted_entities, entity_types, semantic_level
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
            """

            chunk_id = await conn.fetchval(
                query,
                property_id,
                chunk.content,
                i,
                chunk.metadata.get("token_count", 0) if chunk.metadata else 0,
                str(chunk.embedding) if chunk.embedding else None,
                json.dumps(chunk.metadata or {}),
                chunk_type,
                float(semantic_score),
                float(content_density),
                json.dumps(extracted_entities),
                entity_types,
                semantic_level,
            )
            chunk_ids.append(chunk_id)
        return chunk_ids
