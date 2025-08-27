"""
Enhanced Ingestion Pipeline with JSON Chunking for TrackRealties AI Platform.

This module provides a comprehensive pipeline that handles all aspects of data ingestion,
from chunking JSON data semantically to generating embeddings and building a knowledge graph.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.config import Settings, get_settings
# Add this import at the top
from ..data.embedding.optimized_embedder import (NeonDBEmbeddingManager,
                                                 OptimizedEmbeddingPipeline)
from .chunking.chunk import Chunk
from .chunking.json_chunker import JSONChunker
from .database_integration import DatabaseIntegration
from .embedding.openai_embedder import OpenAIEmbedder
from .error_logging import log_error
# UPDATED: Import the enhanced graph builder instead of the old one
from .graph.enhanced_graph_builder import EnhancedGraphBuilder
from .graph.enhanced_relationship_manager import EnhancedRelationshipManager

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    total: int
    processed: int
    failed: int
    chunks_created: int
    embeddings_generated: int
    graph_nodes_created: int
    graph_relationships_created: int = 0  # NEW: Track relationships
    enhanced_features: Dict[str, int] = field(default_factory=dict)  # NEW: Track enhanced features
    errors: List[str] = field(default_factory=list)


class EnhancedIngestionPipeline:
    """
    Enhanced ingestion pipeline with JSON chunking, embedding generation, and knowledge graph integration.

    This class orchestrates the entire ingestion process, from chunking JSON data to generating
    embeddings and building a knowledge graph.
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        skip_embeddings: bool = False,
        skip_graph: bool = False,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the EnhancedIngestionPipeline.

        Args:
            batch_size: Number of records to process in a batch
            skip_embeddings: Whether to skip embedding generation
            skip_graph: Whether to skip graph building
            settings: Application settings
        """
        # Use optimized embedding pipeline
        self.optimized_embedder = OptimizedEmbeddingPipeline(
            batch_size=batch_size or 50, max_concurrent_batches=3, enable_cache_warming=True
        )

        # Use enhanced NeonDB manager
        self.neon_manager = NeonDBEmbeddingManager(os.environ.get("DATABASE_URL"))
        self.logger = logging.getLogger(__name__)
        self.settings = settings or get_settings()
        self.batch_size = batch_size or self.settings.ingestion_batch_size
        self.skip_embeddings = skip_embeddings
        self.skip_graph = skip_graph

        # Initialize components
        self.chunker = None
        self.embedder = None
        self.db_integration = None
        # UPDATED: Use enhanced graph builder and relationship manager
        self.graph_builder = None
        self.relationship_manager = None

        # Track initialization status
        self.initialized = False

    async def initialize(self, dry_run: bool = False) -> None:
        """Initialize all components of the pipeline."""
        try:
            # Initialize JSON chunker
            self.chunker = JSONChunker(
                max_chunk_size=self.settings.max_chunk_size, chunk_overlap=self.settings.chunk_overlap
            )
            self.logger.info("Initialized JSON chunker")

            if not dry_run:
                if not self.skip_embeddings:
                    self.embedder = OpenAIEmbedder(
                        model=self.settings.embedding_model,
                        dimensions=self.settings.embedding_dimensions,
                        batch_size=self.settings.embedding_batch_size,
                        use_cache=True,
                        api_key=self.settings.embedding_api_key or self.settings.llm_api_key,
                    )
                    await self.embedder.initialize()
                    self.logger.info(f"Initialized embedder with model {self.settings.embedding_model}")

                # Initialize database integration
                self.db_integration = DatabaseIntegration()
                await self.db_integration.initialize()
                self.logger.info("Initialized database integration")

                # UPDATED: Initialize enhanced graph builder and relationship manager
                if not self.skip_graph:
                    self.graph_builder = EnhancedGraphBuilder()
                    await self.graph_builder.initialize()
                    self.logger.info("Initialized enhanced graph builder")

                    self.relationship_manager = EnhancedRelationshipManager()
                    await self.relationship_manager.initialize()
                    self.logger.info("Initialized enhanced relationship manager")

            self.initialized = True
            self.logger.info("Enhanced ingestion pipeline initialized successfully")

            # Initialize optimized components
            await self.optimized_embedder.initialize()
            await self.neon_manager.initialize()

            # Optimize database for vector search
            await self.neon_manager.optimize_vector_search_performance()
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced ingestion pipeline: {e}")
            raise

    async def ingest_market_data(self, source: str, data: List[Dict[str, Any]]) -> IngestionResult:
        """
        Ingest market data with proper chunking, embedding, and enhanced graph building.

        Args:
            source: Source of the data (e.g., 'zillow', 'redfin')
            data: List of market data records

        Returns:
            IngestionResult with details of the ingestion
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        result = IngestionResult(
            total=len(data),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0,
            graph_relationships_created=0,
            enhanced_features={
                "price_segmentation": 0,
                "geographic_hierarchy": 0,
                "market_context": 0,
                "agent_performance": 0,
                "property_type": 0,
            },
        )

        for i in range(0, len(data), self.batch_size):
            batch = data[i: i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1} ({len(batch)} records)")

            for record in batch:
                try:
                    # Chunk the record
                    chunks = self.chunker.chunk_json(record, "market")
                    result.chunks_created += len(chunks)

                    # Generate embeddings
                    if not self.skip_embeddings and self.embedder:
                        chunk_texts = [chunk.content for chunk in chunks]
                        embeddings, token_counts = await self.embedder.generate_embeddings_batch(chunk_texts)
                        # Update chunks with embeddings
                        for chunk, embedding in zip(chunks, embeddings):
                            chunk.embedding = embedding
                        result.embeddings_generated += len(embeddings)

                    # Save to database
                    db_result = await self.db_integration.save_market_data_to_database(record, chunks)

                    if db_result.get("success"):
                        result.processed += 1

                        # UPDATED: Add to enhanced knowledge graph
                        if not self.skip_graph and self.graph_builder and self.relationship_manager:
                            # Use enhanced graph building with market context
                            graph_result = await self._process_market_data_with_enhanced_graph(
                                record, data  # Pass all data for market context
                            )

                            if graph_result.get("success"):
                                result.graph_nodes_created += graph_result.get("nodes_created", 0)
                                result.graph_relationships_created += graph_result.get("relationships_created", 0)

                                # Track enhanced features
                                if graph_result.get("geographic_hierarchy"):
                                    result.enhanced_features["geographic_hierarchy"] += 1
                                if graph_result.get("market_context"):
                                    result.enhanced_features["market_context"] += 1
                            else:
                                # Log validation issues but don't fail the entire process
                                if "validation" in graph_result.get("error", "").lower():
                                    self.logger.warning(
                                        f"Market data price validation skipped: {graph_result.get('error')}"
                                    )
                                else:
                                    self.logger.error(f"Market graph processing failed: {graph_result.get('error')}")
                    else:
                        result.failed += 1
                        result.errors.append(f"Failed to save record: {db_result.get('error')}")

                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"Error processing record: {str(e)}")
                    self.logger.error(f"Error processing record from {source}: {str(e)}")

        self.logger.info(
            f"Market data ingestion complete: {result.processed}/{result.total} processed, "
            f"{result.chunks_created} chunks, {result.embeddings_generated} embeddings, "
            f"{result.graph_nodes_created} nodes, {result.graph_relationships_created} relationships"
        )

        return result

    async def ingest_property_listings(self, source: str, data: List[Dict[str, Any]]) -> IngestionResult:
        """
        Ingest property listings with proper chunking, embedding, and enhanced graph building.

        Args:
            source: Source of the data (e.g., 'mls', 'zillow')
            data: List of property listing records

        Returns:
            IngestionResult with details of the ingestion
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        result = IngestionResult(
            total=len(data),
            processed=0,
            failed=0,
            chunks_created=0,
            embeddings_generated=0,
            graph_nodes_created=0,
            graph_relationships_created=0,
            enhanced_features={
                "price_segmentation": 0,
                "geographic_hierarchy": 0,
                "market_context": 0,
                "agent_performance": 0,
                "property_type": 0,
            },
        )

        for i in range(0, len(data), self.batch_size):
            batch = data[i: i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1} ({len(batch)} records)")

            for record in batch:
                try:
                    # Chunk the record
                    chunks = self.chunker.chunk_json(record, "property")
                    result.chunks_created += len(chunks)

                    # Generate embeddings
                    if not self.skip_embeddings and self.embedder:
                        chunk_texts = [chunk.content for chunk in chunks]
                        embeddings, token_counts = await self.embedder.generate_embeddings_batch(chunk_texts)
                        # Update chunks with embeddings
                        for chunk, embedding in zip(chunks, embeddings):
                            chunk.embedding = embedding
                        result.embeddings_generated += len(embeddings)

                    # Save to database
                    db_result = await self.db_integration.save_property_to_database(record, chunks)

                    if db_result.get("success"):
                        result.processed += 1

                        # UPDATED: Add to enhanced knowledge graph
                        if not self.skip_graph and self.graph_builder and self.relationship_manager:
                            # Use enhanced graph building with property context
                            graph_result = await self._process_property_with_enhanced_graph(
                                record, data  # Pass all data for comparables
                            )

                            if graph_result.get("success"):
                                result.graph_nodes_created += graph_result.get("nodes_created", 0)
                                result.graph_relationships_created += graph_result.get("relationships_created", 0)

                                # Track enhanced features
                                enhancements = graph_result.get("enhancements", {})
                                for feature, enabled in enhancements.items():
                                    if enabled and feature in result.enhanced_features:
                                        result.enhanced_features[feature] += 1
                            else:
                                # Log validation issues but don't fail the entire process
                                if "validation" in graph_result.get("error", "").lower():
                                    self.logger.warning(
                                        f"Property price validation skipped: {graph_result.get('error')}"
                                    )
                                else:
                                    self.logger.error(f"Graph processing failed: {graph_result.get('error')}")
                    else:
                        result.failed += 1
                        result.errors.append(f"Failed to save record: {db_result.get('error')}")

                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"Error processing record: {str(e)}")
                    self.logger.error(f"Error processing record from {source}: {str(e)}")

        self.logger.info(
            f"Property listings ingestion complete: {result.processed}/{result.total} processed, "
            f"{result.chunks_created} chunks, {result.embeddings_generated} embeddings, "
            f"{result.graph_nodes_created} nodes, {result.graph_relationships_created} relationships"
        )

        return result

    async def _process_market_data_with_enhanced_graph(
        self, market_record: Dict[str, Any], all_market_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process market data with enhanced graph features."""
        try:
            # Create basic market data node using enhanced builder
            basic_result = await self.graph_builder.add_market_data_to_graph(market_record)

            if not basic_result.get("success"):
                return basic_result

            # Build geographic hierarchy
            geo_result = await self.graph_builder._build_geographic_hierarchy(market_record)

            # Establish market context relationships if there are other market records
            context_relationships = 0
            if len(all_market_data) > 1:
                # Note: Market-to-market relationships are handled differently
                # establish_market_context_relationships is for property-to-market relationships
                # For market data processing, we skip relationship creation here
                context_relationships = 0

                # TODO: Implement market-to-market relationship logic if needed
                # This would require a different function for market data relationships

            return {
                "success": True,
                "nodes_created": basic_result.get("nodes_created", 0) + geo_result.get("nodes_created", 0),
                "relationships_created": (
                    basic_result.get("relationships_created", 0) + geo_result.get("relationships_created", 0) + context_relationships),
                "geographic_hierarchy": geo_result.get("nodes_created", 0) > 0, "market_context": context_relationships > 0,
            }

        except Exception as e:
            self.logger.error(f"Error in enhanced market data graph processing: {e}")
            return {"success": False, "error": str(e), "nodes_created": 0, "relationships_created": 0}

    async def _process_property_with_enhanced_graph(
        self, property_record: Dict[str, Any], all_property_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process property with enhanced graph features."""
        try:
            # Use the integrated graph building method
            # Since we don't have market data in this context, we'll create a mock market data
            # In a real scenario, you might want to load market data from the database
            mock_market_data = [
                {
                    "location": f"{property_record.get('city', 'Unknown')}, {property_record.get('state', 'Unknown')}",
                    "median_price": 400000.0,  # Default median
                    "inventory_count": 1000.0,
                    "days_on_market": 45.0,
                    "region_id": f"mock_{property_record.get('county', 'unknown')}",
                    "city": property_record.get("city"),
                    "state": property_record.get("state"),
                    "county": property_record.get("county"),
                }
            ]

            graph_result = await self.graph_builder.build_integrated_graph(property_record, mock_market_data)

            if graph_result.get("success"):
                # Establish agent performance relationships if we have market context
                agent_result = await self.relationship_manager.establish_agent_performance_relationships(
                    property_record, mock_market_data
                )

                # Find comparable properties (limit to 3 for performance)
                comparable_properties = [
                    p
                    for p in all_property_data[:10]  # Limit search space
                    if (
                        p.get("id") != property_record.get("id") and p.get("city") == property_record.get("city") and p.get("propertyType") == property_record.get("propertyType")
                    )
                ][:3]

                comp_result = {"relationships_created": 0}
                if comparable_properties:
                    comp_result = await self.relationship_manager.establish_comparable_property_relationships(
                        property_record, comparable_properties
                    )

                # Aggregate results
                total_relationships = (
                    graph_result.get("relationships", 0) + agent_result.get("relationships_created", 0) + comp_result.get("relationships_created", 0)
                )

                return {
                    "success": True,
                    "nodes_created": graph_result.get("property_nodes", 0) + graph_result.get("location_nodes", 0) + graph_result.get("price_segments", 0) + graph_result.get("property_types", 0),
                    "relationships_created": total_relationships,
                    "enhancements": graph_result.get("enhancements", {}),
                }
            else:
                return graph_result

        except Exception as e:
            self.logger.error(f"Error in enhanced property graph processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "nodes_created": 0,
                "relationships_created": 0,
                "enhancements": {},
            }

    # Keep existing validation methods unchanged
    async def validate_market_data(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate market data without saving to database.

        Args:
            source: Source of the data
            data: List of market data records

        Returns:
            Validation results including errors and warnings
        """
        validation_results = {"total": len(data), "valid": 0, "invalid": 0, "errors": [], "warnings": []}

        required_fields = self.settings.VALIDATION_REQUIRED_FIELDS_MARKET

        for i, record in enumerate(data):
            errors = []

            # Check required fields
            for field_name in required_fields:
                if field_name not in record or record[field_name] is None:
                    errors.append(f"Missing required field: {field_name}")

            # Validate data types and formats
            if "date" in record:
                try:
                    datetime.fromisoformat(record["date"].replace("Z", "+00:00"))
                except Exception:
                    errors.append("Invalid date format")

            if "median_price" in record:
                try:
                    float(record["median_price"])
                except Exception:
                    errors.append("median_price must be numeric")

            if errors:
                validation_results["invalid"] += 1
                validation_results["errors"].append({"record_index": i, "errors": errors})
            else:
                validation_results["valid"] += 1

        return validation_results

    async def validate_property_listings(self, source: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate property listings without saving to database.

        Args:
            source: Source of the data
            data: List of property listing records

        Returns:
            Validation results including errors and warnings
        """
        validation_results = {"total": len(data), "valid": 0, "invalid": 0, "errors": [], "warnings": []}

        required_fields = self.settings.VALIDATION_REQUIRED_FIELDS_PROPERTY

        for i, record in enumerate(data):
            errors = []

            # Check required fields
            for field_name in required_fields:
                if field_name not in record or record[field_name] is None:
                    errors.append(f"Missing required field: {field_name}")

            # Validate data types and formats
            if "price" in record:
                try:
                    float(record["price"])
                except Exception:
                    errors.append("price must be numeric")

            if "bedrooms" in record:
                try:
                    int(record["bedrooms"])
                except Exception:
                    errors.append("bedrooms must be integer")

            if errors:
                validation_results["invalid"] += 1
                validation_results["errors"].append({"record_index": i, "errors": errors})
            else:
                validation_results["valid"] += 1

                # Add warnings for data quality
                if "description" in record and len(record.get("description", "")) < 50:
                    validation_results["warnings"].append(
                        {"record_index": i, "warning": "Short description may affect search quality"}
                    )

        return validation_results

    async def close(self) -> None:
        """Close all pipeline components and clean up resources."""
        if self.embedder:
            await self.embedder.close()

        if self.relationship_manager and hasattr(self.relationship_manager, "close"):
            await self.relationship_manager.close()

        self.logger.info("Enhanced ingestion pipeline closed")
