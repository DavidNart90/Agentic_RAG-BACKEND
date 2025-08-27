"""
Relationship manager for establishing connections between graph entities.

This module provides functionality to establish and manage relationships
between entities in the knowledge graph.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError:
    AsyncGraphDatabase = None
    AsyncDriver = None

from ...core.config import get_settings
from .formatters import format_relationship_properties

logger = logging.getLogger(__name__)
settings = get_settings()


class RelationshipManager:
    """
    Manages relationships between entities in the knowledge graph.

    This class provides methods to establish and manage relationships
    between entities in the Neo4j graph database.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize the RelationshipManager.

        Args:
            uri: Neo4j URI
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.logger = logging.getLogger(__name__)
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.driver = None

    async def initialize(self) -> None:
        """Initialize the Neo4j driver."""
        try:
            if AsyncGraphDatabase is None:
                self.logger.error("Failed to import Neo4j package. Please install it with 'pip install neo4j'")
                raise ImportError("Failed to import Neo4j package. Please install it with 'pip install neo4j'")

            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))

            # Test connection
            await self._test_connection()

            self.logger.info(f"Initialized Neo4j connection to {self.uri}")

        except ImportError:
            self.logger.error("Failed to import Neo4j package. Please install it with 'pip install neo4j'")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j connection: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test the Neo4j connection."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                if record and record["test"] == 1:
                    self.logger.debug("Neo4j connection test successful")
                else:
                    raise ValueError("Neo4j connection test failed")
        except Exception as e:
            self.logger.error(f"Neo4j connection test failed: {e}")
            raise

    async def create_relationship(
        self,
        from_node: Dict[str, str],
        to_node: Dict[str, str],
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a relationship between two nodes.

        Args:
            from_node: Dictionary with from node label and ID field/value
            to_node: Dictionary with to node label and ID field/value
            relationship_type: Type of relationship to create
            properties: Optional properties for the relationship

        Returns:
            True if relationship was created, False otherwise
        """
        if not self.driver:
            await self.initialize()

        try:
            # Extract node information
            from_label = from_node["label"]
            from_id_field = from_node["id_field"]
            from_id_value = from_node["id_value"]

            to_label = to_node["label"]
            to_id_field = to_node["id_field"]
            to_id_value = to_node["id_value"]

            # Format properties for better readability
            formatted_props = format_relationship_properties(properties or {})

            # Add timestamp if not present
            if "created_at" not in formatted_props:
                formatted_props["created_at"] = datetime.now(timezone.utc).isoformat()

            # Create relationship
            query = f"""
            MATCH (a:{from_label} {{{from_id_field}: $from_id_value}})
            MATCH (b:{to_label} {{{to_id_field}: $to_id_value}})
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r += $properties
            RETURN r
            """

            parameters = {"from_id_value": from_id_value, "to_id_value": to_id_value, "properties": formatted_props}

            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, parameters)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created relationship: ({from_label})-[{relationship_type}]->({to_label})")
                    return True
                else:
                    self.logger.warning(
                        f"Failed to create relationship: ({from_label})-[{relationship_type}]->({to_label})"
                    )
                    return False

        except Exception as e:
            self.logger.error(f"Error creating relationship: {e}")
            return False

    async def establish_property_relationships(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Establish relationships for a property entity.

        This method creates relationships between a property and related entities
        such as location, agent, features, etc.

        Args:
            property_data: Property data

        Returns:
            Dictionary with relationship creation results
        """
        if not self.driver:
            await self.initialize()

        try:
            property_id = property_data.get("id")
            if not property_id:
                raise ValueError("Property ID is required")

            results = {"property_id": property_id, "relationships_created": 0, "success": True}

            # Create location relationship
            if property_data.get("city") and property_data.get("state"):
                city = property_data["city"]
                state = property_data["state"]
                location_id = f"{city.lower().replace(' ', '_')}_{state.lower()}"

                location_rel = await self.create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "Location", "id_field": "location_id", "id_value": location_id},
                    relationship_type="LOCATED_IN",
                    properties={},
                )

                if location_rel:
                    results["relationships_created"] += 1

            # Create agent relationship
            if property_data.get("listingAgent", {}).get("name"):
                agent_name = property_data["listingAgent"]["name"]
                agent_email = property_data["listingAgent"].get("email")

                if agent_email:
                    agent_id = agent_email.lower()
                else:
                    # Create a sanitized ID from the name
                    agent_id = re.sub(r"[^a-z0-9]", "_", agent_name.lower())

                agent_rel = await self.create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_id},
                    relationship_type="LISTED_BY",
                    properties={},
                )

                if agent_rel:
                    results["relationships_created"] += 1

                # Create office relationship if available
                if property_data.get("listingOffice", {}).get("name"):
                    office_name = property_data["listingOffice"]["name"]
                    office_id = re.sub(r"[^a-z0-9]", "_", office_name.lower())

                    office_rel = await self.create_relationship(
                        from_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_id},
                        to_node={"label": "Office", "id_field": "office_id", "id_value": office_id},
                        relationship_type="WORKS_FOR",
                        properties={},
                    )

                    if office_rel:
                        results["relationships_created"] += 1

            # Create history relationships
            history = property_data.get("history", {})
            for date, event in history.items():
                event_type = event.get("event")
                if not event_type:
                    event_type = event.get("listingType", "Unknown")

                # Create event ID
                event_id = f"{property_id}_{date}_{event_type}"

                history_rel = await self.create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "HistoryEvent", "id_field": "event_id", "id_value": event_id},
                    relationship_type="HAS_HISTORY",
                    properties={"date": date, "event_type": event_type, "price": event.get("price")},
                )

                if history_rel:
                    results["relationships_created"] += 1

            # Create feature relationships
            features = []
            if property_data.get("bedrooms"):
                features.append(f"bedrooms_{property_data['bedrooms']}")
            if property_data.get("bathrooms"):
                features.append(f"bathrooms_{property_data['bathrooms']}")
            if property_data.get("propertyType"):
                features.append(property_data["propertyType"].lower().replace(" ", "_"))

            for feature in features:
                feature_rel = await self.create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "Feature", "id_field": "name", "id_value": feature},
                    relationship_type="HAS_FEATURE",
                    properties={},
                )

                if feature_rel:
                    results["relationships_created"] += 1

            return results

        except Exception as e:
            self.logger.error(f"Error establishing property relationships: {e}")
            return {"property_id": property_data.get("id", "unknown"), "error": str(e), "success": False}

    async def establish_market_relationships(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Establish relationships for a market entity.

        This method creates relationships between a market and related entities
        such as regions, time periods, metrics, etc.

        Args:
            market_data: Market data

        Returns:
            Dictionary with relationship creation results
        """
        if not self.driver:
            await self.initialize()

        try:
            region_id = market_data.get("region_id")
            if not region_id:
                raise ValueError("Region ID is required")

            date = market_data.get("date")

            # Create a unique market data ID
            market_data_id = f"{region_id}_{date}"

            results = {
                "market_data_id": market_data_id,
                "region_id": region_id,
                "relationships_created": 0,
                "success": True,
            }

            # Create region relationship
            region_rel = await self.create_relationship(
                from_node={"label": "Region", "id_field": "region_id", "id_value": region_id},
                to_node={"label": "MarketData", "id_field": "market_data_id", "id_value": market_data_id},
                relationship_type="HAS_MARKET_DATA",
                properties={"date": date},
            )

            if region_rel:
                results["relationships_created"] += 1

            # Create location relationship if city and state are available
            if market_data.get("city") and market_data.get("state"):
                city = market_data["city"]
                state = market_data["state"]
                location_id = f"{city.lower().replace(' ', '_')}_{state.lower()}"

                location_rel = await self.create_relationship(
                    from_node={"label": "MarketData", "id_field": "market_data_id", "id_value": market_data_id},
                    to_node={"label": "Location", "id_field": "location_id", "id_value": location_id},
                    relationship_type="FOR_LOCATION",
                    properties={},
                )

                if location_rel:
                    results["relationships_created"] += 1

            # Create metric relationships for important metrics
            important_metrics = [
                "median_price",
                "inventory_count",
                "sales_volume",
                "days_on_market",
                "months_supply",
                "price_per_sqft",
            ]

            for metric_name in important_metrics:
                if metric_name in market_data and market_data[metric_name] is not None:
                    metric_value = market_data[metric_name]
                    metric_id = f"{metric_name}_{market_data_id}"

                    metric_rel = await self.create_relationship(
                        from_node={"label": "MarketData", "id_field": "market_data_id", "id_value": market_data_id},
                        to_node={"label": "Metric", "id_field": "metric_id", "id_value": metric_id},
                        relationship_type="HAS_METRIC",
                        properties={"name": metric_name, "value": metric_value},
                    )

                    if metric_rel:
                        results["relationships_created"] += 1

            return results

        except Exception as e:
            self.logger.error(f"Error establishing market relationships: {e}")
            return {"region_id": market_data.get("region_id", "unknown"), "error": str(e), "success": False}
