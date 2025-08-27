"""
Graph Builder for TrackRealties AI Platform.

This module provides functionality to build a knowledge graph from property listings
and market data, creating nodes and relationships in Neo4j.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError:
    AsyncGraphDatabase = None
    AsyncDriver = None

from ...core.graph import graph_manager
from .formatters import (format_agent_content, format_location_content,
                         format_market_content, format_property_content)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a knowledge graph from property listings and market data.

    This class provides methods to create nodes and relationships in Neo4j,
    representing properties, locations, agents, and market data.
    """

    def __init__(self):
        """
        Initialize the GraphBuilder.
        """
        self.logger = logging.getLogger(__name__)
        self.driver = None

    async def initialize(self) -> None:
        """Initialize the Neo4j driver and create constraints."""
        await graph_manager.initialize()
        self.driver = graph_manager._driver
        if self.driver:
            await self._create_constraints()

    async def _test_connection(self) -> None:
        """Test the Neo4j connection."""
        await graph_manager.test_connection()

    async def _create_constraints(self) -> None:
        """Create Neo4j constraints for uniqueness."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        constraints = [
            "CREATE CONSTRAINT property_id IF NOT EXISTS FOR (p:Property) REQUIRE p.property_id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE",
            "CREATE CONSTRAINT office_id IF NOT EXISTS FOR (o:Office) REQUIRE o.office_id IS UNIQUE",
            "CREATE CONSTRAINT market_data_id IF NOT EXISTS FOR (m:MarketData) REQUIRE m.market_data_id IS UNIQUE",
            "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (r:Region) REQUIRE r.region_id IS UNIQUE",
        ]

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                for constraint in constraints:
                    try:
                        await session.run(constraint)
                    except Neo4jError as e:
                        # Ignore errors about existing constraints
                        if "already exists" not in str(e):
                            raise

                self.logger.debug("Neo4j constraints created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create Neo4j constraints: {e}")
            raise

    async def add_property_to_graph(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add property listing to knowledge graph.

        Args:
            property_data: Property listing data

        Returns:
            Dictionary with graph operation results
        """
        if not self.driver:
            await self.initialize()

        try:
            # Extract property ID
            property_id = property_data.get("id")
            if not property_id:
                raise ValueError("Property ID is required")

            # Create property node
            self.logger.info(f"Creating property node for {property_id}")
            property_node = await self._create_property_node(property_data)
            if not property_node:
                raise ValueError(f"Failed to create property node for {property_id}")

            # Create location node and relationship
            self.logger.info(f"Creating location node for {property_id}")
            location_node = await self._create_location_node(property_data)
            if location_node:
                await self._create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "Location", "id_field": "location_id", "id_value": location_node["location_id"]},
                    relationship_type="LOCATED_IN",
                    properties={},
                )

            # Create agent node and relationship
            self.logger.info(f"Creating agent node for {property_id}")
            agent_node = await self._create_agent_node(property_data)
            office_node = {}
            if agent_node:
                self.logger.info(f"Creating office node for {property_id}")
                await self._create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_node["agent_id"]},
                    relationship_type="LISTED_BY",
                    properties={},
                )

                # Create office node and relationship
                office_node = await self._create_office_node(property_data)
                if office_node and agent_node:
                    await self._create_relationship(
                        from_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_node["agent_id"]},
                        to_node={"label": "Office", "id_field": "office_id", "id_value": office_node["office_id"]},
                        relationship_type="WORKS_FOR",
                        properties={},
                    )

            # Create history nodes and relationships
            self.logger.info(f"Creating history nodes for {property_id}")
            history_nodes = await self._create_history_nodes(property_data)
            for history_node in history_nodes:
                await self._create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                    to_node={"label": "HistoryEvent", "id_field": "event_id", "id_value": history_node["event_id"]},
                    relationship_type="HAS_HISTORY",
                    properties={},
                )

            return {
                "property_id": property_id,
                "nodes_created": 1 + bool(location_node) + bool(agent_node) + bool(office_node) + len(history_nodes),
                "relationships_created": bool(location_node) + bool(agent_node) + bool(office_node) + len(history_nodes),
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Failed to add property to graph: {e}")
            return {"property_id": property_data.get("id", "unknown"), "error": str(e), "success": False}

    async def add_market_data_to_graph(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add market data to knowledge graph.

        Args:
            market_data: Market data

        Returns:
            Dictionary with graph operation results
        """
        if not self.driver:
            await self.initialize()

        try:
            # Extract region ID
            region_id = market_data.get("region_id")
            if not region_id:
                raise ValueError("Region ID is required")

            # Create unique market data ID
            date = market_data.get("date")
            market_data_id = f"{region_id}_{date}"

            # Create region node
            self.logger.info(f"Creating region node for {region_id}")
            region_node = await self._create_region_node(market_data)

            # Create market data node
            self.logger.info(f"Creating market data node for {market_data_id}")
            market_node = await self._create_market_data_node(market_data, market_data_id)

            # Create relationship between region and market data
            if region_node and market_node:
                await self._create_relationship(
                    from_node={"label": "Region", "id_field": "region_id", "id_value": region_id},
                    to_node={"label": "MarketData", "id_field": "market_data_id", "id_value": market_data_id},
                    relationship_type="HAS_MARKET_DATA",
                    properties={"date": date},
                )

            # Create metrics relationships and count only succeeded ones
            self.logger.info(f"Creating metric nodes for {market_data_id}")
            metrics = [
                "median_price",
                "inventory_count",
                "sales_volume",
                "days_on_market",
                "months_supply",
                "price_per_sqft",
            ]
            metric_count = 0
            for metric in metrics:
                if metric in market_data and market_data[metric] is not None:
                    metric_id = f"{metric}_{market_data_id}"
                    metric_node = await self._create_metric_node(market_data, metric, metric_id)

                    if metric_node:
                        metric_count += 1
                        await self._create_relationship(
                            from_node={"label": "MarketData", "id_field": "market_data_id", "id_value": market_data_id},
                            to_node={"label": "Metric", "id_field": "metric_id", "id_value": metric_id},
                            relationship_type="HAS_METRIC",
                            properties={},
                        )

            return {
                "market_data_id": market_data_id,
                "region_id": region_id,
                "nodes_created": 1 + bool(region_node) + metric_count,
                "relationships_created": bool(region_node) + metric_count,
                "success": True,
            }

        except Exception as e:
            self.logger.error(f"Failed to add market data to graph: {e}")
            return {"region_id": market_data.get("region_id", "unknown"), "error": str(e), "success": False}

    async def _create_property_node(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a property node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        property_id = property_data.get("id")
        if not property_id:
            raise ValueError("Property ID is required")

        # Extract property attributes
        node_properties = {
            "property_id": property_id,
            "address": property_data.get("formattedAddress"),
            "property_type": property_data.get("propertyType"),
            "bedrooms": property_data.get("bedrooms"),
            "bathrooms": property_data.get("bathrooms"),
            "square_footage": property_data.get("squareFootage"),
            "lot_size": property_data.get("lotSize"),
            "year_built": property_data.get("yearBuilt"),
            "price": property_data.get("price"),
            "status": property_data.get("status"),
            "days_on_market": property_data.get("daysOnMarket"),
            "listing_type": property_data.get("listingType"),
            "listed_date": property_data.get("listedDate"),
            "created_date": property_data.get("createdDate"),
            "last_seen_date": property_data.get("lastSeenDate"),
            "source": property_data.get("mlsName"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Format content for better readability
        node_properties["content"] = format_property_content(property_data)

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        # Create or merge property node
        query = """
        MERGE (p:Property {property_id: $property_id})
        SET p += $properties
        RETURN p
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, property_id=property_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated property node: {property_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create property node: {property_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating property node: {e}")
            raise

    async def _create_location_node(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a location node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract location attributes
        city = property_data.get("city")
        state = property_data.get("state")

        if not city or not state:
            self.logger.warning("City and state are required for location node")
            return {}

        # Create location ID
        location_id = f"{city.lower().replace(' ', '_')}_{state.lower()}"

        node_properties = {
            "location_id": location_id,
            "city": city,
            "state": state,
            "zip_code": property_data.get("zipCode"),
            "county": property_data.get("county"),
            "latitude": property_data.get("latitude"),
            "longitude": property_data.get("longitude"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Format content for better readability
        node_properties["content"] = format_location_content(property_data)

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        # Create or merge location node
        query = """
        MERGE (l:Location {location_id: $location_id})
        SET l += $properties
        RETURN l
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, location_id=location_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated location node: {location_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create location node: {location_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating location node: {e}")
            raise

    async def _create_agent_node(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an agent node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract agent attributes
        listing_agent = property_data.get("listingAgent", {})
        if not listing_agent or not listing_agent.get("name"):
            self.logger.debug("No agent information available")
            return {}

        agent_name = listing_agent.get("name")
        agent_email = listing_agent.get("email")

        # Create agent ID
        if agent_email:
            agent_id = agent_email.lower()
        else:
            # Create a sanitized ID from the name
            agent_id = re.sub(r"[^a-z0-9]", "_", agent_name.lower())

        node_properties = {
            "agent_id": agent_id,
            "name": agent_name,
            "phone": listing_agent.get("phone"),
            "email": agent_email,
            "website": listing_agent.get("website"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Format content for better readability
        node_properties["content"] = format_agent_content(listing_agent)

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        # Create or merge agent node
        query = """
        MERGE (a:Agent {agent_id: $agent_id})
        SET a += $properties
        RETURN a
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, agent_id=agent_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated agent node: {agent_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create agent node: {agent_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating agent node: {e}")
            raise

    async def _create_office_node(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an office node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract office attributes
        listing_office = property_data.get("listingOffice", {})
        if not listing_office or not listing_office.get("name"):
            self.logger.debug("No office information available")
            return {}

        office_name = listing_office.get("name")

        # Create office ID
        office_id = re.sub(r"[^a-z0-9]", "_", office_name.lower())

        node_properties = {
            "office_id": office_id,
            "name": office_name,
            "phone": listing_office.get("phone"),
            "email": listing_office.get("email"),
            "website": listing_office.get("website"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        # Create or merge office node
        query = """
        MERGE (o:Office {office_id: $office_id})
        SET o += $properties
        RETURN o
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, office_id=office_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated office node: {office_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create office node: {office_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating office node: {e}")
            raise

    async def _create_history_nodes(self, property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create history nodes in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract history attributes
        history = property_data.get("history", {})
        if not history:
            self.logger.debug("No history information available")
            return []

        property_id = property_data.get("id")
        if not property_id:
            raise ValueError("Property ID is required for history nodes")

        history_nodes = []

        for date, event in history.items():
            event_type = event.get("event")
            if not event_type:
                event_type = event.get("listingType", "Unknown")

            # Create event ID
            event_id = f"{property_id}_{date}_{event_type}"

            node_properties = {
                "event_id": event_id,
                "property_id": property_id,
                "date": date,
                "event_type": event_type,
                "price": event.get("price"),
                "listing_type": event.get("listingType"),
                "listed_date": event.get("listedDate"),
                "removed_date": event.get("removedDate"),
                "days_on_market": event.get("daysOnMarket"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Remove None values
            node_properties = {k: v for k, v in node_properties.items() if v is not None}

            # Create history node
            query = """
            MERGE (h:HistoryEvent {event_id: $event_id})
            SET h += $properties
            RETURN h
            """

            try:
                async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                    result = await session.run(query, event_id=event_id, properties=node_properties)
                    record = await result.single()

                    if record:
                        self.logger.debug(f"Created/updated history node: {event_id}")
                        history_nodes.append(node_properties)
                    else:
                        self.logger.warning(f"Failed to create history node: {event_id}")

            except Exception as e:
                self.logger.error(f"Error creating history node: {e}")
                # Continue with other history nodes

        return history_nodes

    async def _create_region_node(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a region node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract region attributes
        region_id = market_data.get("region_id")
        location = market_data.get("location")
        region_type = market_data.get("region_type")

        if not region_id or not location or not region_type:
            self.logger.warning("Region ID, location, and type are required for region node")
            return {}

        node_properties = {
            "region_id": region_id,
            "name": location,
            "type": region_type,
            "city": market_data.get("city"),
            "state": market_data.get("state"),
            "county": market_data.get("county"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        # Create or merge region node
        query = """
        MERGE (r:Region {region_id: $region_id})
        SET r += $properties
        RETURN r
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, region_id=region_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated region node: {region_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create region node: {region_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating region node: {e}")
            raise

    async def _create_market_data_node(self, market_data: Dict[str, Any], market_data_id: str) -> Dict[str, Any]:
        """Create a market data node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract market data attributes
        date = market_data.get("date")

        if not date:
            self.logger.warning("Date is required for market data node")
            return {}

        node_properties = {
            "market_data_id": market_data_id,
            "region_id": market_data.get("region_id"),
            "date": date,
            "median_price": market_data.get("median_price"),
            "inventory_count": market_data.get("inventory_count"),
            "sales_volume": market_data.get("sales_volume"),
            "new_listings": market_data.get("new_listings"),
            "days_on_market": market_data.get("days_on_market"),
            "months_supply": market_data.get("months_supply"),
            "price_per_sqft": market_data.get("price_per_sqft"),
            "duration": market_data.get("duration"),
            "source": market_data.get("source"),
            "last_updated": market_data.get("last_updated"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Format content for better readability
        node_properties["content"] = format_market_content(market_data)

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        # Create or merge market data node
        query = """
        MERGE (m:MarketData {market_data_id: $market_data_id})
        SET m += $properties
        RETURN m
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, market_data_id=market_data_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated market data node: {market_data_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create market data node: {market_data_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating market data node: {e}")
            raise

    async def _create_metric_node(
        self, market_data: Dict[str, Any], metric_name: str, metric_id: str
    ) -> Dict[str, Any]:
        """Create a metric node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract metric value
        metric_value = market_data.get(metric_name)
        if metric_value is None:
            self.logger.debug(f"No value for metric {metric_name}")
            return {}

        node_properties = {
            "metric_id": metric_id,
            "name": metric_name,
            "value": metric_value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Create or merge metric node
        query = """
        MERGE (m:Metric {metric_id: $metric_id})
        SET m += $properties
        RETURN m
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, metric_id=metric_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated metric node: {metric_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create metric node: {metric_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating metric node: {e}")
            raise

    async def _create_relationship(
        self, from_node: Dict[str, str], to_node: Dict[str, str], relationship_type: str, properties: Dict[str, Any]
    ) -> bool:
        """Create a relationship between two nodes in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract node information
        from_label = from_node["label"]
        from_id_field = from_node["id_field"]
        from_id_value = from_node["id_value"]

        to_label = to_node["label"]
        to_id_field = to_node["id_field"]
        to_id_value = to_node["id_value"]

        # Add timestamp if not present
        if "created_at" not in properties:
            properties["created_at"] = datetime.now(timezone.utc).isoformat()

        # Create relationship
        query = f"""
        MATCH (a:{from_label} {{{from_id_field}: $from_id_value}})
        MATCH (b:{to_label} {{{to_id_field}: $to_id_value}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        RETURN r
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(
                    query, from_id_value=from_id_value, to_id_value=to_id_value, properties=properties
                )
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
