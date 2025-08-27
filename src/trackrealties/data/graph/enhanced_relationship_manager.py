"""
Enhanced Relationship manager for establishing real estate-specific connections.

This module provides enhanced functionality to establish and manage relationships
between entities in the knowledge graph with real estate domain knowledge.
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


class EnhancedRelationshipManager:
    """
    Enhanced relationship manager with real estate-specific relationship logic.

    This class provides methods to establish and manage relationships
    between entities in the Neo4j graph database with real estate domain knowledge.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """
        Initialize the EnhancedRelationshipManager.

        Args:
            uri: Neo4j URI
            user: Neo4j username
            pass: Neo4j pass
            database: Neo4j database name
        """
        self.logger = logging.getLogger(__name__)
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.driver = None

    def _is_property_data(self, data: Dict[str, Any]) -> bool:
        """Check if data is property listing data."""
        # Property data has property-specific fields
        property_indicators = ["id", "formattedAddress", "propertyType", "bedrooms", "bathrooms", "squareFootage"]
        return any(data.get(field) for field in property_indicators)

    def _is_market_data(self, data: Dict[str, Any]) -> bool:
        """Check if data is market data."""
        # Market data has market-specific fields
        market_indicators = ["region_id", "median_price", "months_supply", "inventory_count"]
        return any(data.get(field) for field in market_indicators)

    def _is_rental_property(self, property_data: Dict[str, Any]) -> bool:
        """Check if this is a rental property that typically doesn't have agents."""
        # Check if any history event is a rental
        history = property_data.get("history", {})
        if isinstance(history, dict):
            for event_data in history.values():
                if isinstance(event_data, dict):
                    event = event_data.get("event", "")
                    if "rental" in event.lower():
                        return True

        # Check for rental indicators in property type or status
        property_type = property_data.get("propertyType", "").lower()
        status = property_data.get("status", "").lower()

        rental_indicators = ["rental", "rent", "apartment", "condo"]
        return any(indicator in property_type for indicator in rental_indicators) or "rent" in status

    def _is_historical_property_data(self, property_data: Dict[str, Any]) -> bool:
        """Check if this is historical property data that may not have current agent info."""
        # Check if the property was removed (indicating historical data)
        removed_date = property_data.get("removedDate")
        if removed_date:
            return True

        # Check for old property status that indicates historical data
        status = property_data.get("status", "").lower()
        historical_statuses = ["sold", "closed", "expired", "cancelled", "withdrawn"]

        return status in historical_statuses

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

    async def establish_market_context_relationships(
        self, property_data: Dict[str, Any], market_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Establish relationships between properties and market context."""

        # Validate that we have the correct data types
        if not self._is_property_data(property_data):
            self.logger.warning("First parameter is not property data - skipping market context relationships")
            return {
                "relationships_created": 0,
                "context_relationships": [],
                "market_matches": [],
                "price_comparisons": [],
                "error": "Invalid property data provided",
            }

        property_price = property_data.get("price", 0)
        property_county = property_data.get("county", "").lower().strip()
        property_city = property_data.get("city", "").lower().strip()
        property_state = property_data.get("state", "").lower().strip()

        results = {
            "relationships_created": 0,
            "context_relationships": [],
            "market_matches": [],
            "price_comparisons": [],
        }

        if property_price <= 0:
            self.logger.warning("Property has no valid price for market context")
            return results

        # Find matching market data by geographic hierarchy
        for market in market_data:
            market_location = market.get("location", "").lower()
            market_median = market.get("median_price", 0)
            market_county = market.get("county", "").lower().strip() if market.get("county") else ""
            market_city = market.get("city", "").lower().strip() if market.get("city") else ""
            market_state = market.get("state", "").lower().strip() if market.get("state") else ""
            market_region_id = market.get("region_id")

            if market_median <= 0 or not market_region_id:
                continue

            # Determine match level (exact match preferred)
            match_level = None
            match_score = 0

            # City-level match (highest priority)
            if (
                property_city and market_city and property_city == market_city and property_county and market_county and property_county == market_county and property_state and market_state and property_state == market_state
            ):
                match_level = "CITY"
                match_score = 100
            # County-level match
            elif (
                property_county and market_county and property_county == market_county and property_state and market_state and property_state == market_state
            ):
                match_level = "COUNTY"
                match_score = 80
            # State-level match (lowest priority)
            elif property_state and market_state and property_state == market_state:
                match_level = "STATE"
                match_score = 60
            # Location name contains match
            elif property_city and property_city in market_location:
                match_level = "LOCATION_CONTAINS"
                match_score = 70

            if match_level:
                # Calculate price positioning
                price_difference = property_price - market_median
                price_difference_percent = (price_difference / market_median) * 100

                # Determine price context
                if property_price > market_median * 1.2:
                    context = "SIGNIFICANTLY_ABOVE_MARKET"
                elif property_price > market_median:
                    context = "ABOVE_MARKET"
                elif property_price < market_median * 0.8:
                    context = "SIGNIFICANTLY_BELOW_MARKET"
                else:
                    context = "BELOW_MARKET"

                # Create relationship with rich context
                relationship_created = await self.create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                    to_node={"label": "MarketData", "id_field": "market_data_id", "id_value": market_region_id},
                    relationship_type="PRICED_RELATIVE_TO_MARKET",
                    properties={
                        "context": context,
                        "match_level": match_level,
                        "match_score": match_score,
                        "price_difference_percent": round(price_difference_percent, 2),
                        "price_difference_amount": price_difference,
                        "property_price": property_price,
                        "market_median": market_median,
                        "market_location": market.get("location"),
                        "comparison_date": market.get("date"),
                        "market_days_on_market": market.get("days_on_market"),
                        "market_inventory_count": market.get("inventory_count"),
                    },
                )

                if relationship_created:
                    results["relationships_created"] += 1
                    results["context_relationships"].append(context)
                    results["market_matches"].append(
                        {
                            "match_level": match_level,
                            "match_score": match_score,
                            "market_location": market.get("location"),
                        }
                    )
                    results["price_comparisons"].append(
                        {
                            "context": context,
                            "difference_percent": round(price_difference_percent, 2),
                            "market_median": market_median,
                        }
                    )

        return results

    async def establish_agent_performance_relationships(
        self, property_data: Dict[str, Any], market_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Establish agent performance relationships based on market context."""

        results = {"relationships_created": 0, "agent_metrics": {}, "skipped_reason": None}

        # Skip agent relationships for properties that typically don't have agents
        # Check if this is a rental property
        if self._is_rental_property(property_data):
            results["skipped_reason"] = "rental_property"
            return results

        # Check if this is historical data that may not have agent info
        if self._is_historical_property_data(property_data):
            results["skipped_reason"] = "historical_data"
            return results

        # Validate agent data exists and has required fields
        agent_data = property_data.get("listingAgent", {})
        if not agent_data or not isinstance(agent_data, dict):
            results["skipped_reason"] = "no_agent_data"
            return results

        agent_name = agent_data.get("name", "").strip()
        agent_email = agent_data.get("email", "").strip()
        if not agent_name:
            results["skipped_reason"] = "no_agent_name"
            return results

        # Create agent ID using the same logic as EnhancedGraphBuilder
        if agent_email:
            agent_id = agent_email.lower()
        else:
            # Create a sanitized ID from the name (same as EnhancedGraphBuilder)
            agent_id = re.sub(r"[^a-z0-9]", "_", agent_name.lower())

        property_price = property_data.get("price", 0)
        days_on_market = property_data.get("daysOnMarket", 0)

        # Validate we have property price data
        if property_price <= 0:
            results["skipped_reason"] = "no_price_data"
            return results

        # Find relevant market data for performance comparison
        relevant_market = None
        for market in market_data:
            if self._is_market_relevant_to_property(property_data, market):
                relevant_market = market
                break

        if relevant_market and property_price > 0:
            market_median = relevant_market.get("median_price", 0)
            market_days_on_market = relevant_market.get("days_on_market", 0)

            # Calculate agent performance metrics
            price_performance = "AT_MARKET"
            if market_median > 0:
                if property_price > market_median * 1.1:
                    price_performance = "ABOVE_MARKET"
                elif property_price < market_median * 0.9:
                    price_performance = "BELOW_MARKET"

            time_performance = "AVERAGE"
            if market_days_on_market > 0:
                if days_on_market < market_days_on_market * 0.8:
                    time_performance = "FAST"
                elif days_on_market > market_days_on_market * 1.2:
                    time_performance = "SLOW"

            # Create agent performance relationship
            try:
                relationship_created = await self.create_relationship(
                    from_node={
                        "label": "Agent",
                        "id_field": "agent_id",
                        "id_value": agent_id,  # Use the corrected agent_id
                    },
                    to_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                    relationship_type="PERFORMANCE_ON_LISTING",
                    properties={
                        "price_performance": price_performance,
                        "time_performance": time_performance,
                        "price_vs_market_percent": (
                            round(((property_price - market_median) / market_median) * 100, 2)
                            if market_median > 0
                            else 0
                        ),
                        "days_vs_market": days_on_market - market_days_on_market if market_days_on_market > 0 else 0,
                        "listing_price": property_price,
                        "market_median": market_median,
                        "days_on_market": days_on_market,
                        "market_days_on_market": market_days_on_market,
                    },
                )

                if relationship_created:
                    results["relationships_created"] += 1
                    results["agent_metrics"] = {
                        "price_performance": price_performance,
                        "time_performance": time_performance,
                        "agent_id": agent_id,
                        "agent_name": agent_name,
                    }
                else:
                    results["skipped_reason"] = "relationship_creation_failed"

            except Exception as e:
                self.logger.warning(
                    f"Failed to create agent performance relationship for agent {agent_name} (ID: {agent_id}): {e}"
                )
                results["skipped_reason"] = f"error: {str(e)}"
        else:
            results["skipped_reason"] = "no_relevant_market_data"

        return results

    async def establish_comparable_property_relationships(
        self, property_data: Dict[str, Any], comparable_properties: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Establish relationships between similar properties."""

        results = {"relationships_created": 0, "comparables_found": 0}

        property_id = property_data.get("id")
        property_price = property_data.get("price", 0)
        property_sqft = property_data.get("squareFootage", 0)
        property_bedrooms = property_data.get("bedrooms", 0)
        property_bathrooms = property_data.get("bathrooms", 0)

        if not property_id or property_price <= 0:
            return results

        for comp_property in comparable_properties:
            comp_id = comp_property.get("id")
            comp_price = comp_property.get("price", 0)
            comp_sqft = comp_property.get("squareFootage", 0)
            comp_bedrooms = comp_property.get("bedrooms", 0)
            comp_bathrooms = comp_property.get("bathrooms", 0)

            if not comp_id or comp_id == property_id or comp_price <= 0:
                continue

            # Check if comparable property exists in graph before creating relationship
            try:
                check_query = "MATCH (p:Property {property_id: $property_id}) RETURN p.property_id as id LIMIT 1"
                async with self.driver.session(database=self.database) as session:
                    result = await session.run(check_query, property_id=comp_id)
                    record = await result.single()
                    if not record:
                        self.logger.debug(f"Comparable property {comp_id} not yet in graph, skipping relationship")
                        continue
            except Exception as e:
                self.logger.warning(f"Failed to check existence of comparable property {comp_id}: {e}")
                continue

            # Calculate similarity metrics
            similarity_score = self._calculate_property_similarity(property_data, comp_property)

            if similarity_score >= 0.7:  # 70% similarity threshold
                price_difference = property_price - comp_price
                price_difference_percent = (price_difference / comp_price) * 100 if comp_price > 0 else 0

                # Determine relationship type based on price difference
                if abs(price_difference_percent) <= 5:
                    relationship_type = "SIMILAR_PRICED_COMPARABLE"
                elif price_difference_percent > 5:
                    relationship_type = "HIGHER_PRICED_COMPARABLE"
                else:
                    relationship_type = "LOWER_PRICED_COMPARABLE"

                try:
                    relationship_created = await self.create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                        to_node={"label": "Property", "id_field": "property_id", "id_value": comp_id},
                        relationship_type=relationship_type,
                        properties={
                            "similarity_score": round(similarity_score, 3),
                            "price_difference": price_difference,
                            "price_difference_percent": round(price_difference_percent, 2),
                            "sqft_difference": property_sqft - comp_sqft if property_sqft and comp_sqft else None,
                            "bedroom_difference": property_bedrooms - comp_bedrooms,
                            "bathroom_difference": property_bathrooms - comp_bathrooms,
                            "comparison_type": "automated_comparable",
                        },
                    )

                    if relationship_created:
                        results["relationships_created"] += 1
                        results["comparables_found"] += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to create comparable property relationship between {property_id} and {comp_id}: {e}"
                    )
                    continue

        return results

    def _calculate_property_similarity(self, prop1: Dict[str, Any], prop2: Dict[str, Any]) -> float:
        """Calculate similarity score between two properties."""

        similarity_factors = []

        # Price similarity (weight: 0.3)
        price1 = prop1.get("price", 0)
        price2 = prop2.get("price", 0)
        if price1 > 0 and price2 > 0:
            price_diff_percent = abs(price1 - price2) / max(price1, price2)
            price_similarity = max(0, 1 - price_diff_percent)
            similarity_factors.append((price_similarity, 0.3))

        # Size similarity (weight: 0.25)
        sqft1 = prop1.get("squareFootage", 0)
        sqft2 = prop2.get("squareFootage", 0)
        if sqft1 > 0 and sqft2 > 0:
            sqft_diff_percent = abs(sqft1 - sqft2) / max(sqft1, sqft2)
            sqft_similarity = max(0, 1 - sqft_diff_percent)
            similarity_factors.append((sqft_similarity, 0.25))

        # Bedroom similarity (weight: 0.15)
        bed1 = prop1.get("bedrooms", 0)
        bed2 = prop2.get("bedrooms", 0)
        if bed1 == bed2:
            similarity_factors.append((1.0, 0.15))
        elif abs(bed1 - bed2) == 1:
            similarity_factors.append((0.7, 0.15))
        else:
            similarity_factors.append((0.3, 0.15))

        # Bathroom similarity (weight: 0.1)
        bath1 = prop1.get("bathrooms", 0)
        bath2 = prop2.get("bathrooms", 0)
        bath_diff = abs(bath1 - bath2)
        if bath_diff == 0:
            similarity_factors.append((1.0, 0.1))
        elif bath_diff <= 0.5:
            similarity_factors.append((0.8, 0.1))
        elif bath_diff <= 1:
            similarity_factors.append((0.6, 0.1))
        else:
            similarity_factors.append((0.3, 0.1))

        # Property type similarity (weight: 0.1)
        type1 = prop1.get("propertyType", "").lower()
        type2 = prop2.get("propertyType", "").lower()
        if type1 == type2:
            similarity_factors.append((1.0, 0.1))
        else:
            similarity_factors.append((0.0, 0.1))

        # Location similarity (weight: 0.1)
        city1 = prop1.get("city", "").lower()
        city2 = prop2.get("city", "").lower()
        county1 = prop1.get("county", "").lower()
        county2 = prop2.get("county", "").lower()

        if city1 == city2:
            similarity_factors.append((1.0, 0.1))
        elif county1 == county2:
            similarity_factors.append((0.7, 0.1))
        else:
            similarity_factors.append((0.3, 0.1))

        # Calculate weighted average
        if similarity_factors:
            total_score = sum(score * weight for score, weight in similarity_factors)
            total_weight = sum(weight for _, weight in similarity_factors)
            return total_score / total_weight if total_weight > 0 else 0

        return 0

    def _is_market_relevant_to_property(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Check if market data is relevant to a property."""

        prop_city = property_data.get("city", "").lower().strip()
        prop_county = property_data.get("county", "").lower().strip()
        prop_state = property_data.get("state", "").lower().strip()

        market_location = market_data.get("location", "").lower()
        market_city = market_data.get("city", "").lower().strip() if market_data.get("city") else ""
        market_county = market_data.get("county", "").lower().strip() if market_data.get("county") else ""
        market_state = market_data.get("state", "").lower().strip() if market_data.get("state") else ""

        # Exact city match
        if prop_city and market_city and prop_city == market_city:
            return True

        # County match
        if prop_county and market_county and prop_county == market_county:
            return True

        # Location name contains property city
        if prop_city and prop_city in market_location:
            return True

        # State match as last resort
        if prop_state and market_state and prop_state == market_state:
            return True

        return False

    async def create_relationship(
        self,
        from_node: Dict[str, str],
        to_node: Dict[str, str],
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between two nodes."""

        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        if properties is None:
            properties = {}

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

        # Format properties for Neo4j
        formatted_properties = format_relationship_properties(properties)

        # First check if both nodes exist
        check_nodes_query = f"""
        OPTIONAL MATCH (a:{from_label} {{{from_id_field}: $from_id_value}})
        OPTIONAL MATCH (b:{to_label} {{{to_id_field}: $to_id_value}})
        RETURN a IS NOT NULL as from_exists, b IS NOT NULL as to_exists
        """

        try:
            async with self.driver.session(database=self.database) as session:
                # Check node existence first
                check_result = await session.run(
                    check_nodes_query, from_id_value=from_id_value, to_id_value=to_id_value
                )
                check_record = await check_result.single()

                if not check_record:
                    self.logger.warning("Failed to check node existence for relationship creation")
                    return False

                from_exists = check_record["from_exists"]
                to_exists = check_record["to_exists"]

                if not from_exists:
                    self.logger.warning(
                        f"Source node does not exist: {from_label} with {from_id_field}={from_id_value}"
                    )
                    return False

                if not to_exists:
                    self.logger.warning(f"Target node does not exist: {to_label} with {to_id_field}={to_id_value}")
                    return False

                # Create relationship
                create_query = f"""
                MATCH (a:{from_label} {{{from_id_field}: $from_id_value}})
                MATCH (b:{to_label} {{{to_id_field}: $to_id_value}})
                MERGE (a)-[r:{relationship_type}]->(b)
                SET r += $properties
                RETURN r
                """

                result = await session.run(
                    create_query, from_id_value=from_id_value, to_id_value=to_id_value, properties=formatted_properties
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

    async def find_comparable_properties(
        self, target_property_id: str, radius_miles: float = 5, price_variance: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Find comparable properties within geographic and price radius."""

        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        query = """
        MATCH (target:Property {property_id: $target_id})
        MATCH (target)-[:BELONGS_TO_SEGMENT]->(segment:PriceSegment)
        MATCH (comp:Property)-[:BELONGS_TO_SEGMENT]->(segment)
        MATCH (target)-[:LOCATED_IN]->(target_loc:Location)
        MATCH (comp)-[:LOCATED_IN]->(comp_loc:Location)
        WHERE (target_loc = comp_loc OR 
               (target_loc)-[:PART_OF*1..2]-(comp_loc) OR
               (comp_loc)-[:PART_OF*1..2]-(target_loc))
        AND target.property_id <> comp.property_id
        AND comp.price >= target.price * (1 - $price_variance)
        AND comp.price <= target.price * (1 + $price_variance)
        RETURN comp, segment.name as price_segment,
               target_loc.name as target_location,
               comp_loc.name as comp_location
        LIMIT 10
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, target_id=target_property_id, price_variance=price_variance)

                comparables = []
                async for record in result:
                    comp_data = dict(record["comp"])
                    comp_data["price_segment"] = record["price_segment"]
                    comp_data["target_location"] = record["target_location"]
                    comp_data["comp_location"] = record["comp_location"]
                    comparables.append(comp_data)

                self.logger.debug(f"Found {len(comparables)} comparable properties for {target_property_id}")
                return comparables

        except Exception as e:
            self.logger.error(f"Error finding comparable properties: {e}")
            return []

    async def get_market_performance_summary(self, location_id: str) -> Dict[str, Any]:
        """Get market performance summary for a location."""

        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        query = """
        MATCH (loc:Location {location_id: $location_id})
        MATCH (prop:Property)-[:LOCATED_IN]->(loc)
        MATCH (prop)-[:PRICED_RELATIVE_TO_MARKET]->(market:MarketData)
        RETURN 
            COUNT(prop) as total_properties,
            AVG(prop.price) as avg_price,
            MIN(prop.price) as min_price,
            MAX(prop.price) as max_price,
            AVG(prop.daysOnMarket) as avg_days_on_market,
            COLLECT(DISTINCT prop.propertyType) as property_types,
            market.median_price as market_median,
            market.days_on_market as market_days_on_market
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(query, location_id=location_id)
                record = await result.single()

                if record:
                    return {
                        "location_id": location_id,
                        "total_properties": record["total_properties"],
                        "avg_price": record["avg_price"],
                        "min_price": record["min_price"],
                        "max_price": record["max_price"],
                        "avg_days_on_market": record["avg_days_on_market"],
                        "property_types": record["property_types"],
                        "market_median": record["market_median"],
                        "market_days_on_market": record["market_days_on_market"],
                    }
                else:
                    return {"location_id": location_id, "error": "No data found"}

        except Exception as e:
            self.logger.error(f"Error getting market performance summary: {e}")
            return {"location_id": location_id, "error": str(e)}

    def _categorize_market_price_level(self, market_data: Dict[str, Any]) -> str:
        """
        Categorize market price level adjusted for region.

        Args:
            market_data: Market data dictionary with median_price, state, etc.

        Returns:
            Price level category: "luxury", "premium", "moderate", "affordable"
        """
        median_price = market_data.get("median_price", 0)
        state = market_data.get("state", "").upper()

        if median_price <= 0:
            return "unknown"

        # Regional cost of living adjustments
        regional_multipliers = {
            "CA": 2.5,
            "NY": 2.2,
            "MA": 2.0,
            "WA": 1.8,
            "CO": 1.6,
            "TX": 1.2,
            "FL": 1.3,
            "NC": 1.0,
            "GA": 1.0,
            "TN": 0.9,
            "OH": 0.8,
            "MI": 0.8,
            "PA": 0.9,
            "IL": 1.1,
            "AZ": 1.1,
            "NV": 1.0,
            "OR": 1.4,
            "UT": 1.3,
            "ID": 0.9,
            "MT": 0.8,
            "default": 1.0,
        }

        # Apply regional adjustment
        regional_multiplier = regional_multipliers.get(state, regional_multipliers["default"])
        adjusted_price = median_price / regional_multiplier

        # Categorize based on adjusted price
        if adjusted_price >= 800000:
            return "luxury"
        elif adjusted_price >= 400000:
            return "premium"
        elif adjusted_price >= 200000:
            return "moderate"
        else:
            return "affordable"

    def _categorize_market_speed(self, market_data: Dict[str, Any]) -> str:
        """
        Categorize market speed based on days on market.

        Args:
            market_data: Market data dictionary with days_on_market

        Returns:
            Market speed category: "very_fast", "fast", "normal", "slow"
        """
        days_on_market = market_data.get("days_on_market", 0)

        if days_on_market <= 0:
            return "unknown"

        # Market speed thresholds
        if days_on_market <= 15:
            return "very_fast"
        elif days_on_market <= 30:
            return "fast"
        elif days_on_market <= 60:
            return "normal"
        else:
            return "slow"

    def _categorize_market_health(self, market_data: Dict[str, Any]) -> str:
        """
        Categorize overall market health based on months supply and other indicators.

        Args:
            market_data: Market data dictionary with months_supply, days_on_market, etc.

        Returns:
            Market health category: "seller_market", "balanced_market", "buyer_market", "stagnant_market"
        """
        months_supply = market_data.get("months_supply", 0)
        days_on_market = market_data.get("days_on_market", 0)
        inventory_count = market_data.get("inventory_count", 0)

        if months_supply <= 0:
            return "unknown"

        # Primary indicator: months supply
        if months_supply <= 3:
            market_health = "seller_market"
        elif months_supply <= 6:
            market_health = "balanced_market"
        elif months_supply <= 12:
            market_health = "buyer_market"
        else:
            market_health = "stagnant_market"

        # Adjust based on days on market if available
        if days_on_market > 0:
            if days_on_market <= 20 and market_health in ["balanced_market", "buyer_market"]:
                market_health = "seller_market"
            elif days_on_market >= 90 and market_health in ["seller_market", "balanced_market"]:
                market_health = "buyer_market"

        # Adjust for extremely low inventory (hot market indicator)
        if inventory_count > 0 and inventory_count < 50 and market_health != "seller_market":
            market_health = "seller_market"

        return market_health

    def _segment_market_data(self, market_data: Dict[str, Any]) -> str:
        """
        Segment market data based on median price, region type, and location.

        Args:
            market_data: Market data dictionary

        Returns:
            Market segment identifier combining price level, region type, and location
        """
        region_type = market_data.get("region_type", "unknown").lower()
        region_id = market_data.get("region_id", "unknown")
        location = market_data.get("location", "unknown").lower()

        # Get price level category
        price_level = self._categorize_market_price_level(market_data)

        # Get market health for additional context
        market_health = self._categorize_market_health(market_data)

        # Create comprehensive segment ID
        if price_level == "unknown":
            segment_id = f"unknown_market_{region_type}_{region_id}"
        else:
            # Include market health and location for more granular segmentation
            location_key = location.replace(" ", "_").replace(",", "")
            segment_id = f"{price_level}_{market_health}_{region_type}_{location_key}_{region_id}"

        return segment_id

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            self.logger.debug("Neo4j connection closed")
