"""
Enhanced Graph Builder for TrackRealties AI Platform.

This module provides enhanced functionality to build a knowledge graph from property listings
and market data, creating meaningful real estate relationships in Neo4j.
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


class EnhancedGraphBuilder:
    """
    Enhanced graph builder with real estate-specific relationships and price segmentation.

    This class provides methods to create nodes and relationships in Neo4j,
    with enhanced real estate domain knowledge and market context.
    """

    def __init__(self):
        """
        Initialize the EnhancedGraphBuilder.
        """
        self.logger = logging.getLogger(__name__)
        self.driver = None

        # Price segments for market analysis
        self.price_segments = {
            "AFFORDABLE": {"min": 0, "max": 200000, "id": "affordable_under_200k"},
            "MODERATE": {"min": 200000, "max": 400000, "id": "moderate_200k_400k"},
            "PREMIUM": {"min": 400000, "max": 1000000, "id": "premium_400k_1m"},
            "LUXURY": {"min": 1000000, "max": float("inf"), "id": "luxury_over_1m"},
        }

        # Price ranges by property type (per sq ft for sales, monthly for rentals)
        self.sale_price_ranges = {
            "single_family": {"min": 30, "max": 1500, "typical_min": 80, "typical_max": 400},
            "condo": {"min": 50, "max": 2000, "typical_min": 120, "typical_max": 600},
            "townhouse": {"min": 40, "max": 1200, "typical_min": 100, "typical_max": 500},
            "multi_family": {"min": 30, "max": 800, "typical_min": 60, "typical_max": 300},
            "apartment": {"min": 50, "max": 2000, "typical_min": 100, "typical_max": 800},
            "manufactured": {"min": 20, "max": 200, "typical_min": 40, "typical_max": 120},
            "commercial": {"min": 50, "max": 3000, "typical_min": 100, "typical_max": 1000},
            "land": {"min": 0.1, "max": 500, "typical_min": 1, "typical_max": 50},  # per sq ft
        }

        # Rental price ranges (monthly per sq ft)
        self.rental_price_ranges = {
            "single_family": {"min": 0.5, "max": 8, "typical_min": 1, "typical_max": 4},
            "condo": {"min": 0.8, "max": 12, "typical_min": 1.5, "typical_max": 6},
            "townhouse": {"min": 0.7, "max": 10, "typical_min": 1.2, "typical_max": 5},
            "multi_family": {"min": 0.6, "max": 8, "typical_min": 1, "typical_max": 4},
            "apartment": {"min": 0.8, "max": 15, "typical_min": 1.5, "typical_max": 8},
            "manufactured": {"min": 0.3, "max": 3, "typical_min": 0.5, "typical_max": 2},
        }

        # Regional multipliers for major markets
        self.regional_multipliers = {
            # Tier 1 Markets (Very High Cost)
            "san_francisco": 2.8,
            "palo_alto": 2.5,
            "mountain_view": 2.4,
            "new_york": 2.2,
            "manhattan": 3.0,
            "brooklyn": 1.8,
            "los_angeles": 1.9,
            "beverly_hills": 2.5,
            "santa_monica": 2.1,
            "seattle": 1.7,
            "boston": 1.6,
            "washington_dc": 1.5,
            # Tier 2 Markets (High Cost)
            "miami": 1.4,
            "denver": 1.3,
            "austin": 1.3,
            "san_diego": 1.8,
            "chicago": 1.2,
            "portland": 1.3,
            "atlanta": 1.1,
            "nashville": 1.2,
            # Tier 3 Markets (Moderate Cost)
            "phoenix": 1.0,
            "dallas": 0.9,
            "houston": 0.8,
            "tampa": 0.9,
            "las_vegas": 0.9,
            "charlotte": 0.9,
            "raleigh": 0.9,
            "orlando": 0.8,
            # Tier 4 Markets (Lower Cost)
            "kansas_city": 0.7,
            "oklahoma_city": 0.6,
            "cleveland": 0.6,
            "memphis": 0.5,
            "birmingham": 0.5,
            "little_rock": 0.5,
            # Default for unknown areas
            "default": 1.0,
        }

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

    def _extract_price(self, property_data: Dict[str, Any]) -> float:
        """Extract and normalize price from property data."""
        price = property_data.get("price", 0)

        if isinstance(price, str):
            # Clean price string
            price_clean = re.sub(r"[^0-9.]", "", price)
            try:
                price = float(price_clean)
            except ValueError:
                price = 0

        return float(price) if price else 0

    def _determine_listing_type(self, property_data: Dict[str, Any]) -> str:
        """Determine listing type from property data."""
        # Check multiple possible field names for listing type
        listing_type = (property_data.get("listing_type", "") or property_data.get("listingType", "") or property_data.get("type", "")).lower()

        # Check various fields that might indicate rental
        rental_indicators = ["rental", "rent", "lease", "rented", "for_rent"]

        # Direct check for rental indicators in listing_type
        for indicator in rental_indicators:
            if indicator in listing_type:
                return "rental"

        # Check for monthly pricing indicators
        price = property_data.get("price", 0) or property_data.get("rent", 0)
        if price and price < 50000:  # Likely monthly rental if under $50k
            return "rental"

        # Check status field for rental indicators
        status = property_data.get("status", "").lower()
        for indicator in rental_indicators:
            if indicator in status:
                return "rental"

        # Check for new construction
        if "new" in listing_type or "construction" in listing_type:
            return "new_construction"

        # Check for auction
        if "auction" in listing_type:
            return "auction"

        # Check for explicit sale indicators
        if "sale" in listing_type or "for_sale" in listing_type:
            return "sale"

        # Default to sale
        return "sale"

    def _determine_property_type(self, property_data: Dict[str, Any]) -> str:
        """Determine property type from property data."""
        # Check for rental first based on listing type
        listing_type = self._determine_listing_type(property_data)
        if listing_type == "rental":
            return "rental"

        # Check explicit property type fields
        prop_type = (property_data.get("property_type", "") or property_data.get("propertyType", "") or property_data.get("type", "")).lower()

        if "single" in prop_type or "sfr" in prop_type:
            return "single_family"
        elif "condo" in prop_type or "condominium" in prop_type:
            return "condo"
        elif "town" in prop_type:
            return "townhouse"
        elif "multi" in prop_type or "duplex" in prop_type:
            return "multi_family"
        elif "apartment" in prop_type or "apt" in prop_type:
            return "apartment"
        elif "manufactured" in prop_type or "mobile" in prop_type:
            return "manufactured"
        elif "commercial" in prop_type or "office" in prop_type or "retail" in prop_type:
            return "commercial"
        elif "land" in prop_type or "lot" in prop_type:
            return "land"
        else:
            return "single_family"  # Default assumption

    def _should_validate_price(
        self, price: float, listing_type: str, property_type: str, property_data: Dict[str, Any]
    ) -> bool:
        """Determine if price validation should be performed."""

        # Skip validation for zero or negative prices
        if price <= 0:
            return False

        # Skip validation for land without improvements (often has $0 assessed value)
        if property_type == "land" and price < 1000:
            return False

        # Skip validation for auction properties (prices can be misleading)
        if listing_type == "auction":
            return False

        # Skip validation for properties marked as "Coming Soon" or similar
        status = property_data.get("status", "").lower()
        if status in ["coming_soon", "pending", "sold", "off_market"]:
            return False

        return True

    def _extract_location(self, property_data: Dict[str, Any]) -> str:
        """Extract location from property data."""
        city = property_data.get("city", "").lower()

        # Try to match known markets
        location_key = city.replace(" ", "_")
        if location_key in self.regional_multipliers:
            return location_key

        # Check for major metro areas
        major_metros = {
            "san_francisco": ["san francisco", "sf", "palo alto", "mountain view", "fremont"],
            "new_york": ["new york", "nyc", "manhattan", "brooklyn", "queens"],
            "los_angeles": ["los angeles", "la", "hollywood", "santa monica"],
            "chicago": ["chicago"],
            "seattle": ["seattle"],
            "boston": ["boston"],
            "washington_dc": ["washington", "dc", "arlington", "alexandria"],
            "miami": ["miami", "miami beach"],
            "denver": ["denver"],
            "austin": ["austin"],
        }

        for metro, cities in major_metros.items():
            if any(c in city for c in cities):
                return metro

        return "default"

    def _get_regional_multiplier(self, location: str) -> float:
        """Get regional price multiplier."""
        return self.regional_multipliers.get(location, self.regional_multipliers["default"])

    def _validate_and_segment_property(self, property_data: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """Validate property price and return segmentation info."""
        warnings = []

        # Check if this is actually property listing data
        if not self._is_property_data(property_data):
            return (
                False,
                "unknown",
                ["Not property listing data - use _validate_and_segment_market_data for market data"],
            )

        # Get price and status
        price = self._extract_price(property_data)
        listing_type = self._determine_listing_type(property_data)
        property_type = self._determine_property_type(property_data)

        # Skip validation for certain statuses
        if not self._should_validate_price(price, listing_type, property_type, property_data):
            return True, "unknown", [f"Skipping price validation for {listing_type} or status"]

        # Handle missing or invalid price
        if price <= 0:
            if listing_type == "rental":
                warnings.append("Rental property missing monthly rent price")
                return False, "unknown", warnings
            else:
                warnings.append("Property missing valid sale price")
                return False, "unknown", warnings

        # Different validation for rentals vs sales
        if listing_type == "rental":
            return self._validate_rental_price(property_data, price, warnings)
        else:
            return self._validate_sale_price(property_data, price, warnings)

    def _validate_rental_price(
        self, property_data: Dict[str, Any], price: float, warnings: List[str]
    ) -> Tuple[bool, str, List[str]]:
        """Validate rental property pricing."""
        # Rental price validation (monthly rent)
        if price < 500:
            warnings.append(f"Unusually low rental price: ${price}/month")
            return False, "unknown", warnings
        elif price > 50000:
            warnings.append(f"Unusually high rental price: ${price}/month")
            return False, "unknown", warnings

        # Convert monthly rent to annual equivalent for segmentation
        annual_equivalent = price * 12 * 15  # Rough price-to-rent ratio

        # Apply regional adjustment
        location = self._extract_location(property_data)
        regional_multiplier = self._get_regional_multiplier(location)
        adjusted_price = annual_equivalent / regional_multiplier

        # Determine segment
        segment_name, segment_id = self._determine_price_segment(adjusted_price)

        warnings.append(f"Rental property segmented based on annual equivalent: ${annual_equivalent:,.0f}")
        return True, segment_id, warnings

    def _validate_sale_price(
        self, property_data: Dict[str, Any], price: float, warnings: List[str]
    ) -> Tuple[bool, str, List[str]]:
        """Validate sale property pricing."""
        property_type = self._determine_property_type(property_data)

        # Basic price range validation
        min_price = 1000 if property_type == "land" else 10000
        max_price = 100000000  # $100M cap

        if price < min_price:
            warnings.append(f"Price below minimum threshold for {property_type}: ${price:,.0f}")
            return False, "unknown", warnings
        elif price > max_price:
            warnings.append(f"Price above maximum threshold: ${price:,.0f}")
            return False, "unknown", warnings

        # Apply regional adjustment
        location = self._extract_location(property_data)
        regional_multiplier = self._get_regional_multiplier(location)
        adjusted_price = price / regional_multiplier

        # Determine segment
        segment_name, segment_id = self._determine_price_segment(adjusted_price)

        if regional_multiplier != 1.0:
            warnings.append(f"Regional adjustment applied: {regional_multiplier}x")

        return True, segment_id, warnings

    def _validate_and_segment_market_data(self, market_data: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
        """Validate market data pricing."""
        warnings = []

        # Check if this is actually market data
        if not self._is_market_data(market_data):
            return False, "unknown", ["Not market data - use _validate_and_segment_property for property data"]

        # Get median price
        median_price = market_data.get("median_price")

        if median_price is None or median_price <= 0:
            warnings.append("Market data missing valid median price")
            return True, "unknown", warnings  # Don't fail, just skip segmentation

        # Validate median price range
        if median_price < 50000:
            warnings.append(f"Unusually low median price: ${median_price:,.0f}")
        elif median_price > 10000000:
            warnings.append(f"Unusually high median price: ${median_price:,.0f}")

        # Apply regional adjustment
        location = self._extract_location(market_data)
        regional_multiplier = self._get_regional_multiplier(location)
        adjusted_price = median_price / regional_multiplier

        # Determine segment
        segment_name, segment_id = self._determine_price_segment(adjusted_price)

        if regional_multiplier != 1.0:
            warnings.append(f"Regional adjustment applied to market data: {regional_multiplier}x")

        return True, segment_id, warnings

    async def initialize(self) -> None:
        """Initialize the Neo4j driver and create enhanced constraints."""
        await graph_manager.initialize()
        self.driver = graph_manager._driver
        if self.driver:
            await self._create_enhanced_constraints()

    async def _test_connection(self) -> None:
        """Test the Neo4j connection."""
        await graph_manager.test_connection()

    async def _create_enhanced_constraints(self) -> None:
        """Create Neo4j constraints for enhanced schema."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        constraints = [
            # Original constraints
            "CREATE CONSTRAINT property_id IF NOT EXISTS FOR (p:Property) REQUIRE p.property_id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE",
            "CREATE CONSTRAINT office_id IF NOT EXISTS FOR (o:Office) REQUIRE o.office_id IS UNIQUE",
            "CREATE CONSTRAINT market_data_id IF NOT EXISTS FOR (m:MarketData) REQUIRE m.market_data_id IS UNIQUE",
            "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (r:Region) REQUIRE r.region_id IS UNIQUE",
            # Enhanced constraints
            "CREATE CONSTRAINT price_segment_id IF NOT EXISTS FOR (ps:PriceSegment) REQUIRE ps.segment_id IS UNIQUE",
            "CREATE CONSTRAINT property_type_id IF NOT EXISTS FOR (pt:PropertyType) REQUIRE pt.type_id IS UNIQUE",
            "CREATE CONSTRAINT geographic_level_id IF NOT EXISTS FOR (gl:GeographicLevel) REQUIRE gl.level_id IS UNIQUE",
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

                self.logger.debug("Enhanced Neo4j constraints created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create enhanced Neo4j constraints: {e}")
            raise

    def _determine_price_segment(self, price: float) -> Tuple[str, str]:
        """Determine price segment for a given price."""
        for segment_name, segment_info in self.price_segments.items():
            if segment_info["min"] <= price < segment_info["max"]:
                return segment_name, segment_info["id"]
        return "LUXURY", self.price_segments["LUXURY"]["id"]  # Default to luxury for very high prices

    async def _create_price_segment_node(
        self, segment_id: str, segment_name: str, min_price: float, max_price: float
    ) -> Dict[str, Any]:
        """Create a price segment node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        node_properties = {
            "segment_id": segment_id,
            "name": segment_name,
            "min_price": min_price,
            "max_price": max_price if max_price != float("inf") else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        query = """
        MERGE (ps:PriceSegment {segment_id: $segment_id})
        SET ps += $properties
        RETURN ps
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, segment_id=segment_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated price segment node: {segment_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create price segment node: {segment_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating price segment node: {e}")
            raise

    async def _create_property_type_node(self, property_type: str) -> Dict[str, Any]:
        """Create a property type node in Neo4j."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        type_id = property_type.lower().replace(" ", "_").replace("-", "_")

        node_properties = {
            "type_id": type_id,
            "name": property_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        query = """
        MERGE (pt:PropertyType {type_id: $type_id})
        SET pt += $properties
        RETURN pt
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, type_id=type_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated property type node: {type_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create property type node: {type_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating property type node: {e}")
            raise

    async def _create_listing_type_node(self, listing_type: str) -> Dict[str, Any]:
        """Create a listing type node in Neo4j to distinguish rental vs sale properties."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Normalize listing type ID
        type_id = listing_type.lower().replace(" ", "_").replace("-", "_")

        # Enhanced properties with rental/sale specific metadata
        node_properties = {
            "type_id": type_id,
            "name": listing_type,
            "category": self._categorize_listing_type(listing_type),
            "is_rental": self._is_rental_listing_type(listing_type),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        query = """
        MERGE (lt:ListingType {type_id: $type_id})
        SET lt += $properties
        RETURN lt
        """

        try:
            async with self.driver.session(database=graph_manager.settings.NEO4J_DATABASE) as session:
                result = await session.run(query, type_id=type_id, properties=node_properties)
                record = await result.single()

                if record:
                    self.logger.debug(f"Created/updated listing type node: {type_id}")
                    return node_properties
                else:
                    self.logger.warning(f"Failed to create listing type node: {type_id}")
                    return {}

        except Exception as e:
            self.logger.error(f"Error creating listing type node: {e}")
            raise

    def _categorize_listing_type(self, listing_type: str) -> str:
        """Categorize listing type into broader categories."""
        listing_lower = listing_type.lower()

        if "rental" in listing_lower or "rent" in listing_lower or "lease" in listing_lower:
            return "rental"
        elif "sale" in listing_lower or "standard" in listing_lower:
            return "sale"
        elif "auction" in listing_lower:
            return "auction"
        elif "new" in listing_lower or "construction" in listing_lower:
            return "new_construction"
        else:
            return "other"

    def _is_rental_listing_type(self, listing_type: str) -> bool:
        """Determine if this is a rental listing type."""
        return self._categorize_listing_type(listing_type) == "rental"

    def _determine_listing_type_from_data(self, property_data: Dict[str, Any]) -> str:
        """Smart detection of listing type from property data."""
        # First check explicit listingType field
        explicit_type = property_data.get("listingType", "").strip()
        if explicit_type:
            # Check if it's already a recognized listing type
            category = self._categorize_listing_type(explicit_type)
            if category != "other":
                return explicit_type

        # Check history for rental events
        history = property_data.get("history", {})
        if isinstance(history, dict):
            for event_data in history.values():
                if isinstance(event_data, dict):
                    event = event_data.get("event", "")
                    if "rental" in event.lower():
                        return "Rental Listing"

        # Check price indicators (rentals are typically under $50k monthly)
        price = property_data.get("price", 0)
        if price > 0 and price < 50000:
            # Additional checks for rental indicators
            property_type = property_data.get("propertyType", "").lower()
            if "apartment" in property_type or "condo" in property_type:
                return "Rental Listing"

        # Default to sale
        return explicit_type if explicit_type else "Standard"

    def _extract_office_info(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract office information from property data."""
        office_info = {}

        # Check for listing_office first
        if "listing_office" in property_data:
            office_data = property_data["listing_office"]
            if isinstance(office_data, dict):
                office_info = {
                    "name": office_data.get("name"),
                    "phone": office_data.get("phone"),
                    "email": office_data.get("email"),
                    "address": office_data.get("address"),
                    "website": office_data.get("website"),
                }

        # Also check for office in agent data
        elif "listing_agent" in property_data:
            agent_data = property_data["listing_agent"]
            if isinstance(agent_data, dict) and "office" in agent_data:
                office_data = agent_data["office"]
                if isinstance(office_data, dict):
                    office_info = {
                        "name": office_data.get("name"),
                        "phone": office_data.get("phone"),
                        "email": office_data.get("email"),
                        "address": office_data.get("address"),
                        "website": office_data.get("website"),
                    }
                elif isinstance(office_data, str):
                    office_info = {"name": office_data}

        # Check for office field directly
        elif "office" in property_data:
            office_data = property_data["office"]
            if isinstance(office_data, dict):
                office_info = {
                    "name": office_data.get("name"),
                    "phone": office_data.get("phone"),
                    "email": office_data.get("email"),
                    "address": office_data.get("address"),
                    "website": office_data.get("website"),
                }
            elif isinstance(office_data, str):
                office_info = {"name": office_data}

        # Filter out None values
        return {k: v for k, v in office_info.items() if v is not None}

    async def _create_location_node(
        self, location_id: str, location_type: str, location_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a location node with enhanced geographic information."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        node_properties = {
            "location_id": location_id,
            "location_type": location_type,
            "name": location_data.get("name"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add additional location data
        if location_type == "City":
            node_properties.update(
                {
                    "county": location_data.get("county"),
                    "state": location_data.get("state"),
                    "zipCode": location_data.get("zipCode"),
                }
            )
        elif location_type == "County":
            node_properties.update({"state": location_data.get("state")})

        # Add coordinates if available
        if location_data.get("latitude") and location_data.get("longitude"):
            node_properties.update(
                {"latitude": float(location_data["latitude"]), "longitude": float(location_data["longitude"])}
            )

        # Remove None values
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

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

    async def _build_geographic_hierarchy(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build hierarchical location relationships."""
        city = location_data.get("city")
        county = location_data.get("county")
        state = location_data.get("state")

        results = {"nodes_created": 0, "relationships_created": 0, "location_ids": []}

        # Create state node
        if state:
            state_id = f"state_{state.lower().replace(' ', '_')}"
            state_node = await self._create_location_node(state_id, "State", {"name": state})
            if state_node:
                results["nodes_created"] += 1
                results["location_ids"].append(state_id)

        # Create county node and relationship to state
        if county and state:
            county_id = f"county_{county.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}"
            county_node = await self._create_location_node(county_id, "County", {"name": county, "state": state})
            if county_node:
                results["nodes_created"] += 1
                results["location_ids"].append(county_id)

                # County -> State relationship
                relationship_created = await self._create_relationship(
                    from_node={"label": "Location", "id_field": "location_id", "id_value": county_id},
                    to_node={"label": "Location", "id_field": "location_id", "id_value": state_id},
                    relationship_type="PART_OF",
                    properties={"hierarchy_level": "county_to_state"},
                )
                if relationship_created:
                    results["relationships_created"] += 1

        # Create city node and relationship to county
        if city and county and state:
            city_id = f"city_{city.lower().replace(' ', '_')}_{county.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}"
            city_node = await self._create_location_node(
                city_id,
                "City",
                {
                    "name": city,
                    "county": county,
                    "state": state,
                    "latitude": location_data.get("latitude"),
                    "longitude": location_data.get("longitude"),
                },
            )
            if city_node:
                results["nodes_created"] += 1
                results["location_ids"].append(city_id)

                # City -> County relationship
                relationship_created = await self._create_relationship(
                    from_node={"label": "Location", "id_field": "location_id", "id_value": city_id},
                    to_node={"label": "Location", "id_field": "location_id", "id_value": county_id},
                    relationship_type="PART_OF",
                    properties={"hierarchy_level": "city_to_county"},
                )
                if relationship_created:
                    results["relationships_created"] += 1

        return results

    async def _create_price_segment_relationships(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create price segment relationships for properties with comprehensive validation."""

        # Validate and get price segment
        is_valid, price_segment_id, warnings = self._validate_and_segment_property(property_data)

        if not is_valid:
            return {
                "error": "Invalid price data",
                "warnings": warnings,
                "property_id": property_data.get("id", "unknown"),
            }

        if price_segment_id == "unknown":
            return {
                "skipped": "Price validation not applicable",
                "warnings": warnings,
                "property_id": property_data.get("id", "unknown"),
            }

        # Extract segment name from ID
        segment_name = price_segment_id.split("_")[0].upper()
        segment_info = self.price_segments.get(segment_name, self.price_segments["LUXURY"])

        # Create/update price segment node
        segment_node = await self._create_price_segment_node(
            price_segment_id, segment_name, segment_info["min"], segment_info["max"]
        )

        # Create relationship
        if segment_node:
            price = property_data.get("price", 0)
            relationship_created = await self._create_relationship(
                from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                to_node={"label": "PriceSegment", "id_field": "segment_id", "id_value": price_segment_id},
                relationship_type="BELONGS_TO_SEGMENT",
                properties={
                    "price": price,
                    "segment_name": segment_name,
                    "price_percentile": self._calculate_price_percentile(price, segment_info),
                    "validation_warnings": warnings,
                },
            )

            if relationship_created:
                return {"segment": segment_name, "segment_id": price_segment_id, "success": True, "warnings": warnings}

        return {"error": "Failed to create price segment relationship", "warnings": warnings}

    async def _create_rental_relationships(
        self, property_data: Dict[str, Any], market_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create rental-specific relationships for rental properties."""
        results = {
            "rental_office_created": False,
            "rental_manager_created": False,
            "rental_agreement_created": False,
            "rental_status_created": False,
            "warnings": [],
        }

        # Check if this is actually a rental property
        property_type = self._determine_property_type(property_data)
        if property_type not in ["rental", "lease"]:
            results["warnings"].append(f"Property type '{property_type}' not a rental")
            return results

        # Create rental office relationship if office info available
        office_info = self._extract_office_info(property_data)
        if office_info and office_info.get("name"):
            office_node = await self._create_office_node(property_data)

            if office_node:
                relationship_created = await self._create_relationship(
                    from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                    to_node={"label": "ListingOffice", "id_field": "office_name", "id_value": office_info["name"]},
                    relationship_type="MANAGED_BY",
                    properties={
                        "management_type": "rental",
                        "contact_phone": office_info.get("phone"),
                        "contact_email": office_info.get("email"),
                    },
                )
                results["rental_office_created"] = relationship_created

        # Create rental manager relationship if available
        manager_info = property_data.get("property_manager") or property_data.get("contact")
        if manager_info and isinstance(manager_info, dict):
            manager_name = manager_info.get("name") or manager_info.get("contact_name")
            if manager_name:
                # First create the agent node for the property manager
                manager_property_data = {
                    "id": property_data["id"],  # Use same ID for agent creation
                    "listing_agent": {
                        "name": manager_name,
                        "phone": manager_info.get("phone"),
                        "email": manager_info.get("email"),
                    },
                }
                agent_node = await self._create_agent_node(manager_property_data)

                if agent_node:
                    relationship_created = await self._create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                        to_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_node["agent_id"]},
                        relationship_type="MANAGED_BY",
                        properties={
                            "manager_type": "property_manager",
                            "phone": manager_info.get("phone"),
                            "email": manager_info.get("email"),
                            "role": "rental_manager",
                        },
                    )
                    results["rental_manager_created"] = relationship_created

        # Create rental status relationship
        rental_status = property_data.get("status", "available").lower()
        rental_terms = property_data.get("rental_terms", {})

        relationship_created = await self._create_relationship(
            from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
            to_node={"label": "PropertyType", "id_field": "type_name", "id_value": "Rental"},
            relationship_type="HAS_RENTAL_STATUS",
            properties={
                "status": rental_status,
                "lease_term": rental_terms.get("lease_term"),
                "deposit_required": rental_terms.get("deposit"),
                "pets_allowed": rental_terms.get("pets_allowed"),
                "utilities_included": rental_terms.get("utilities_included"),
                "available_date": property_data.get("available_date"),
                "rental_price": property_data.get("price") or property_data.get("rent"),
            },
        )
        results["rental_status_created"] = relationship_created

        return results

    def _calculate_price_percentile(self, price: float, segment_info: Dict[str, Any]) -> float:
        """Calculate price percentile within segment."""
        min_price = segment_info["min"]
        max_price = segment_info["max"]

        if max_price == float("inf"):
            # For luxury segment, use a reasonable upper bound for calculation
            max_price = min_price * 3

        range_size = max_price - min_price
        if range_size > 0:
            percentile = ((price - min_price) / range_size) * 100
            return min(100, max(0, percentile))
        return 50  # Default to middle percentile

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

    async def build_integrated_graph(
        self, property_data: Dict[str, Any], market_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build integrated graph with property and market data relationships."""

        results = {
            "property_nodes": 0,
            "market_nodes": 0,
            "location_nodes": 0,
            "relationships": 0,
            "price_segments": 0,
            "property_types": 0,
            "market_context": 0,
            "success": True,
            "enhancements": {},
            "errors": [],
        }

        try:
            # 1. Create property node (using existing method)
            property_node = await self._create_property_node(property_data)
            if property_node:
                results["property_nodes"] += 1

                # 2. Build geographic hierarchy
                geo_result = await self._build_geographic_hierarchy(property_data)
                results["location_nodes"] += geo_result["nodes_created"]
                results["relationships"] += geo_result["relationships_created"]

                if geo_result["nodes_created"] > 0:
                    results["enhancements"] = results.get("enhancements", {})
                    results["enhancements"]["geographic_hierarchy"] = True

                # 2.5. Create price segment for market data with validation
                for market in market_data:

                    # Validate market data pricing
                    is_valid, price_segment_id, warnings = self._validate_and_segment_market_data(market)

                    if is_valid and price_segment_id != "unknown":
                        segment_name = price_segment_id.split("_")[0].upper()
                        segment_info = self.price_segments.get(segment_name, self.price_segments["LUXURY"])

                        # Create/update price segment node for market data
                        segment_node = await self._create_price_segment_node(
                            price_segment_id, segment_name, segment_info["min"], segment_info["max"]
                        )

                        if segment_node:
                            results["price_segments"] += 1
                            results["enhancements"]["price_segmentation"] = True

                            # Log any warnings
                            if warnings:
                                results["enhancements"]["validation_warnings"] = warnings
                # 3. Create property type relationships using smart detection
                property_type = self._determine_property_type(property_data)
                type_node = await self._create_property_type_node(property_type)
                if type_node:
                    results["property_types"] += 1
                    results["enhancements"] = results.get("enhancements", {})
                    results["enhancements"]["property_type"] = True

                    # Create property -> property type relationship
                    relationship_created = await self._create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                        to_node={"label": "PropertyType", "id_field": "type_id", "id_value": type_node["type_id"]},
                        relationship_type="HAS_TYPE",
                        properties={"property_type": property_type},
                    )
                    if relationship_created:
                        results["relationships"] += 1

                # 3.5. Create listing type relationships for rental/sale identification
                listing_type = self._determine_listing_type_from_data(property_data)
                if listing_type:
                    listing_type_node = await self._create_listing_type_node(listing_type)
                    if listing_type_node:
                        results["enhancements"] = results.get("enhancements", {})
                        results["enhancements"]["listing_type"] = True

                        # Create property -> listing type relationship
                        relationship_created = await self._create_relationship(
                            from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                            to_node={
                                "label": "ListingType",
                                "id_field": "type_id",
                                "id_value": listing_type_node["type_id"],
                            },
                            relationship_type="HAS_LISTING_TYPE",
                            properties={
                                "listing_type": listing_type,
                                "category": listing_type_node["category"],
                                "is_rental": listing_type_node["is_rental"],
                            },
                        )
                        if relationship_created:
                            results["relationships"] += 1

                # 4. Create agent node and relationship
                self.logger.info(f"Creating agent node for {property_data['id']}")
                agent_node = await self._create_agent_node(property_data)
                if agent_node:
                    await self._create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                        to_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_node["agent_id"]},
                        relationship_type="LISTED_BY",
                        properties={},
                    )
                    results["relationships"] += 1

                # 5. Create office relationships (for both sales and rentals)
                office_node = await self._create_office_node(property_data)
                if office_node:
                    results["enhancements"] = results.get("enhancements", {})
                    results["enhancements"]["listing_office"] = True

                    # Create property -> office relationship
                    relationship_created = await self._create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                        to_node={"label": "Office", "id_field": "office_id", "id_value": office_node["office_id"]},
                        relationship_type="LISTED_BY_OFFICE",
                        properties={
                            "office_name": office_node["name"],
                            "identification_method": office_node["identification_method"],
                        },
                    )
                    if relationship_created:
                        results["relationships"] += 1

                    # Create agent -> office relationship if both exist
                    if agent_node and office_node:
                        await self._create_relationship(
                            from_node={"label": "Agent", "id_field": "agent_id", "id_value": agent_node["agent_id"]},
                            to_node={"label": "Office", "id_field": "office_id", "id_value": office_node["office_id"]},
                            relationship_type="WORKS_AT",
                            properties={},
                        )
                        results["relationships"] += 1

                # 6. Create rental-specific relationships if this is a rental property
                listing_type_determined = self._determine_listing_type_from_data(property_data)
                if self._is_rental_listing_type(listing_type_determined):
                    rental_result = await self._create_rental_relationships(property_data, market_data)
                    if rental_result.get("success"):
                        results["enhancements"] = results.get("enhancements", {})
                        results["enhancements"]["rental_relationships"] = True
                        results["relationships"] += rental_result.get("relationships_created", 0)

                # 7. Connect property to city location
                # 8. Create price segment relationships
                segment_result = await self._create_price_segment_relationships(property_data)
                if segment_result.get("success"):
                    results["enhancements"] = results.get("enhancements", {})
                    results["enhancements"]["price_segmentation"] = True

                # 9. Connect property to city location
                city = property_data.get("city")
                county = property_data.get("county")
                state = property_data.get("state")

                if city and county and state:
                    city_id = f"city_{city.lower().replace(' ', '_')}_{county.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}"
                    relationship_created = await self._create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_data["id"]},
                        to_node={"label": "Location", "id_field": "location_id", "id_value": city_id},
                        relationship_type="LOCATED_IN",
                        properties={
                            "address": property_data.get("formattedAddress"),
                            "latitude": property_data.get("latitude"),
                            "longitude": property_data.get("longitude"),
                        },
                    )
                    if relationship_created:
                        results["relationships"] += 1

            else:
                results["errors"].append("Failed to create property node")
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Error building integrated graph: {str(e)}")
            self.logger.error(f"Error in build_integrated_graph: {e}")

        return results

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
            city = property_data.get("city")
            state = property_data.get("state")
            if city and state:
                location_id = f"{city.lower().replace(' ', '_')}_{state.lower()}"
                location_node = await self._create_location_node(location_id, "City", property_data)
                if location_node:
                    await self._create_relationship(
                        from_node={"label": "Property", "id_field": "property_id", "id_value": property_id},
                        to_node={"label": "Location", "id_field": "location_id", "id_value": location_id},
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
            await self._create_market_data_node(market_data, market_data_id)

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
        """Create a listing office node in Neo4j with smart office detection."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized")

        # Extract office attributes - check both camelCase and snake_case
        listing_office = property_data.get("listing_office", {}) or property_data.get("listingOffice", {})
        if not listing_office:
            self.logger.debug("No listing office information available")
            return {}

        # Smart office identification - use name, email domain, or website
        office_name = listing_office.get("name", "").strip()
        office_email = listing_office.get("email", "").strip()
        office_website = listing_office.get("website", "").strip()
        office_phone = listing_office.get("phone", "").strip()

        # Determine office identifier and name
        if office_name:
            office_identifier = office_name
            display_name = office_name
        elif office_email and "@" in office_email:
            # Use email domain as office identifier
            domain = office_email.split("@")[1]
            office_identifier = domain
            display_name = domain.replace(".", " ").title() + " (via email)"
        elif office_website:
            # Use website domain as office identifier
            domain = office_website.replace("http://", "").replace("https://", "").replace("www.", "").split("/")[0]
            office_identifier = domain
            display_name = domain.replace(".", " ").title() + " (via website)"
        elif office_phone:
            # Use phone as last resort
            office_identifier = f"office_phone_{office_phone}"
            display_name = f"Office ({office_phone})"
        else:
            self.logger.debug("No identifiable office information available")
            return {}

        # Create office ID from identifier
        office_id = re.sub(r"[^a-z0-9]", "_", office_identifier.lower())

        node_properties = {
            "office_id": office_id,
            "name": display_name,
            "original_name": office_name if office_name else None,
            "phone": office_phone if office_phone else None,
            "email": office_email if office_email else None,
            "website": office_website if office_website else None,
            "identification_method": (
                "name" if office_name else ("email" if office_email else ("website" if office_website else "phone"))
            ),
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
