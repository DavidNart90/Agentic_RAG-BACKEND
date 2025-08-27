"""
Enhanced Semantic Chunking Strategy for TrackRealties
With improved property data extraction
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import tiktoken


@dataclass
class ChunkMetadata:
    """Enhanced metadata for chunks with semantic context"""

    chunk_type: str
    semantic_level: str  # 'property_overview', 'location_details', 'financial_details'
    entity_types: List[str]  # ['property', 'location', 'agent', 'price']
    parent_context: str
    chunk_relationships: List[str]  # References to related chunks
    content_density: float  # Information density score
    token_count: int
    semantic_score: float  # How semantically coherent the chunk is


class EnhancedSemanticChunker:
    """
    Advanced chunking with semantic awareness and relationship mapping
    """

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Semantic field importance weights
        self.field_weights = {
            "high_importance": ["price", "address", "bedrooms", "bathrooms", "square_footage"],
            "medium_importance": ["description", "property_type", "listing_agent", "school_district"],
            "low_importance": ["listing_id", "last_updated", "mls_number"],
        }

        # Entity extraction patterns
        self.entity_patterns = {
            "price": r"\$[\d,]+",
            "address": r"\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)",
            "room_count": r"\d+\s*(?:bed|bedroom|bath|bathroom)",
            "area": r"\d+\s*(?:sq|square)\s*(?:ft|feet)",
            "year": r"19\d{2}|20\d{2}",
        }

    def chunk_with_semantic_awareness(self, data: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """
        Create semantically aware chunks with relationship mapping
        """
        # Normalize data type to handle variations
        normalized_type = data_type.lower()

        if normalized_type in ["property_listing", "property"]:
            return self._chunk_property_semantically(data)
        elif normalized_type in ["market_data", "market"]:
            return self._chunk_market_data_semantically(data)
        else:
            # For truly generic data, use the generic chunker
            return self._chunk_generic_semantically(data, data_type)

    def _chunk_property_semantically(self, property_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for property listings with enhanced context"""
        chunks = []

        # 1. Property Core Chunk (High Priority) - changed from property_overview
        overview_content = self._extract_property_overview(property_data)
        if overview_content:
            chunks.append(
                {
                    "content": overview_content,
                    "metadata": {
                        "chunk_type": "property_core",  # Changed to match DB constraint
                        "semantic_level": "primary",
                        "entity_types": ["property", "price", "listing"],
                        "parent_context": property_data.get("id", property_data.get("formattedAddress", "unknown")),
                        "semantic_score": self._calculate_semantic_score(overview_content, "property_core"),
                        "content_density": self._calculate_density(overview_content),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(overview_content)),
                        "extracted_entities": self._extract_property_entities(property_data),
                        "source": "property_listing",
                    },
                }
            )

        # 2. Location Context Chunk - changed from location_details
        location_content = self._extract_location_details(property_data)
        if location_content:
            chunks.append(
                {
                    "content": location_content,
                    "metadata": {
                        "chunk_type": "location_context",  # Already matches
                        "semantic_level": "secondary",
                        "entity_types": ["location", "address", "county", "coordinates"],
                        "parent_context": property_data.get("id", property_data.get("formattedAddress", "unknown")),
                        "semantic_score": self._calculate_semantic_score(location_content, "location"),
                        "content_density": self._calculate_density(location_content),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(location_content)),
                        "extracted_entities": self._extract_location_entities(property_data),
                        "source": "property_listing",
                    },
                }
            )

        # 3. Features & Amenities Chunk - changed from property_specifications
        specs_content = self._extract_property_specifications(property_data)
        if specs_content:
            chunks.append(
                {
                    "content": specs_content,
                    "metadata": {
                        "chunk_type": "features_amenities",  # Changed to match DB constraint
                        "semantic_level": "secondary",
                        "entity_types": ["features", "specifications", "rooms"],
                        "parent_context": property_data.get("id", property_data.get("formattedAddress", "unknown")),
                        "semantic_score": self._calculate_semantic_score(specs_content, "features"),
                        "content_density": self._calculate_density(specs_content),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(specs_content)),
                        "extracted_entities": self._extract_specification_entities(property_data),
                        "source": "property_listing",
                    },
                }
            )

        # 4. Financial Analysis Chunk - changed from financial_market_info
        financial_content = self._extract_financial_market_info(property_data)
        if financial_content:
            chunks.append(
                {
                    "content": financial_content,
                    "metadata": {
                        "chunk_type": "financial_analysis",  # Changed to match DB constraint
                        "semantic_level": "primary",
                        "entity_types": ["price", "market", "financial"],
                        "parent_context": property_data.get("id", property_data.get("formattedAddress", "unknown")),
                        "semantic_score": self._calculate_semantic_score(financial_content, "financial"),
                        "content_density": self._calculate_density(financial_content),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(financial_content)),
                        "extracted_entities": self._extract_financial_entities(property_data),
                        "source": "property_listing",
                    },
                }
            )

        # 5. Agent Info Chunk - changed from agent_office_info
        agent_content = self._extract_agent_office_info(property_data)
        if agent_content:
            chunks.append(
                {
                    "content": agent_content,
                    "metadata": {
                        "chunk_type": "agent_info",  # Changed to match DB constraint
                        "semantic_level": "tertiary",
                        "entity_types": ["agent", "office", "contact"],
                        "parent_context": property_data.get("id", property_data.get("formattedAddress", "unknown")),
                        "semantic_score": self._calculate_semantic_score(agent_content, "agent"),
                        "content_density": self._calculate_density(agent_content),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(agent_content)),
                        "extracted_entities": self._extract_agent_entities(property_data),
                        "source": "property_listing",
                    },
                }
            )

        # 6. General Chunk for Property History - changed from property_history
        history_content = self._extract_property_history(property_data)
        if history_content:
            chunks.append(
                {
                    "content": history_content,
                    "metadata": {
                        "chunk_type": "general",  # Changed to use 'general' for history
                        "semantic_level": "secondary",
                        "entity_types": ["history", "events", "timeline"],
                        "parent_context": property_data.get("id", property_data.get("formattedAddress", "unknown")),
                        "semantic_score": self._calculate_semantic_score(history_content, "history"),
                        "content_density": self._calculate_density(history_content),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(history_content)),
                        "extracted_entities": self._extract_history_entities(property_data),
                        "source": "property_listing",
                    },
                }
            )

        return chunks

    def _extract_property_overview(self, data: Dict[str, Any]) -> str:
        """Extract property overview information"""
        content = "PROPERTY OVERVIEW:\n"

        # Property ID
        if "id" in data:
            content += f"Property ID: {data['id']}\n"

        # Address
        if "formattedAddress" in data:
            content += f"Address: {data['formattedAddress']}\n"

        # Property Type
        if "propertyType" in data:
            content += f"Property Type: {data['propertyType']}\n"

        # Status
        if "status" in data:
            content += f"Status: {data['status']}\n"

        # Listing Type
        if "listingType" in data:
            content += f"Listing Type: {data['listingType']}\n"

        # MLS Information
        if "mlsName" in data:
            content += f"MLS Name: {data['mlsName']}\n"
        if "mlsNumber" in data:
            content += f"MLS Number: {data['mlsNumber']}\n"

        return content

    def _extract_location_details(self, data: Dict[str, Any]) -> str:
        """Extract detailed location information"""
        content = "LOCATION DETAILS:\n"

        # Address components
        if "addressLine1" in data:
            content += f"Address Line 1: {data['addressLine1']}\n"
        if "addressLine2" in data and data["addressLine2"]:
            content += f"Address Line 2: {data['addressLine2']}\n"

        # City, State, Zip
        if "city" in data:
            content += f"City: {data['city']}\n"
        if "state" in data:
            content += f"State: {data['state']}\n"
        if "zipCode" in data:
            content += f"ZIP Code: {data['zipCode']}\n"

        # County
        if "county" in data:
            content += f"County: {data['county']}\n"

        # Coordinates
        if "latitude" in data and "longitude" in data:
            content += f"Coordinates: {data['latitude']}, {data['longitude']}\n"
            content += f"Latitude: {data['latitude']}\n"
            content += f"Longitude: {data['longitude']}\n"

        return content

    def _extract_property_specifications(self, data: Dict[str, Any]) -> str:
        """Extract property specifications"""
        content = "PROPERTY SPECIFICATIONS:\n"

        # Bedrooms and Bathrooms
        if "bedrooms" in data:
            content += f"Bedrooms: {data['bedrooms']}\n"
        if "bathrooms" in data:
            content += f"Bathrooms: {data['bathrooms']}\n"

        # Square Footage
        if "squareFootage" in data:
            content += f"Square Footage: {data['squareFootage']:,} sq ft\n"

        # Lot Size
        if "lotSize" in data:
            lot_size = data["lotSize"]
            if isinstance(lot_size, (int, float)):
                content += f"Lot Size: {lot_size:,} sq ft\n"
                # Also show in acres if large enough
                if lot_size >= 43560:
                    acres = lot_size / 43560
                    content += f"Lot Size (Acres): {acres:.2f} acres\n"

        # Year Built
        if "yearBuilt" in data:
            content += f"Year Built: {data['yearBuilt']}\n"
            # Calculate age
            current_year = datetime.now().year
            age = current_year - data["yearBuilt"]
            content += f"Property Age: {age} years\n"

        return content

    def _extract_financial_market_info(self, data: Dict[str, Any]) -> str:
        """Extract financial and market information"""
        content = "FINANCIAL & MARKET INFORMATION:\n"

        # Price
        if "price" in data:
            price = data["price"]
            content += f"List Price: ${price:,.2f}\n"

            # Calculate price per square foot if possible
            if "squareFootage" in data and data["squareFootage"] > 0:
                price_per_sqft = price / data["squareFootage"]
                content += f"Price per Sq Ft: ${price_per_sqft:.2f}\n"

        # Market timing
        if "listedDate" in data:
            content += f"Listed Date: {data['listedDate']}\n"
        if "createdDate" in data:
            content += f"Created Date: {data['createdDate']}\n"
        if "lastSeenDate" in data:
            content += f"Last Seen Date: {data['lastSeenDate']}\n"
        if "removedDate" in data and data["removedDate"]:
            content += f"Removed Date: {data['removedDate']}\n"

        # Days on Market
        if "daysOnMarket" in data:
            content += f"Days on Market: {data['daysOnMarket']}\n"

        return content

    def _extract_agent_office_info(self, data: Dict[str, Any]) -> str:
        """Extract listing agent and office information"""
        content = "AGENT & OFFICE INFORMATION:\n"

        # Listing Agent
        if "listingAgent" in data and data["listingAgent"]:
            agent = data["listingAgent"]
            content += "\nListing Agent:\n"

            if "name" in agent:
                content += f"- Name: {agent['name']}\n"
            if "phone" in agent:
                content += f"- Phone: {agent['phone']}\n"
            if "email" in agent:
                content += f"- Email: {agent['email']}\n"
            if "website" in agent:
                content += f"- Website: {agent['website']}\n"

        # Listing Office
        if "listingOffice" in data and data["listingOffice"]:
            office = data["listingOffice"]
            content += "\nListing Office:\n"

            if "name" in office:
                content += f"- Name: {office['name']}\n"
            if "phone" in office:
                content += f"- Phone: {office['phone']}\n"
            if "email" in office:
                content += f"- Email: {office['email']}\n"
            if "website" in office:
                content += f"- Website: {office['website']}\n"

        return content

    def _extract_property_history(self, data: Dict[str, Any]) -> str:
        """Extract property history information"""
        content = "PROPERTY HISTORY:\n"

        if "history" in data and data["history"]:
            history = data["history"]

            # Sort history by date - convert keys to list
            sorted_dates = sorted(list(history.keys()))

            for date in sorted_dates:
                event_data = history[date]
                content += f"\n{date}:\n"

                if "event" in event_data:
                    content += f"- Event: {event_data['event']}\n"
                if "price" in event_data:
                    content += f"- Price: ${event_data['price']:,.2f}\n"
                if "listingType" in event_data:
                    content += f"- Listing Type: {event_data['listingType']}\n"
                if "daysOnMarket" in event_data:
                    content += f"- Days on Market: {event_data['daysOnMarket']}\n"
                if "listedDate" in event_data:
                    content += f"- Listed Date: {event_data['listedDate']}\n"
                if "removedDate" in event_data and event_data["removedDate"]:
                    content += f"- Removed Date: {event_data['removedDate']}\n"

        return content

    # Enhanced entity extraction methods for properties
    def _extract_property_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract main property entities"""
        entities = {}

        if "id" in data:
            entities["property_id"] = data["id"]
        if "formattedAddress" in data:
            entities["address"] = data["formattedAddress"]
        if "propertyType" in data:
            entities["property_type"] = data["propertyType"]
        if "status" in data:
            entities["status"] = data["status"]
        if "price" in data:
            entities["price"] = data["price"]
        if "mlsNumber" in data:
            entities["mls_number"] = data["mlsNumber"]

        return entities

    def _extract_location_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location entities"""
        entities = {}

        if "city" in data:
            entities["city"] = data["city"]
        if "state" in data:
            entities["state"] = data["state"]
        if "zipCode" in data:
            entities["zip_code"] = data["zipCode"]
        if "county" in data:
            entities["county"] = data["county"]
        if "latitude" in data and "longitude" in data:
            entities["coordinates"] = {"lat": data["latitude"], "lng": data["longitude"]}

        return entities

    def _extract_specification_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specification entities"""
        entities = {}

        if "bedrooms" in data:
            entities["bedrooms"] = data["bedrooms"]
        if "bathrooms" in data:
            entities["bathrooms"] = data["bathrooms"]
        if "squareFootage" in data:
            entities["square_footage"] = data["squareFootage"]
        if "lotSize" in data:
            entities["lot_size"] = data["lotSize"]
        if "yearBuilt" in data:
            entities["year_built"] = data["yearBuilt"]

        return entities

    def _extract_financial_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial entities"""
        entities = {}

        if "price" in data:
            entities["list_price"] = data["price"]
            if "squareFootage" in data and data["squareFootage"] > 0:
                entities["price_per_sqft"] = data["price"] / data["squareFootage"]
        if "daysOnMarket" in data:
            entities["days_on_market"] = data["daysOnMarket"]
        if "listedDate" in data:
            entities["listed_date"] = data["listedDate"]

        return entities

    def _extract_agent_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent and office entities"""
        entities = {}

        if "listingAgent" in data and data["listingAgent"]:
            entities["agent_name"] = data["listingAgent"].get("name")
            entities["agent_phone"] = data["listingAgent"].get("phone")
            entities["agent_email"] = data["listingAgent"].get("email")

        if "listingOffice" in data and data["listingOffice"]:
            entities["office_name"] = data["listingOffice"].get("name")
            entities["office_phone"] = data["listingOffice"].get("phone")

        return entities

    def _extract_history_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract history entities"""
        entities = {}

        if "history" in data and data["history"]:
            entities["event_count"] = len(data["history"])
            entities["events"] = []

            for date, event in data["history"].items():
                entities["events"].append({"date": date, "event_type": event.get("event"), "price": event.get("price")})

        return entities

    # Keep all the market data methods unchanged
    def _chunk_market_data_semantically(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks for market data with enhanced context"""
        chunks = []
        # 1. Market Overview Chunk (High Priority)
        overview_content = self._extract_market_overview(market_data)
        if overview_content:
            chunks.append(
                {
                    "content": overview_content,
                    "metadata": {
                        "chunk_type": "market_overview",
                        "semantic_level": "primary",
                        "entity_types": ["location", "price", "volume"],
                        "source": market_data.get("source", "unknown"),
                        "parent_context": market_data.get("region_id", market_data.get("location", "unknown")),
                        "semantic_score": self._calculate_semantic_score(overview_content, "market_overview"),
                        "chunk_created_at": datetime.now().isoformat(),
                        "content_density": self._calculate_density(overview_content),
                        "token_count": len(self.tokenizer.encode(overview_content)),
                        "extracted_entities": self._extract_market_entities(market_data),
                    },
                }
            )

        # 2. Geographic Location Chunk
        geographic_content = self._extract_geographic_location(market_data)
        if geographic_content:
            chunks.append(
                {
                    "content": geographic_content,
                    "metadata": {
                        "chunk_type": "geographic_location",
                        "semantic_level": "secondary",
                        "entity_types": ["location", "geography", "coordinates"],
                        "parent_context": market_data.get("region_id", market_data.get("location", "unknown")),
                        "source": market_data.get("source", "unknown"),
                        "semantic_score": self._calculate_semantic_score(geographic_content, "geographic"),
                        "chunk_created_at": datetime.now().isoformat(),
                        "content_density": self._calculate_density(geographic_content),
                        "token_count": len(self.tokenizer.encode(geographic_content)),
                        "extracted_entities": self._extract_geographic_entities(market_data),
                    },
                }
            )

        # 3. Price Trends Chunk
        price_content = self._extract_price_trends(market_data)
        if price_content:
            chunks.append(
                {
                    "content": price_content,
                    "metadata": {
                        "chunk_type": "price_trends",
                        "semantic_level": "secondary",
                        "entity_types": ["price", "trend", "metrics"],
                        "source": market_data.get("source", "unknown"),
                        "parent_context": market_data.get("region_id", market_data.get("location", "unknown")),
                        "semantic_score": self._calculate_semantic_score(price_content, "price_trends"),
                        "chunk_created_at": datetime.now().isoformat(),
                        "content_density": self._calculate_density(price_content),
                        "token_count": len(self.tokenizer.encode(price_content)),
                        "extracted_entities": self._extract_price_entities(market_data),
                    },
                }
            )

        # 4. Inventory Analysis Chunk
        inventory_content = self._extract_inventory_analysis(market_data)
        if inventory_content:
            chunks.append(
                {
                    "content": inventory_content,
                    "metadata": {
                        "chunk_type": "inventory_analysis",
                        "semantic_level": "tertiary",
                        "entity_types": ["inventory", "listing", "metrics"],
                        "parent_context": market_data.get("region_id", market_data.get("location", "unknown")),
                        "source": market_data.get("source", "unknown"),
                        "semantic_score": self._calculate_semantic_score(inventory_content, "inventory_analysis"),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(inventory_content)),
                        "content_density": self._calculate_density(inventory_content),
                        "extracted_entities": self._extract_inventory_entities(market_data),
                    },
                }
            )

        return chunks

    # Keep all other methods unchanged (market data methods, calculation methods, etc.)
    def _extract_market_overview(self, data: Dict[str, Any]) -> str:
        """Extract market overview information"""
        content = "MARKET OVERVIEW:\n"

        # Location info
        location = data.get("location", data.get("region_name", "Unknown Location"))
        content += f"Location: {location}\n"

        # Date info
        date = data.get("date", data.get("data_date", ""))
        if date:
            content += f"Date: {date}\n"

        # Core metrics
        if "median_home_price" in data or "median_price" in data:
            price = data.get("median_home_price", data.get("median_price"))
            content += f"Median Price: ${price:,.2f}\n"

        # Sales data
        if "homes_sold" in data or "sales_volume" in data:
            sales = data.get("homes_sold", data.get("sales_volume"))
            content += f"Sales Volume: {sales}\n"

        return content

    def _extract_price_trends(self, data: Dict[str, Any]) -> str:
        """Extract price trend information"""
        content = "PRICE TRENDS:\n"

        # Price per sqft
        if "price_per_sqft_median" in data or "price_per_sqft" in data:
            price_sqft = data.get("price_per_sqft_median", data.get("price_per_sqft"))
            content += f"Price Per Sqft: ${price_sqft:.2f}\n"

        # Price changes
        if "price_change_monthly" in data:
            monthly = data["price_change_monthly"] * 100
            content += f"Monthly Change: {monthly:+.1f}%\n"

        if "price_change_yearly" in data:
            yearly = data["price_change_yearly"] * 100
            content += f"Yearly Change: {yearly:+.1f}%\n"

        return content

    def _extract_inventory_analysis(self, data: Dict[str, Any]) -> str:
        """Extract inventory analysis information"""
        content = "INVENTORY ANALYSIS:\n"

        # Inventory count
        if "inventory_count" in data:
            content += f"Inventory Count: {data['inventory_count']}\n"

        # Days on market
        if "days_on_market_median" in data or "days_on_market" in data:
            dom = data.get("days_on_market_median", data.get("days_on_market"))
            content += f"Days on Market: {dom}\n"

        # Months supply
        if "months_supply" in data:
            content += f"Months Supply: {data['months_supply']}\n"

        # New listings
        if "new_listings" in data:
            content += f"New Listings: {data['new_listings']}\n"

        return content

    def _extract_geographic_location(self, data: Dict[str, Any]) -> str:
        """Extract geographic location information with region and county parsing"""
        content = "GEOGRAPHIC LOCATION:\n"

        # Get location string and parse it
        location = data.get("location", data.get("region_name", ""))

        if location:
            # Parse region and county from location string
            region, county = self._parse_location_string(location)

            if region:
                content += f"Region: {region}\n"
            if county:
                content += f"County: {county}\n"

            content += f"Full Location: {location}\n"

        # Basic location info
        if "city" in data:
            content += f"City: {data['city']}\n"
        if "state" in data:
            content += f"State: {data['state']}\n"

        # Coordinates
        if "latitude" in data and "longitude" in data:
            content += f"Coordinates: {data['latitude']}, {data['longitude']}\n"

        return content

    # Keep all other helper methods unchanged
    def _parse_location_string(self, location: str) -> Tuple[str, str]:
        """Parse region and county from location string like 'Ruston, LA metro area'"""
        region = ""
        county = ""

        if not location:
            return region, county

        # Extract region (city/area name)
        if "," in location:
            parts = location.split(",")
            region = parts[0].strip()

            # Look for county indicators in remaining parts
            remaining = ",".join(parts[1:]).strip()

            # Check for county patterns
            county_patterns = [
                r"(\w+)\s+county",
                r"(\w+)\s+co\.",
                r"(\w+)\s+parish",
            ]

            for pattern in county_patterns:
                match = re.search(pattern, remaining, re.IGNORECASE)
                if match:
                    county = match.group(1).strip()
                    break

            # If no county found but we have state info, try to infer
            if not county and len(parts) >= 2:
                state_part = parts[1].strip()
                # If it's just state code, the region might include county info
                if len(state_part) == 2:
                    # For Louisiana, many areas are parishes
                    if state_part.upper() == "LA":
                        county = f"{region} Parish"
        else:
            region = location

        return region, county

    def _extract_market_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from market data"""
        entities = {}

        if "median_home_price" in data or "median_price" in data:
            entities["median_price"] = data.get("median_home_price", data.get("median_price"))
        if "sales_volume" in data:
            entities["sales_volume"] = data["sales_volume"]
        if "location" in data:
            entities["location"] = data["location"]

        return entities

    def _extract_geographic_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geographic entities"""
        entities = {}

        location = data.get("location", data.get("region_name", ""))
        if location:
            region, county = self._parse_location_string(location)
            if region:
                entities["region"] = region
            if county:
                entities["county"] = county

        if "city" in data:
            entities["city"] = data["city"]
        if "state" in data:
            entities["state"] = data["state"]
        if "latitude" in data and "longitude" in data:
            entities["coordinates"] = f"{data['latitude']}, {data['longitude']}"

        return entities

    def _extract_price_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract price-related entities"""
        entities = {}

        if "price_per_sqft_median" in data or "price_per_sqft" in data:
            entities["price_per_sqft"] = data.get("price_per_sqft_median", data.get("price_per_sqft"))
        if "price_change_monthly" in data:
            entities["monthly_change"] = data["price_change_monthly"]
        if "price_change_yearly" in data:
            entities["yearly_change"] = data["price_change_yearly"]

        return entities

    def _extract_inventory_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract inventory-related entities"""
        entities = {}

        if "inventory_count" in data:
            entities["inventory_count"] = data["inventory_count"]
        if "days_on_market_median" in data or "days_on_market" in data:
            entities["days_on_market"] = data.get("days_on_market_median", data.get("days_on_market"))
        if "months_supply" in data:
            entities["months_supply"] = data["months_supply"]
        if "new_listings" in data:
            entities["new_listings"] = data["new_listings"]

        return entities

    def _calculate_density(self, content: str) -> float:
        """Calculate information density of content"""
        words = content.split()
        unique_words = set(word.lower().strip(".,!?;:") for word in words)

        # Calculate metrics
        word_count = len(words)
        unique_ratio = len(unique_words) / word_count if word_count > 0 else 0

        # Count important entities
        entity_count = sum(1 for pattern in self.entity_patterns.values() if re.search(pattern, content, re.IGNORECASE))

        # Density score (0-1)
        density = min(1.0, (unique_ratio * 0.4) + (entity_count / word_count * 0.6))
        return round(density, 3)

    def _calculate_semantic_score(self, content: str, chunk_type: str) -> float:
        """Calculate semantic coherence score"""
        # Count relevant terms based on chunk type
        relevant_terms = {
            "property_core": ["property", "address", "type", "status", "mls"],
            "location": ["city", "state", "zip", "county", "coordinates"],
            "features": ["bedroom", "bathroom", "sqft", "square", "lot", "year"],
            "financial": ["price", "cost", "market", "listed", "days"],
            "agent": ["agent", "office", "phone", "email", "contact"],
            "history": ["event", "date", "price", "listing", "removed"],
            "market_overview": ["median", "price", "volume", "location", "date"],
            "geographic": ["location", "city", "state", "region", "county"],
            "price_trends": ["price", "change", "monthly", "yearly", "trend"],
            "inventory_analysis": ["inventory", "supply", "listings", "market", "days"],
        }

        terms = relevant_terms.get(chunk_type, [])
        content_lower = content.lower()

        # Calculate relevance score
        found_terms = sum(1 for term in terms if term in content_lower)
        relevance_score = found_terms / len(terms) if terms else 0.5

        # Factor in content length appropriateness
        length_score = min(1.0, len(content) / 500)  # Optimal around 500 chars

        semantic_score = (relevance_score * 0.7) + (length_score * 0.3)
        return round(semantic_score, 3)

    def _chunk_generic_semantically(self, data: Dict[str, Any], data_type: str = "generic") -> List[Dict[str, Any]]:
        """Create semantic chunks for generic data"""
        chunks = []

        # 1. Main Data Chunk
        main_content = self._extract_generic_main_content(data)
        if main_content:
            # Use 'general' instead of 'generic_main' for database constraint compliance
            chunk_type = "general"  # Always use 'general' for generic data

            chunks.append(
                {
                    "content": main_content,
                    "metadata": {
                        "chunk_type": chunk_type,  # Changed from 'generic_main'
                        "semantic_level": "primary",
                        "entity_types": self._extract_entity_types_from_data(data),
                        "parent_context": data.get("id", data.get("identifier", data.get("region_id", "unknown"))),
                        "semantic_score": self._calculate_semantic_score(main_content, "generic"),
                        "chunk_created_at": datetime.now().isoformat(),
                        "token_count": len(self.tokenizer.encode(main_content)),
                        "data_type": data_type,
                        "data_keys": list(data.keys()) if isinstance(data, dict) else [],
                        "source": data.get("source", "unknown"),
                    },
                }
            )

        # 2. If data is large, create additional chunks for complex nested objects
        if isinstance(data, dict) and len(str(data)) > self.max_chunk_size:
            nested_chunks = self._chunk_large_generic_data(data, data_type)
            chunks.extend(nested_chunks)

        return chunks

    def _extract_generic_main_content(self, data: Dict[str, Any]) -> str:
        """Extract main content from generic data"""
        content = "DATA SUMMARY:\n"

        # Handle the data based on its structure
        if isinstance(data, dict):
            # Prioritize important fields first
            important_fields = [
                "id",
                "name",
                "title",
                "description",
                "type",
                "status",
                "date",
                "created_at",
                "updated_at",
            ]

            # Add important fields first
            for field in important_fields:
                if field in data and data[field] is not None:
                    formatted_field = field.replace("_", " ").title()
                    value = data[field]

                    # Format specific field types
                    if "date" in field.lower() or "time" in field.lower():
                        content += f"{formatted_field}: {value}\n"
                    elif isinstance(value, (int, float)) and any(
                        keyword in field.lower() for keyword in ["price", "cost", "amount", "fee"]
                    ):
                        content += f"{formatted_field}: ${value:,.2f}\n"
                    elif isinstance(value, str) and len(value) > 100:
                        # Truncate long text fields
                        content += f"{formatted_field}: {value[:100]}...\n"
                    else:
                        content += f"{formatted_field}: {value}\n"

            # Add other fields (excluding the ones already processed)
            remaining_fields = [k for k in data.keys() if k not in important_fields]
            for field in remaining_fields[:10]:  # Limit to first 10 remaining fields
                if data[field] is not None:
                    formatted_field = field.replace("_", " ").title()
                    value = data[field]

                    # Skip complex nested objects for main content
                    if isinstance(value, (dict, list)):
                        if isinstance(value, list):
                            content += f"{formatted_field}: {len(value)} items\n"
                        else:
                            content += f"{formatted_field}: {len(value)} properties\n"
                    else:
                        # Handle simple values
                        if isinstance(value, str) and len(value) > 50:
                            content += f"{formatted_field}: {value[:50]}...\n"
                        else:
                            content += f"{formatted_field}: {value}\n"

        return content

    def _chunk_large_generic_data(self, data: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """Create additional chunks for large generic data with nested objects"""
        chunks = []

        # Look for complex nested objects that deserve their own chunks
        for key, value in data.items():
            if isinstance(value, dict) and len(str(value)) > 200:
                # Create a chunk for this nested object
                nested_content = self._format_nested_object(key, value)

                chunks.append(
                    {
                        "content": nested_content,
                        "metadata": {
                            "chunk_type": "general",  # Changed from 'generic_nested'
                            "semantic_level": "secondary",
                            "entity_types": [key.lower(), "nested_data"],
                            "parent_context": data.get("id", data.get("identifier", data.get("region_id", "unknown"))),
                            "nested_field": key,
                            "semantic_score": self._calculate_semantic_score(nested_content, "nested"),
                            "chunk_created_at": datetime.now().isoformat(),
                            "token_count": len(self.tokenizer.encode(nested_content)),
                            "data_type": data_type,
                            "source": data.get("source", "unknown"),
                        },
                    }
                )

            elif isinstance(value, list) and len(value) > 5:
                # Create a chunk for large lists
                list_content = self._format_list_data(key, value)

                chunks.append(
                    {
                        "content": list_content,
                        "metadata": {
                            "chunk_type": "general",  # Changed from 'generic_list'
                            "semantic_level": "secondary",
                            "entity_types": [key.lower(), "list_data"],
                            "parent_context": data.get("id", data.get("identifier", data.get("region_id", "unknown"))),
                            "list_field": key,
                            "list_size": len(value),
                            "semantic_score": self._calculate_semantic_score(list_content, "list"),
                            "chunk_created_at": datetime.now().isoformat(),
                            "token_count": len(self.tokenizer.encode(list_content)),
                            "data_type": data_type,
                            "source": data.get("source", "unknown"),
                        },
                    }
                )

        return chunks

    def _format_nested_object(self, key: str, obj: Dict[str, Any]) -> str:
        """Format a nested object for chunking"""
        content = f"{key.replace('_', ' ').title().upper()}:\n\n"

        for sub_key, sub_value in obj.items():
            formatted_key = sub_key.replace("_", " ").title()

            if isinstance(sub_value, (dict, list)):
                if isinstance(sub_value, list):
                    content += f"{formatted_key}: {len(sub_value)} items\n"
                else:
                    content += f"{formatted_key}: {len(sub_value)} properties\n"
            elif isinstance(sub_value, str) and len(sub_value) > 100:
                content += f"{formatted_key}: {sub_value[:100]}...\n"
            else:
                content += f"{formatted_key}: {sub_value}\n"

        return content

    def _format_list_data(self, key: str, lst: List[Any]) -> str:
        """Format list data for chunking"""
        content = f"{key.replace('_', ' ').title().upper()} ({len(lst)} items):\n\n"

        # Show first few items
        for i, item in enumerate(lst[:5]):
            if isinstance(item, dict):
                # For dict items, show key info
                content += f"Item {i + 1}:\n"
                for item_key, item_value in list(item.items())[:3]:  # First 3 keys
                    formatted_key = item_key.replace("_", " ").title()
                    content += f"  {formatted_key}: {item_value}\n"
            else:
                content += f"Item {i + 1}: {item}\n"

        if len(lst) > 5:
            content += f"... and {len(lst) - 5} more items\n"

        return content

    def _extract_entity_types_from_data(self, data: Dict[str, Any]) -> List[str]:
        """Extract entity types from generic data based on field names"""
        entity_types = ["data"]

        # Look for common entity indicators in field names
        field_indicators = {
            "user": ["user", "person", "customer", "client"],
            "location": ["address", "city", "state", "country", "location", "geo"],
            "financial": ["price", "cost", "amount", "fee", "payment", "revenue"],
            "temporal": ["date", "time", "created", "updated", "timestamp"],
            "contact": ["email", "phone", "contact", "website"],
            "product": ["product", "item", "service", "offering"],
            "organization": ["company", "organization", "business", "agency"],
        }

        if isinstance(data, dict):
            field_names = [key.lower() for key in data.keys()]

            for entity_type, indicators in field_indicators.items():
                if any(indicator in field_name for field_name in field_names for indicator in indicators):
                    entity_types.append(entity_type)

        return entity_types
