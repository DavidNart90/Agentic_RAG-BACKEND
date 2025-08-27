"""
LLM-Based Entity Extraction System for TrackRealties
Production-ready implementation following the same pattern as intent_classifier.py
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic_ai.agent import Agent as PydanticAI

from ..core.config import (ENTITY_CACHE_ENABLED, ENTITY_CACHE_TTL_HOURS,
                           ENTITY_EXTRACTOR_MODEL, settings)

logger = logging.getLogger(__name__)


class EntityExtractionResult(BaseModel):
    """Structured entity extraction result."""

    locations: List[str]
    properties: List[str]
    property_specifications: List[str]
    metrics: List[str]
    agents: List[str]
    prices: List[str]
    property_types: List[str]
    timeframes: List[str]
    confidence: float
    reasoning: str
    processing_time_ms: int
    source: str  # 'llm', 'cache', 'fallback'


@dataclass
class EntityCacheEntry:
    """Cache entry for entity extraction results."""

    result: EntityExtractionResult
    timestamp: datetime
    hit_count: int = 0


class LLMEntityExtractor:
    """
    Production-ready LLM entity extractor for TrackRealties.
    Integrates with existing architecture following intent_classifier pattern.
    """

    def __init__(self, cache_ttl_hours: int = None, enable_cache: bool = None):
        """Initialize with TrackRealties configuration."""
        # Use TrackRealties settings for model configuration
        self.model = ENTITY_EXTRACTOR_MODEL

        # Use config defaults if not specified
        cache_ttl_hours = cache_ttl_hours or ENTITY_CACHE_TTL_HOURS
        enable_cache = enable_cache if enable_cache is not None else ENTITY_CACHE_ENABLED

        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.enable_cache = enable_cache
        self.cache: Dict[str, EntityCacheEntry] = {}

        # Performance metrics
        self.metrics = {
            "total_extractions": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_uses": 0,
            "avg_response_time": 0,
        }

        self.extraction_prompt = self._build_extraction_prompt()

    def _build_extraction_prompt(self) -> str:
        """Build the entity extraction prompt for TrackRealities."""
        return """You are an expert real estate entity extractor for TrackRealities, a real estate AI platform.

Your job is to extract structured entities from real estate queries with high accuracy and context awareness.

ENTITY TYPES TO EXTRACT:

1. LOCATIONS:
   - Cities with states: "Austin, TX", "San Francisco, CA", "San Antonio, TX"
   - Counties: "Macon County, IL", "Travis County, TX", "Bexar County, TX", "Cook County, IL"
   - States: "Texas", "California", "Illinois", "Florida" 
   - ZIP codes: "78701", "90210", "60601", "33101"
   - Metro areas: "San Francisco metro area", "Dallas-Fort Worth", "Greater Austin Area"
   - Neighborhoods: "Downtown Austin", "SoMa San Francisco", "River Oaks Houston"
   - Regions: "North Dallas", "West Austin", "South Bay", "Central Texas"
   - Market areas: "MLS area", "school district", "suburb", "urban core"
   - GEOGRAPHIC FLEXIBILITY: Accept various location formats and regional terms

2. PROPERTIES:
   - Street addresses: "123 Main St", "333 Florida St", "1234 Oak Avenue"
   - Property IDs: "ABC123", "PROP-456", "listing-789"
   - MLS numbers: "1851140", "MLS-789", "NTREIS-12345"
   - Property names: "Sunset Villa", "Oak Creek Apartments", "Downtown Lofts"
   - Building identifiers: "Building A", "Unit 202", "Phase 1"
   - Legal descriptions: "Lot 15 Block 3", "Tract 45", "Section 21"
   - PROPERTY FLEXIBILITY: Accept various property identification formats

3. PROPERTY_SPECIFICATIONS:
   - Bedrooms: "2-bedroom", "3 bedrooms", "2 bed", "3br", "2-bed", "4 bedroom apartment"
   - Bathrooms: "2-bathroom", "2.5 bath", "3 baths", "2.5ba", "full bath", "half bath"
   - Square footage: "1500 sqft", "2000 square feet", "1200 sq ft", "3000SF"
   - Lot size: "0.25 acres", "10,000 sqft lot", "quarter acre", "5000 sq ft lot"
   - Garage: "2-car garage", "attached garage", "detached garage", "parking"
   - Stories: "2-story", "single story", "3-level", "ranch style"
   - Year built: "built in 2020", "2015 construction", "new construction"
   - Condition: "move-in ready", "fixer-upper", "renovated", "updated"
   - Features: "pool", "fireplace", "hardwood floors", "granite counters", "stainless appliances"
   - SPECIFICATION FLEXIBILITY: Accept natural language property descriptions

4. METRICS:
   - Financial metrics: "median price", "average price", "price per sqft", "list price", "sale price"
   - Market metrics: "days on market", "inventory count", "active listings", "sales volume", "new listings"
   - Investment metrics: "ROI", "cap rate", "cash flow", "rental yield", "IRR", "cash on cash return"
   - Performance metrics: "appreciation rate", "price trends", "market velocity", "absorption rate"
   - Supply metrics: "months supply", "inventory", "listing count", "available units"
   - Risk metrics: "market volatility", "vacancy rate", "turnover rate", "default rate"
   - Comparative metrics: "price comparison", "market comparison", "neighborhood analysis"
   - NATURAL LANGUAGE FLEXIBILITY: Accept real-world variations and industry terms
     * "median home price" → "median price"
     * "average days on market" → "days on market" 
     * "price per square foot" → "price per sqft"
     * "return on investment" → "ROI"
     * "capitalization rate" → "cap rate"
     * "cash-on-cash" → "cash on cash return"

5. AGENTS:
   - Agent names: "John Smith", "Jane Doe", "Maria Rodriguez", "David Chen"
   - Agent titles: "realtor", "broker", "listing agent", "buyer's agent"
   - Team names: "Smith Team", "Rodriguez Group", "Premier Realty Team"
   - Brokerage names: "Keller Williams", "RE/MAX", "Coldwell Banker", "JPAR"
   - License numbers: "TX-123456", "CA-987654"
   - AGENT FLEXIBILITY: Accept various agent identification formats and titles
   - Must be properly capitalized names (not "john smith")

6. PRICES:
   - Dollar amounts: "$500K", "$1.2M", "$750,000", "$2,500/month"
   - Price ranges: "$200K - $500K", "$1M to $2M", "between $300K and $500K"
   - Price qualifiers: "under $500K", "over $1M", "around $750K", "approximately $600K"
   - Monthly rates: "$2,500/mo", "$3,200 per month", "$1,800 monthly"
   - Price per unit: "$200/sqft", "$150 per square foot", "$25/sqft/year"
   - Percentage changes: "10% increase", "5% appreciation", "down 3%"
   - PRICE FLEXIBILITY: Accept various price formats and expressions

7. PROPERTY_TYPES:
   - RESIDENTIAL: "single family", "multi family", "condo", "apartment", "townhouse", "duplex"
   - INVESTMENT: "rental property", "investment property", "income property", "buy and hold"
   - COMMERCIAL: "commercial", "office", "retail", "industrial", "warehouse", "mixed use"
   - LAND: "land", "lot", "acreage", "vacant land", "development land"
   - SPECIALIZED: "manufactured", "mobile home", "luxury", "waterfront", "new construction"
   - RENTAL CATEGORIES: "rental listings", "rental properties", "rental apartments", "rental homes", "lease properties"
   - MARKET SEGMENTS: "starter homes", "luxury homes", "affordable housing", "senior housing"
   - REAL-WORLD FLEXIBILITY: Accept natural language and industry terms
     * "single-family home" → "single family"
     * "multi-family property" → "multi family"
     * "apartment complex" → "apartment"
     * "for rent" → "rental listings"
     * "investment real estate" → "investment property"
     * "income-producing property" → "income property"
   - RENTAL DETECTION: If query contains "rent", "rental", "lease", "monthly", ALWAYS include appropriate rental types

8. TIMEFRAMES:
   - Years: "2024", "2025", "2023"
   - Quarters: "Q1 2024", "Q4 2025", "first quarter"
   - Months: "January 2025", "March", "last month", "this month"
   - Relative time: "current", "latest", "recent", "YTD", "year-to-date"
   - Periods: "last 3 months", "past year", "next 6 months", "historical"
   - Market cycles: "pre-pandemic", "post-COVID", "recession period"
   - TIMEFRAME FLEXIBILITY: Accept various temporal expressions and market periods

EXTRACTION RULES:

1. REAL-WORLD CONTEXT AWARENESS:
   - "for condos in Manhattan, NY" → location: ["Manhattan, NY"], property_types: ["condo"]
   - "median price in Austin" → location: ["Austin"], metrics: ["median price"]
   - "Tell me about Dallas market" → location: ["Dallas"], NOT location: ["Tell, ME"]
   - "investment properties under $500K" → prices: ["under $500K"], property_types: ["investment property"]
   - "cash flow analysis for rental" → metrics: ["cash flow"], property_types: ["rental property"]

2. NATURAL LANGUAGE UNDERSTANDING:
   - Accept industry jargon: "cap rate", "NOI", "GRM", "price per door"
   - Understand colloquialisms: "fixer-upper", "move-in ready", "tear-down"
   - Recognize abbreviations: "SF" (single family), "MF" (multi family), "REI" (real estate investment)
   - Handle variations: "single-family" = "single family", "multi-unit" = "multi family"

3. INVESTMENT QUERY INTELLIGENCE:
   - Investment indicators: "cash flow", "ROI", "rental income", "buy and hold", "flip"
   - Map investment terms to actual metrics in database
   - For "investment properties" → extract actual property types: ["single family", "multi family", "commercial"]
   - For "cash flow analysis" → metrics: ["cash flow", "rental yield", "cap rate"]

4. RENTAL CONTEXT DETECTION:
   - ANY query with "rent", "rental", "lease", "monthly" MUST include appropriate rental types
   - Examples: "monthly rent under $1000" → property_types: ["rental listings"]
   - Examples: "apartments for rent" → property_types: ["apartment", "rental listings"]
   - Examples: "rental income properties" → property_types: ["rental property", "investment property"]

5. LOCATION NORMALIZATION & INTELLIGENCE:
   - Always use 2-letter state codes: "Texas" → "TX", "California" → "CA"
   - Format as "City, ST": "austin texas" → "Austin, TX"
   - Include "County" for counties: "Travis County, TX" (not just "Travis")
   - Recognize metro areas: "DFW" → "Dallas-Fort Worth", "Bay Area" → "San Francisco metro area"
   - Handle neighborhoods: "Downtown" + context → "Downtown Austin" if Austin mentioned

6. CLEAN & CONTEXTUAL EXTRACTION:
   - Extract ONLY the entity, not surrounding words
   - Skip obvious non-entities (articles, prepositions, etc.)
   - Validate that extracted text makes sense in real estate context
   - Preserve industry-standard formatting and terminology
   - Connect related entities: location + property type + metric combinations

7. CONFIDENCE & ACCURACY SCORING:
   - 0.9+: Very clear entities with strong real estate context
   - 0.7+: Clear entities with good context and industry terms
   - 0.5+: Ambiguous but probable entities with real estate relevance
   - <0.5: Skip uncertain extractions that don't fit real estate domain

EXAMPLES:

Query: "I'm looking for a 2-bedroom apartment in New York, NY with a budget between $100,000 and $400,000"
Extract: {
  "locations": ["New York, NY"],
  "property_types": ["apartment"],
  "property_specifications": ["2-bedroom"],
  "prices": ["between $100,000 and $400,000"],
  "confidence": 0.95
}

Query: "Find 3 bedroom 2 bath single family homes under $500K in Austin, TX"
Extract: {
  "locations": ["Austin, TX"],
  "property_types": ["single family"],
  "property_specifications": ["3 bedroom", "2 bath"],
  "prices": ["under $500K"],
  "confidence": 0.95
}

Query: "What's the median price in Macon County, IL?"
Extract: {
  "locations": ["Macon County, IL"],
  "metrics": ["median price"],
  "confidence": 0.95
}

Query: "Find investment properties under $500K in Austin, TX with good cash flow"
Extract: {
  "locations": ["Austin, TX"],
  "prices": ["under $500K"],
  "property_types": ["investment property"],
  "metrics": ["cash flow"],
  "confidence": 0.95
}

Query: "ROI analysis for rental properties in Travis County, Texas"
Extract: {
  "locations": ["Travis County, TX"],
  "metrics": ["ROI"],
  "property_types": ["rental property", "investment property"],
  "confidence": 0.9
}

Query: "What's the cap rate for multi-family properties in San Antonio?"
Extract: {
  "locations": ["San Antonio, TX"],
  "metrics": ["cap rate"],
  "property_types": ["multi family"],
  "confidence": 0.95
}

Query: "monthly rent under $1000 for apartments"
Extract: {
  "prices": ["under $1000"],
  "property_types": ["apartment", "rental listings"],
  "timeframes": ["monthly"],
  "confidence": 0.95
}

Query: "single family homes for sale by John Smith in Dallas"
Extract: {
  "locations": ["Dallas, TX"],
  "property_types": ["single family"],
  "agents": ["John Smith"],
  "confidence": 0.9
}

Query: "price per square foot trends in luxury condos downtown"
Extract: {
  "metrics": ["price per sqft", "price trends"],
  "property_types": ["luxury", "condo"],
  "locations": ["downtown"],
  "confidence": 0.85
}

Query: "What office does Rj Reyes work for?"
Extract: {
  "agents": ["Rj Reyes"],
  "confidence": 0.8
}

Query: "new construction homes under $400K in Cedar Park"
Extract: {
  "property_types": ["new construction", "single family"],
  "prices": ["under $400K"],
  "locations": ["Cedar Park, TX"],
  "confidence": 0.9
}

Query: "buy and hold properties with 8% cap rate in Houston metro"
Extract: {
  "property_types": ["buy and hold", "investment property"],
  "metrics": ["cap rate"],
  "prices": ["8%"],
  "locations": ["Houston metro"],
  "confidence": 0.85
}

Query: "apartments for rent under $500"
Extract: {
  "prices": ["under $500"],
  "property_types": ["apartment", "rental listings"],
  "confidence": 0.95
}

Query: "rental properties under $300 per month"
Extract: {
  "prices": ["under $300 per month"],
  "property_types": ["rental properties", "rental listings"],
  "confidence": 0.95
}

Query: "Contact agent John Smith about 123 Main St, Dallas TX"
Extract: {
  "locations": ["Dallas, TX"],
  "properties": ["123 Main St"],
  "agents": ["John Smith"],
  "confidence": 0.95
}

Query: "Days on market for condos in Manhattan, NY"
Extract: {
  "locations": ["Manhattan, NY"],
  "metrics": ["days on market"],
  "property_types": ["condos"],
  "confidence": 0.9
}

Query: "Properties listed by JPAR San Antonio office"
Extract: {
  "locations": ["San Antonio, TX"],
  "agents": ["JPAR"],
  "confidence": 0.85
}

Query: "Average price per sqft in San Francisco metro area"
Extract: {
  "locations": ["San Francisco metro area"],
  "metrics": ["price per sqft"],
  "confidence": 0.9
}
Query: "What office does Rj Reyes work for?"
Extract: {
  "agents": ["Rj Reyes"],
  "confidence": 0.8
}


Respond with ONLY a valid JSON object in this format:
{
    "locations": ["list", "of", "locations"],
    "properties": ["list", "of", "properties"],
    "property_specifications": ["list", "of", "property_specifications"],
    "metrics": ["list", "of", "metrics"],
    "agents": ["list", "of", "agents"],
    "prices": ["list", "of", "prices"],
    "property_types": ["list", "of", "property_types"],
    "timeframes": ["list", "of", "timeframes"],
    "confidence": 0.95,
    "reasoning": "Brief explanation of extraction decisions"
}"""

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        content = query.strip().lower()
        return hashlib.md5(content.encode()).hexdigest()

    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = [key for key, entry in self.cache.items() if now - entry.timestamp > self.cache_ttl]
        for key in expired_keys:
            del self.cache[key]

    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities with caching and fallback.

        Args:
            query: Real estate query to extract entities from

        Returns:
            Dictionary with entity types as keys and lists of extracted entities as values
        """
        start_time = time.time()
        self.metrics["total_extractions"] += 1

        # Check cache first
        if self.enable_cache:
            cache_key = self._get_cache_key(query)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.hit_count += 1
                self.metrics["cache_hits"] += 1

                # Update processing time and source
                result = entry.result.model_copy()
                result.processing_time_ms = int((time.time() - start_time) * 1000)
                result.source = "cache"

                logger.debug(f"Cache hit for query: {query[:50]}...")

                # Convert to dictionary format expected by smart_search
                return {
                    "locations": result.locations,
                    "properties": result.properties,
                    "property_specifications": result.property_specifications,
                    "metrics": result.metrics,
                    "agents": result.agents,
                    "prices": result.prices,
                    "property_types": result.property_types,
                    "timeframes": result.timeframes,
                }

        try:
            # Clean expired cache entries periodically
            if len(self.cache) > 1000:
                self._clean_cache()

            # Use LLM for extraction
            result = await self._extract_with_llm(query, start_time)

            # Cache the result
            if self.enable_cache:
                cache_entry = EntityCacheEntry(result=result, timestamp=datetime.now())
                self.cache[cache_key] = cache_entry

            # Convert to dictionary format expected by smart_search
            return {
                "locations": result.locations,
                "properties": result.properties,
                "property_specifications": result.property_specifications,
                "metrics": result.metrics,
                "agents": result.agents,
                "prices": result.prices,
                "property_types": result.property_types,
                "timeframes": result.timeframes,
            }

        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            self.metrics["fallback_uses"] += 1

            # Fallback to simple heuristics
            return await self._fallback_extraction(query, start_time)

    async def _extract_with_llm(self, query: str, start_time: float) -> EntityExtractionResult:
        """Extract entities using LLM."""
        self.metrics["llm_calls"] += 1

        # Create extraction agent
        extractor_agent = PydanticAI(model=self.model, system_prompt=self.extraction_prompt)

        # Get extraction with the query as the user message
        llm_result = await extractor_agent.run(f"Extract entities from this query: {query}")

        try:
            # Get the raw response and clean it
            raw_response = str(llm_result.output).strip()
            logger.debug(f"Raw LLM response: {raw_response}")

            # Try to extract JSON from the response
            json_str = self._extract_json_from_response(raw_response)

            # Parse JSON response
            extraction_data = json.loads(json_str)

            # Create result
            result = EntityExtractionResult(
                locations=extraction_data.get("locations", []),
                properties=extraction_data.get("properties", []),
                property_specifications=extraction_data.get("property_specifications", []),
                metrics=extraction_data.get("metrics", []),
                agents=extraction_data.get("agents", []),
                prices=extraction_data.get("prices", []),
                property_types=extraction_data.get("property_types", []),
                timeframes=extraction_data.get("timeframes", []),
                confidence=min(max(extraction_data.get("confidence", 0.5), 0.0), 1.0),
                reasoning=extraction_data.get("reasoning", "LLM extraction"),
                processing_time_ms=int((time.time() - start_time) * 1000),
                source="llm",
            )

            logger.debug(
                f"LLM extracted from '{query[:50]}...': {len(result.locations)} locations, {len(result.metrics)} metrics"
            )
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response was: {raw_response}")
            return await self._fallback_extraction(query, start_time)

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling various formats."""
        # Remove common prefixes and suffixes
        response = response.strip()

        # Look for JSON object boundaries
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx: end_idx + 1]

        # If no clear JSON boundaries, try the whole response
        return response

    async def _fallback_extraction(self, query: str, start_time: float) -> Dict[str, List[str]]:
        """Simple fallback extraction when LLM fails."""
        # Simple regex-based fallback for critical cases
        import re

        result = {
            "locations": [],
            "properties": [],
            "property_specifications": [],
            "metrics": [],
            "agents": [],
            "prices": [],
            "property_types": [],
            "timeframes": [],
        }

        # Basic location extraction (cities, states, zip codes)
        location_patterns = [
            r"\b([A-Z][a-z]+ (?:County|Co\.?),?\s*[A-Z]{2})\b",  # Counties
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s*([A-Z]{2})\b",  # Cities with states
            r"\b(\d{5})\b",  # ZIP codes
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    result["locations"].extend([m for m in match if m])
                else:
                    result["locations"].append(match)

        # Basic price extraction with enhanced patterns
        price_patterns = [
            r"(\$[\d,]+(?:\.\d{2})?[KMB]?)",  # $300,000 or $300K
            r"(under\s+\$?[\d,]+[KMB]?)",  # under $500000 or under 500000
            r"(over\s+\$?[\d,]+[KMB]?)",  # over $500000 or over 500000
            r"(below\s+\$?[\d,]+[KMB]?)",  # below $500000 or below 500000
            r"(above\s+\$?[\d,]+[KMB]?)",  # above $500000 or above 500000
            r"(less\s+than\s+\$?[\d,]+[KMB]?)",  # less than $500000
            r"(more\s+than\s+\$?[\d,]+[KMB]?)",  # more than $500000
            r"(between\s+\$?[\d,]+[KMB]?\s+(?:and|to)\s+\$?[\d,]+[KMB]?)",  # between amounts
            r"(\$?[\d,]{6,}[KMB]?)",  # Large numbers like 500000 (6+ digits)
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            result["prices"].extend(matches)

        # Basic property specifications extraction
        spec_patterns = [
            r"(\d+[-\s]*(?:bed|bedroom)s?)",  # 2-bedroom, 3 bed, etc.
            r"(\d+(?:\.\d+)?[-\s]*(?:bath|bathroom)s?)",  # 2-bathroom, 2.5 bath, etc.
            r"(\d+[-\s]*(?:br|bdr))",  # 2br, 3bdr
            r"(\d+(?:\.\d+)?[-\s]*(?:ba))",  # 2.5ba
        ]

        for pattern in spec_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            result["property_specifications"].extend(matches)

        # Basic agent name extraction (proper names)
        agent_pattern = r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b"
        agent_matches = re.findall(agent_pattern, query)
        result["agents"].extend(agent_matches)

        logger.warning(f"Used fallback extraction for query: {query[:50]}...")
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        total = self.metrics["total_extractions"]
        if total > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total * 100
            self.metrics["cache_hit_rate"] = cache_hit_rate

        return self.metrics.copy()


# Convenience function for backward compatibility
async def extract_real_estate_entities_llm(query: str) -> Dict[str, List[str]]:
    """
    Convenience function for extracting entities from a real estate query using LLM.

    Args:
        query: Natural language real estate query

    Returns:
        Dictionary with extracted entities by type
    """
    extractor = LLMEntityExtractor()
    return await extractor.extract_entities(query)


# Test function for development
async def test_llm_entity_extraction():
    """Test function to validate LLM entity extraction with sample queries"""

    test_queries = [
        "What's the median price in Macon County, IL?",
        "Find properties under $500K in Austin, TX",
        "ROI analysis for rental properties in Travis County, Texas",
        "Contact agent John Smith about 123 Main St, Dallas TX",
        "Average price per sqft in San Francisco metro area",
        "Properties listed by broker Jane Doe in Chicago",
        "Market trends for single family homes in 78701",
        "Days on market for condos in Manhattan, NY",
    ]

    extractor = LLMEntityExtractor()

    print("🧪 Testing LLM-Based Entity Extraction")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        try:
            entities = await extractor.extract_entities(query)
            print(f"\n{i}. Query: {query}")
            print(f"   Locations: {entities['locations']}")
            print(f"   Metrics: {entities['metrics']}")
            print(f"   Properties: {entities['properties']}")
            print(f"   Agents: {entities['agents']}")
            print(f"   Prices: {entities['prices']}")
            print(f"   Property Types: {entities['property_types']}")
            print(f"   Timeframes: {entities['timeframes']}")
        except Exception as e:
            print(f"\n{i}. Query: {query}")
            print(f"   ERROR: {e}")

    # Print performance metrics
    print("\n📊 Performance Metrics:")
    metrics = extractor.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_llm_entity_extraction())
