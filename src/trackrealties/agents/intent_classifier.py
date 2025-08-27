"""
LLM-Based Intent Classification System for TrackRealties
Production-ready implementation integrated with existing architecture.
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

from ..core.config import settings

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Real estate intent types for TrackRealties."""

    PURE_GREETING = "pure_greeting"
    GREETING_WITH_VAGUE_HELP = "greeting_with_vague_help"
    GREETING_WITH_QUERY = "greeting_with_query"
    INVESTMENT_ANALYSIS = "investment_analysis"
    PROPERTY_SEARCH = "property_search"
    MARKET_RESEARCH = "market_research"
    FINANCING_INQUIRY = "financing_inquiry"
    NEIGHBORHOOD_RESEARCH = "neighborhood_research"
    AFFORDABILITY_CHECK = "affordability_check"
    VIEWING_REQUEST = "viewing_request"
    OFFER_GUIDANCE = "offer_guidance"
    LISTING_OPTIMIZATION = "listing_optimization"
    LEAD_MANAGEMENT = "lead_management"
    CLIENT_MATCHING = "client_matching"
    MARKET_INTELLIGENCE = "market_intelligence"
    BUSINESS_DEVELOPMENT = "business_development"
    SITE_ANALYSIS = "site_analysis"
    FEASIBILITY_STUDY = "feasibility_study"
    ZONING_INQUIRY = "zoning_inquiry"
    CONSTRUCTION_PLANNING = "construction_planning"
    PERMIT_GUIDANCE = "permit_guidance"
    GENERAL_QUESTION = "general_question"
    FOLLOW_UP = "follow_up"


class ClassificationResult(BaseModel):
    """Structured classification result."""

    intent: str
    confidence: float
    user_role_match: float
    requires_search: bool
    reasoning: str
    entities: Dict[str, List[str]]
    keywords: List[str]
    processing_time_ms: int
    source: str  # 'llm', 'cache', 'fallback'


@dataclass
class CacheEntry:
    """Cache entry for classification results."""

    result: ClassificationResult
    timestamp: datetime
    hit_count: int = 0


class TrackRealtiesIntentClassifier:
    """
    Production-ready LLM intent classifier for TrackRealties.
    Integrates with existing architecture and dependencies.
    """

    def __init__(self, cache_ttl_hours: int = 24, enable_cache: bool = True):
        """Initialize with TrackRealties configuration."""
        # Use TrackRealties settings for model configuration
        self.model = getattr(settings, "INTENT_CLASSIFIER_MODEL", "openai:gpt-4o-mini")

        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.enable_cache = enable_cache
        self.cache: Dict[str, CacheEntry] = {}

        # Performance metrics
        self.metrics = {
            "total_classifications": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_uses": 0,
            "avg_response_time": 0,
        }

        self.classification_prompt = self._build_classification_prompt()

    def _build_classification_prompt(self) -> str:
        """Build the classification prompt for TrackRealties."""
        return """You are an expert intent classifier for TrackRealties, a real estate AI platform.

Your job is to analyze user messages and classify their intent with high accuracy.

REAL ESTATE INTENTS:
- pure_greeting: Simple greetings only ("Hi", "Hello", "Good morning")
- greeting_with_vague_help: Greeting + vague help request ("Hi, I need help with real estate", "Hello, looking for assistance")
- greeting_with_query: Greeting + specific business question ("Hi, I want to invest in apartments", "Hello, find me properties under $300k")
- investment_analysis: ROI analysis, cap rates, cash flow, investment evaluation
- property_search: Looking for properties to buy/rent ("Find me a 3-bedroom house")
- market_research: Market trends, analysis, forecasts, data requests
- financing_inquiry: Loans, mortgages, financing options, rates
- neighborhood_research: Schools, safety, amenities, walkability
- affordability_check: "What can I afford", budget calculations
- viewing_request: Schedule tours, showings, open houses
- offer_guidance: Making offers, negotiation help, contract advice
- listing_optimization: Pricing strategy, marketing, staging (for agents)
- lead_management: Lead generation, conversion, tracking (for agents)
- client_matching: Matching buyers with properties (for agents)
- market_intelligence: Competitive analysis, business insights (for agents)
- business_development: Growing real estate business (for agents)
- site_analysis: Land evaluation, development potential (for developers)
- feasibility_study: Project viability, financial modeling (for developers)
- zoning_inquiry: Zoning rules, permits, regulations (for developers)
- construction_planning: Timeline, costs, contractors (for developers)
- permit_guidance: Building permits, applications (for developers)
- general_question: Simple real estate questions
- follow_up: Continuation of previous conversation

USER ROLES:
- investor: Focused on ROI, cash flow, investment properties
- buyer: Looking for personal residence, first-time buyers
- agent: Real estate professionals growing their business
- developer: Building/developing properties

CLASSIFICATION RULES:
1. If message contains BOTH greeting AND specific business request, classify as "greeting_with_query"
2. If message contains greeting + vague help request ("help with real estate", "assistance", "looking for help"), classify as "greeting_with_vague_help"
3. Look for the PRIMARY business intent, not just keywords
4. confidence: 0.9+ for obvious cases, 0.7+ for clear cases, 0.5+ for uncertain
5. requires_search: true for specific business queries, false for greetings and vague help requests
6. Extract ALL relevant entities, even if implied

SPECIAL CASE EXAMPLES:
- "Hi Agent, I want to invest in 300 unit apartments in Texas, TX." 
  -> intent: "greeting_with_query" (specific request)
- "Hello, what are cap rates in Austin?"
  -> intent: "greeting_with_query" (specific question)
- "Hello! I'm looking for help with real estate"
  -> intent: "greeting_with_vague_help" (vague help request, NOT specific)
- "Hi, I need assistance with property investment"
  -> intent: "greeting_with_vague_help" (vague assistance request)
- "Find me properties in Dallas"
  -> intent: "property_search" (specific request, no greeting)

USER ROLE: {user_role}
MESSAGE: "{message}"

Respond with ONLY a valid JSON object:
{{
    "intent": "intent_name",
    "confidence": 0.95,
    "user_role_match": 0.85,
    "requires_search": true,
    "reasoning": "Brief explanation",
    "entities": {{"locations": ["Texas"], "property_specs": ["300 unit", "apartments"]}},
    "keywords": ["invest", "apartment", "units"]
}}"""

    def _get_cache_key(self, message: str, user_role: str) -> str:
        """Generate cache key for message and role."""
        content = f"{message.strip().lower()}|{user_role or 'none'}"
        return hashlib.md5(content.encode()).hexdigest()

    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = [key for key, entry in self.cache.items() if now - entry.timestamp > self.cache_ttl]
        for key in expired_keys:
            del self.cache[key]

    async def classify_intent(self, message: str, user_role: Optional[str] = None) -> ClassificationResult:
        """
        Classify user intent with caching and fallback.

        Args:
            message: User message to classify
            user_role: User's role (investor, buyer, agent, developer)

        Returns:
            ClassificationResult with intent and metadata
        """
        start_time = time.time()
        self.metrics["total_classifications"] += 1

        # Check cache first
        if self.enable_cache:
            cache_key = self._get_cache_key(message, user_role or "")
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.hit_count += 1
                self.metrics["cache_hits"] += 1

                # Update processing time and source
                result = entry.result.model_copy()
                result.processing_time_ms = int((time.time() - start_time) * 1000)
                result.source = "cache"

                logger.debug(f"Cache hit for message: {message[:50]}...")
                return result

        try:
            # Clean expired cache entries periodically
            if len(self.cache) > 1000:
                self._clean_cache()

            # Use LLM for classification
            result = await self._classify_with_llm(message, user_role, start_time)

            # Cache the result
            if self.enable_cache:
                cache_entry = CacheEntry(result=result, timestamp=datetime.now())
                self.cache[cache_key] = cache_entry

            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            self.metrics["fallback_uses"] += 1

            # Fallback to simple heuristics
            return self._fallback_classification(message, user_role, start_time)

    async def _classify_with_llm(
        self, message: str, user_role: Optional[str], start_time: float
    ) -> ClassificationResult:
        """Classify using LLM."""
        self.metrics["llm_calls"] += 1

        # Create classification agent
        classifier_agent = PydanticAI(
            model=self.model,
            system_prompt=self.classification_prompt.format(user_role=user_role or "unknown", message=message),
        )

        # Get classification
        llm_result = await classifier_agent.run(message)

        try:
            # Parse JSON response
            classification_data = json.loads(str(llm_result.output))

            # Ensure all entity values are lists
            entities = classification_data.get("entities", {})
            normalized_entities = {}
            for key, value in entities.items():
                if isinstance(value, list):
                    normalized_entities[key] = value
                else:
                    # Convert single values to lists
                    normalized_entities[key] = [str(value)]

            # Create result
            result = ClassificationResult(
                intent=classification_data.get("intent", "general_question"),
                confidence=min(max(classification_data.get("confidence", 0.5), 0.0), 1.0),
                user_role_match=min(max(classification_data.get("user_role_match", 0.5), 0.0), 1.0),
                requires_search=classification_data.get("requires_search", True),
                reasoning=classification_data.get("reasoning", "LLM classification"),
                entities=normalized_entities,
                keywords=classification_data.get("keywords", []),
                processing_time_ms=int((time.time() - start_time) * 1000),
                source="llm",
            )

            logger.debug(f"LLM classified '{message[:50]}...' as {result.intent} ({result.confidence:.2f})")
            return result

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_classification(message, user_role, start_time)

    def _fallback_classification(
        self, message: str, user_role: Optional[str], start_time: float
    ) -> ClassificationResult:
        """Simple fallback classification when LLM fails."""
        message_lower = message.strip().lower()

        # Check for greeting + business content pattern
        has_greeting = any(greeting in message_lower for greeting in ["hi", "hello", "hey", "good morning"])

        # Vague help patterns
        vague_help_patterns = ["help with real estate", "assistance", "looking for help", "need help", "help me with"]
        has_vague_help = any(pattern in message_lower for pattern in vague_help_patterns)

        # Specific business keywords
        specific_business = any(
            keyword in message_lower
            for keyword in [
                "find",
                "search",
                "buy",
                "sell",
                "analyze",
                "calculate",
                "cap rate",
                "roi",
                "price",
                "budget",
                "properties under",
                "apartments in",
                "houses in",
            ]
        )

        if has_greeting and has_vague_help and not specific_business:
            return ClassificationResult(
                intent="greeting_with_vague_help",
                confidence=0.8,
                user_role_match=1.0,
                requires_search=False,
                reasoning="Fallback: Greeting with vague help request detected",
                entities={},
                keywords=[kw for kw in ["help", "assistance", "real estate"] if kw in message_lower],
                processing_time_ms=int((time.time() - start_time) * 1000),
                source="fallback",
            )

        if has_greeting and specific_business:
            return ClassificationResult(
                intent="greeting_with_query",
                confidence=0.8,
                user_role_match=1.0,
                requires_search=True,
                reasoning="Fallback: Greeting with specific business content detected",
                entities={},
                keywords=[kw for kw in ["invest", "property", "buy", "sell", "find"] if kw in message_lower],
                processing_time_ms=int((time.time() - start_time) * 1000),
                source="fallback",
            )

        # Simple greeting detection
        if len(message.split()) <= 3 and has_greeting:
            return ClassificationResult(
                intent="pure_greeting",
                confidence=0.9,
                user_role_match=1.0,
                requires_search=False,
                reasoning="Fallback: Simple greeting detected",
                entities={},
                keywords=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                source="fallback",
            )

        # Default classification
        return ClassificationResult(
            intent="general_question",
            confidence=0.6,
            user_role_match=0.7,
            requires_search=True,
            reasoning="Fallback: Default classification",
            entities={},
            keywords=[],
            processing_time_ms=int((time.time() - start_time) * 1000),
            source="fallback",
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        total = self.metrics["total_classifications"]
        if total > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total * 100
            self.metrics["cache_hit_rate"] = cache_hit_rate

        return self.metrics.copy()
