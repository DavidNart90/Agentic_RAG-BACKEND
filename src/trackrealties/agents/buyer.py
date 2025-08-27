"""
Buyer-specific agent for the TrackRealties AI Platform.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from .base import AgentDependencies, BaseAgent, BaseTool
from .prompts import (BASE_SYSTEM_CONTEXT, BUYER_SYSTEM_PROMPT,
                      get_adaptive_analysis_prompt,
                      get_context_enhanced_prompt)
from .tools import (MarketAnalysisTool, PropertyRecommendationTool,
                    VectorSearchTool)


class BuyerAgent(BaseAgent):
    """An agent specialized in assisting home buyers with context-aware property recommendations."""

    MODEL_PATH = "models/buyer_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("buyer") if role_models else None
        tools = [
            VectorSearchTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="buyer_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{BUYER_SYSTEM_PROMPT}"

    async def generate_context_aware_response(self, query: str, session_id: str, **kwargs) -> str:
        """
        Generate context-aware response for home buyers using stored preferences and search history.

        This method provides personalized property recommendations based on user preferences,
        search patterns, and available market data.
        """
        try:
            # Get stored context from session
            context = await self.get_session_context(session_id)

            # Extract buyer-specific context
            buyer_context = self._extract_buyer_context(context)

            # Determine data availability for adaptive analysis
            data_availability = self._assess_data_availability(buyer_context)

            # Build enhanced prompt with buyer context
            enhanced_prompt = self._build_buyer_context_prompt(
                query=query,
                buyer_context=buyer_context,
                conversation_history=context.messages[-5:] if context else [],
                data_availability=data_availability,
            )

            # Generate response using enhanced prompt
            response = await self.llm_client.generate_response(enhanced_prompt, **kwargs)

            return response

        except Exception as e:
            print(f"Error generating context-aware response: {e}")
            # Fallback to standard response generation
            return await super().generate_response(query, session_id, **kwargs)

    def _extract_buyer_context(self, context) -> Dict[str, Any]:
        """Extract buyer-specific data from session context."""
        if not context or not hasattr(context, "metadata"):
            return {}

        buyer_data = {}
        metadata = context.metadata

        # Extract user preferences
        if hasattr(context, "user_preferences") and context.user_preferences:
            buyer_data["preferences"] = context.user_preferences

        # Extract market analysis for buyer's area of interest
        if "market_analysis" in metadata:
            market = metadata["market_analysis"]
            buyer_data["market_analysis"] = {
                "location": market.get("location"),
                "trend_direction": market.get("market_trends", {}).get("trend_direction"),
                "affordability_index": market.get("market_metrics", {}).get("affordability_index"),
                "inventory_levels": market.get("market_metrics", {}).get("inventory_levels"),
                "days_on_market": market.get("market_metrics", {}).get("average_days_on_market"),
            }

        # Extract property search patterns
        if hasattr(context, "search_history") and context.search_history:
            recent_searches = context.search_history[-5:]
            buyer_data["search_patterns"] = [
                {
                    "query": search.query_text,
                    "location": getattr(search, "location", None),
                    "price_range": getattr(search, "price_range", None),
                    "property_type": getattr(search, "property_type", None),
                }
                for search in recent_searches
            ]

        # Extract validation results for property recommendations
        if hasattr(context, "validation_results") and context.validation_results:
            recent_validations = context.validation_results[-3:]
            buyer_data["validation_history"] = [
                {
                    "validation_type": validation.validation_type,
                    "confidence": validation.confidence,
                    "recommendations": validation.recommendations,
                }
                for validation in recent_validations
            ]

        return buyer_data

    def _assess_data_availability(self, buyer_context: Dict[str, Any]) -> str:
        """Assess the richness of available buyer context data."""
        data_points = 0

        if buyer_context.get("preferences"):
            data_points += 2
        if buyer_context.get("market_analysis", {}).get("location"):
            data_points += 2
        if buyer_context.get("search_patterns"):
            data_points += 1
        if buyer_context.get("validation_history"):
            data_points += 1

        if data_points >= 4:
            return "rich"
        elif data_points >= 2:
            return "mixed"
        else:
            return "limited"

    def _build_buyer_context_prompt(
        self, query: str, buyer_context: Dict[str, Any], conversation_history: List, data_availability: str
    ) -> str:
        """Build context-aware prompt for buyer agent."""

        # Start with enhanced base prompt
        base_prompt = get_context_enhanced_prompt(self.get_role_specific_prompt(), buyer_context)

        # Add adaptive analysis instructions
        adaptive_instructions = get_adaptive_analysis_prompt(data_availability)

        # Build context sections
        context_sections = []

        if buyer_context.get("preferences"):
            prefs = buyer_context["preferences"]
            context_sections.append(
                f"""
**🏡 BUYER PREFERENCES AVAILABLE:**
- Budget/Price Range: {prefs.get('budget', 'Not specified')}
- Preferred Locations: {prefs.get('preferred_locations', 'Not specified')}
- Property Type: {prefs.get('property_type', 'Not specified')}
- Bedrooms/Bathrooms: {prefs.get('bedrooms', 'N/A')}/{prefs.get('bathrooms', 'N/A')}
- Special Requirements: {prefs.get('special_requirements', 'None specified')}
            """
            )

        if buyer_context.get("market_analysis"):
            market = buyer_context["market_analysis"]
            context_sections.append(
                f"""
**📈 MARKET CONDITIONS AVAILABLE:**
- Target Location: {market.get('location', 'N/A')}
- Market Trend: {market.get('trend_direction', 'N/A')}
- Affordability Index: {market.get('affordability_index', 'N/A')}
- Inventory Levels: {market.get('inventory_levels', 'N/A')}
- Average Days on Market: {market.get('days_on_market', 'N/A')}
            """
            )

        if buyer_context.get("search_patterns"):
            patterns = buyer_context["search_patterns"]
            recent_patterns = patterns[-3:] if len(patterns) > 3 else patterns
            context_sections.append(
                f"""
**🔍 RECENT SEARCH PATTERNS:**
{chr(10).join([f"- {pattern.get('query', 'N/A')} in {pattern.get('location', 'N/A')}" for pattern in recent_patterns])}
            """
            )

        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]
            conversation_context = f"""
**💬 RECENT CONVERSATION:**
{chr(10).join([f"- {msg.role}: {msg.content[:100]}..." for msg in recent_messages])}
            """

        return f"""
{base_prompt}

{adaptive_instructions}

**🎯 USER QUERY:** {query}

**📋 AVAILABLE BUYER CONTEXT:**
{chr(10).join(context_sections) if context_sections else "No stored buyer preferences available - provide general home buying guidance."}

{conversation_context}

**🎯 RESPONSE INSTRUCTIONS:**
1. **Personalize Recommendations**: Use buyer preferences and search patterns for targeted suggestions
2. **Reference Market Conditions**: Incorporate current market data for timing and negotiation advice
3. **Address Lifestyle Needs**: Consider both financial and lifestyle factors in recommendations
4. **Provide Actionable Steps**: Include specific next steps for the home buying process
5. **Use Buyer-Friendly Language**: Explain complex real estate concepts in accessible terms

**📊 RESPONSE FORMAT:**
🏡 **Perfect Matches** (curated property recommendations based on preferences)
🌟 **Neighborhood Spotlight** (area analysis and lifestyle fit)
💰 **Financial Picture** (affordability, monthly costs, financing options)
🔍 **Property Deep Dive** (detailed analysis of top recommendations)
📋 **Buying Game Plan** (strategy, timeline, negotiation approach)
✅ **Action Items** (immediate next steps and preparations)

Generate a personalized home buying response that leverages all available buyer context and preferences.
        """

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of tools available to the buyer agent."""
        return [VectorSearchTool(deps=deps), PropertyRecommendationTool(deps=deps), MarketAnalysisTool(deps=deps)]
