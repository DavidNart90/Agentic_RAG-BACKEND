"""
Agent-specific agent for the TrackRealties AI Platform.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import AgentDependencies, BaseAgent, BaseTool
from .prompts import (AGENT_SYSTEM_PROMPT, BASE_SYSTEM_CONTEXT,
                      get_adaptive_analysis_prompt,
                      get_context_enhanced_prompt)
from .tools import (ConstructionCostEstimationTool, FeasibilityAnalysisTool,
                    GraphSearchTool, InvestmentOpportunityAnalysisTool,
                    MarketAnalysisTool, PropertyRecommendationTool,
                    RiskAssessmentTool, ROIProjectionTool, SiteAnalysisTool,
                    VectorSearchTool, ZoningAnalysisTool)


class AgentAgent(BaseAgent):
    """Agent specializing in real estate agent tasks with context-aware market insights."""

    MODEL_PATH = "models/agent_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("agent") if role_models else None

        tools = [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="agent_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{AGENT_SYSTEM_PROMPT}"

    async def generate_context_aware_response(self, query: str, session_id: str, **kwargs) -> str:
        """
        Generate context-aware response for real estate agents using stored market intelligence,
        client data, and business analytics.
        """
        try:
            # Get stored context from session
            context = await self.get_session_context(session_id)

            # Extract agent-specific context
            agent_context = self._extract_agent_context(context)

            # Determine data availability for adaptive analysis
            data_availability = self._assess_data_availability(agent_context)

            # Build enhanced prompt with agent context
            enhanced_prompt = self._build_agent_context_prompt(
                query=query,
                agent_context=agent_context,
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

    def _extract_agent_context(self, context) -> Dict[str, Any]:
        """Extract agent-specific data from session context."""
        if not context or not hasattr(context, "metadata"):
            return {}

        agent_data = {}
        metadata = context.metadata

        # Extract market intelligence
        if "market_analysis" in metadata:
            market = metadata["market_analysis"]
            agent_data["market_intelligence"] = {
                "location": market.get("location"),
                "trend_direction": market.get("market_trends", {}).get("trend_direction"),
                "price_change_percent": market.get("market_trends", {}).get("price_change_percent"),
                "inventory_levels": market.get("market_metrics", {}).get("inventory_levels"),
                "days_on_market": market.get("market_metrics", {}).get("average_days_on_market"),
                "market_health_score": market.get("market_metrics", {}).get("market_health_score"),
            }

        # Extract client analytics (from investment analysis)
        if "current_analysis" in metadata:
            analysis = metadata["current_analysis"]
            agent_data["client_analytics"] = {
                "analysis_type": analysis.get("type"),
                "client_profile": analysis.get("client_profile", "investor"),
                "properties_reviewed": analysis.get("properties_analyzed", 0),
                "avg_budget_range": analysis.get("budget_range"),
                "last_interaction": analysis.get("analysis_timestamp"),
            }

        # Extract property performance data
        if hasattr(context, "search_history") and context.search_history:
            recent_searches = context.search_history[-10:]
            agent_data["property_activity"] = [
                {
                    "property_id": getattr(search, "property_id", None),
                    "query_type": getattr(search, "query_type", "general"),
                    "location": getattr(search, "location", None),
                    "price_range": getattr(search, "price_range", None),
                    "client_interest_level": getattr(search, "relevance_score", 0),
                }
                for search in recent_searches
            ]

        # Extract business intelligence from validation results
        if hasattr(context, "validation_results") and context.validation_results:
            recent_validations = context.validation_results[-5:]
            agent_data["business_intelligence"] = {
                "conversion_indicators": [v.confidence for v in recent_validations],
                "recommendation_success": sum(1 for v in recent_validations if v.confidence > 0.8),
                "client_satisfaction_signals": [v.validation_type for v in recent_validations],
            }

        # Extract competitive intelligence from tools used
        if hasattr(context, "tools_used") and context.tools_used:
            agent_data["competitive_intelligence"] = {
                "tools_utilized": context.tools_used,
                "analysis_depth": len(context.tools_used),
                "service_differentiation": (
                    "comprehensive_analytics" if len(context.tools_used) > 3 else "basic_service"
                ),
            }

        return agent_data

    def _assess_data_availability(self, agent_context: Dict[str, Any]) -> str:
        """Assess the richness of available agent context data."""
        data_points = 0

        if agent_context.get("market_intelligence", {}).get("market_health_score"):
            data_points += 3
        if agent_context.get("client_analytics", {}).get("properties_reviewed", 0) > 0:
            data_points += 2
        if agent_context.get("property_activity"):
            data_points += 2
        if agent_context.get("business_intelligence", {}).get("conversion_indicators"):
            data_points += 1
        if agent_context.get("competitive_intelligence", {}).get("analysis_depth", 0) > 2:
            data_points += 1

        if data_points >= 6:
            return "rich"
        elif data_points >= 3:
            return "mixed"
        else:
            return "limited"

    def _build_agent_context_prompt(
        self, query: str, agent_context: Dict[str, Any], conversation_history: List, data_availability: str
    ) -> str:
        """Build context-aware prompt for real estate agent."""

        # Start with enhanced base prompt
        base_prompt = get_context_enhanced_prompt(self.get_role_specific_prompt(), agent_context)

        # Add adaptive analysis instructions
        adaptive_instructions = get_adaptive_analysis_prompt(data_availability)

        # Build context sections
        context_sections = []

        if agent_context.get("market_intelligence"):
            market = agent_context["market_intelligence"]
            context_sections.append(
                f"""
**📊 MARKET INTELLIGENCE AVAILABLE:**
- Target Market: {market.get('location', 'N/A')}
- Market Trend: {market.get('trend_direction', 'N/A')} ({market.get('price_change_percent', 'N/A')}%)
- Inventory Levels: {market.get('inventory_levels', 'N/A')}
- Average Days on Market: {market.get('days_on_market', 'N/A')}
- Market Health Score: {market.get('market_health_score', 'N/A')}
            """
            )

        if agent_context.get("client_analytics"):
            client = agent_context["client_analytics"]
            context_sections.append(
                f"""
**🎯 CLIENT ANALYTICS AVAILABLE:**
- Client Profile: {client.get('client_profile', 'N/A')}
- Properties Reviewed: {client.get('properties_reviewed', 0)}
- Budget Range: {client.get('avg_budget_range', 'N/A')}
- Last Interaction: {client.get('last_interaction', 'N/A')}
- Analysis Type: {client.get('analysis_type', 'N/A')}
            """
            )

        if agent_context.get("property_activity"):
            activity = agent_context["property_activity"]
            recent_activity = activity[-5:] if len(activity) > 5 else activity
            context_sections.append(
                f"""
**🏡 RECENT PROPERTY ACTIVITY:**
{chr(10).join([f"- {prop.get('query_type', 'N/A')} in {prop.get('location', 'N/A')} (Interest: {prop.get('client_interest_level', 0):.1f})" for prop in recent_activity])}
            """
            )

        if agent_context.get("business_intelligence"):
            business = agent_context["business_intelligence"]
            context_sections.append(
                f"""
**📈 BUSINESS INTELLIGENCE AVAILABLE:**
- Recommendation Success Rate: {business.get('recommendation_success', 0)} successful recommendations
- Average Client Confidence: {sum(business.get('conversion_indicators', [0])) / max(len(business.get('conversion_indicators', [1])), 1):.2f}
- Client Satisfaction Signals: {business.get('client_satisfaction_signals', [])}
            """
            )

        if agent_context.get("competitive_intelligence"):
            competitive = agent_context["competitive_intelligence"]
            context_sections.append(
                f"""
**🏆 COMPETITIVE POSITIONING:**
- Service Level: {competitive.get('service_differentiation', 'N/A')}
- Analysis Tools Used: {competitive.get('tools_utilized', [])}
- Analysis Depth Score: {competitive.get('analysis_depth', 0)}
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

**📋 AVAILABLE AGENT CONTEXT:**
{chr(10).join(context_sections)
            if context_sections else "No stored business intelligence available - provide general real estate agent guidance."}

{conversation_context}

**🎯 RESPONSE INSTRUCTIONS:**
1. **Leverage Market Intelligence**: Use specific market data for competitive insights and client advice
2. **Reference Client Analytics**: Incorporate client interaction patterns and preferences
3. **Provide Business Guidance**: Focus on lead generation, client conversion, and market positioning
4. **Include Marketing Strategy**: Offer specific tactics for property promotion and client acquisition
5. **Use Agent Terminology**: Employ real estate professional language and industry best practices

**📊 RESPONSE FORMAT:**
📊 **Market Intelligence** (current conditions and opportunities)
🎯 **Lead Insights** (prospect analysis and prioritization)
💡 **Marketing Strategy** (pricing, promotion, positioning recommendations)
📈 **Business Opportunities** (growth areas and strategic moves)
🏆 **Competitive Edge** (differentiation and value proposition)
📋 **Action Plan** (immediate priorities and implementation steps)

Generate a comprehensive business intelligence response that leverages all available context and market data.
        """

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of all tools available to the agent."""
        return [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
        ]
