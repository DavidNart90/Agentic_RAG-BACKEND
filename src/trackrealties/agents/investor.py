"""
Investor-specific agent for the TrackRealties AI Platform.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from .base import AgentDependencies, BaseAgent, BaseTool
from .prompts import (BASE_SYSTEM_CONTEXT, INVESTOR_SYSTEM_PROMPT,
                      get_adaptive_analysis_prompt,
                      get_context_enhanced_prompt)
from .tools import (GraphSearchTool, InvestmentOpportunityAnalysisTool,
                    MarketAnalysisTool, RiskAssessmentTool, ROIProjectionTool,
                    VectorSearchTool)


class InvestorAgent(BaseAgent):
    """Agent specializing in real estate investor tasks with context-aware response generation."""

    MODEL_PATH = "models/investor_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("investor") if role_models else None

        tools = [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
        ]
        super().__init__(
            agent_name="investor_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{INVESTOR_SYSTEM_PROMPT}"

    async def generate_context_aware_response(self, query: str, session_id: str, **kwargs) -> str:
        """
        Generate context-aware response using stored analytics and conversation history.

        This method implements the context-aware response generation that was missing
        from the base implementation. It accesses stored analytics data and adapts
        the response based on available context.
        """
        try:
            # Get stored context from session
            context = await self.get_session_context(session_id)

            # Extract analytics context from session metadata
            analytics_context = self._extract_analytics_context(context)

            # Determine data availability for adaptive analysis
            data_availability = self._assess_data_availability(analytics_context)

            # Build enhanced prompt with context
            enhanced_prompt = self._build_investor_context_prompt(
                query=query,
                analytics_context=analytics_context,
                conversation_history=context.messages[-5:] if context else [],
                data_availability=data_availability,
            )

            # Generate response using enhanced prompt
            response = await self.llm_client.generate_response(enhanced_prompt, **kwargs)

            return response

        except Exception:
            # Fallback to standard response generation
            return await super().generate_response(query, session_id, **kwargs)

    def _extract_investor_context(self, context) -> Dict[str, Any]:
        """Extract investor-specific data from session context (alias for compatibility)."""
        return self._extract_analytics_context(context)

    def _extract_analytics_context(self, context) -> Dict[str, Any]:
        """Extract analytics data from session context."""
        if not context or not hasattr(context, "metadata"):
            return {}

        analytics_data = {}
        metadata = context.metadata

        # Extract investment analysis
        if "current_analysis" in metadata:
            analysis = metadata["current_analysis"]
            analytics_data["investment_analysis"] = {
                "type": analysis.get("type"),
                "financial_metrics": analysis.get("financial_metrics", {}),
                "risk_factors": analysis.get("risk_factors", {}),
                "properties_analyzed": analysis.get("properties_analyzed", 0),
                "analysis_timestamp": analysis.get("analysis_timestamp"),
            }

        # Extract market analysis
        if "market_analysis" in metadata:
            market = metadata["market_analysis"]
            analytics_data["market_analysis"] = {
                "location": market.get("location"),
                "trend_direction": market.get("market_trends", {}).get("trend_direction"),
                "volatility_score": market.get("market_metrics", {}).get("volatility_score"),
                "forecast": market.get("market_metrics", {}).get("12_month_forecast"),
                "market_health_score": market.get("market_metrics", {}).get("market_health_score"),
            }

        # Extract ROI analysis
        if "roi_analysis" in metadata:
            roi = metadata["roi_analysis"]
            analytics_data["roi_analysis"] = {
                "annual_roi": roi.get("annual_roi"),
                "monthly_cash_flow": roi.get("monthly_cash_flow"),
                "confidence_score": roi.get("confidence_score"),
                "investment_grade": roi.get("investment_grade"),
            }

        return analytics_data

    def _assess_data_availability(self, analytics_context: Dict[str, Any]) -> str:
        """Assess the richness of available analytics data."""
        data_points = 0

        if analytics_context.get("investment_analysis", {}).get("financial_metrics"):
            data_points += 3
        if analytics_context.get("market_analysis", {}).get("trend_direction"):
            data_points += 2
        if analytics_context.get("roi_analysis", {}).get("annual_roi"):
            data_points += 2

        if data_points >= 5:
            return "rich"
        elif data_points >= 2:
            return "mixed"
        else:
            return "limited"

    def _build_investor_context_prompt(
        self, query: str, analytics_context: Dict[str, Any], conversation_history: List, data_availability: str
    ) -> str:
        """Build context-aware prompt for investor agent."""

        # Start with enhanced base prompt
        base_prompt = get_context_enhanced_prompt(self.get_role_specific_prompt(), analytics_context)

        # Add adaptive analysis instructions
        adaptive_instructions = get_adaptive_analysis_prompt(data_availability)

        # Build context sections
        context_sections = []

        if analytics_context.get("investment_analysis"):
            investment = analytics_context["investment_analysis"]
            context_sections.append(
                f"""
**📊 INVESTMENT ANALYSIS AVAILABLE:**
- Analysis Type: {investment.get('type', 'N/A')}
- Properties Analyzed: {investment.get('properties_analyzed', 0)}
- Financial Metrics: {investment.get('financial_metrics', {})}
- Risk Assessment: {investment.get('risk_factors', {})}
- Analysis Date: {investment.get('analysis_timestamp', 'N/A')}
            """
            )

        if analytics_context.get("market_analysis"):
            market = analytics_context["market_analysis"]
            context_sections.append(
                f"""
**📈 MARKET ANALYSIS AVAILABLE:**
- Location: {market.get('location', 'N/A')}
- Market Trend: {market.get('trend_direction', 'N/A')}
- Volatility Score: {market.get('volatility_score', 'N/A')}
- 12-Month Forecast: {market.get('forecast', 'N/A')}
- Market Health Score: {market.get('market_health_score', 'N/A')}
            """
            )

        if analytics_context.get("roi_analysis"):
            roi = analytics_context["roi_analysis"]
            context_sections.append(
                f"""
**💰 ROI ANALYSIS AVAILABLE:**
- Annual ROI: {roi.get('annual_roi', 'N/A')}
- Monthly Cash Flow: {roi.get('monthly_cash_flow', 'N/A')}
- Confidence Score: {roi.get('confidence_score', 'N/A')}
- Investment Grade: {roi.get('investment_grade', 'N/A')}
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

**📋 AVAILABLE ANALYTICS CONTEXT:**
{chr(10).join(context_sections)
            if context_sections else "No stored analytics available - provide general guidance based on market principles."}

{conversation_context}

**🎯 RESPONSE INSTRUCTIONS:**
1. **Reference Specific Data**: Use exact numbers and metrics from the available analytics
2. **Provide Investment Recommendations**: Based on the analysis context provided
3. **Address Query Directly**: Ensure your response specifically answers the user's question
4. **Use Professional Investment Language**: Maintain expert-level investment terminology
5. **Include Action Items**: Provide specific, actionable next steps for the investor

**📊 RESPONSE FORMAT:**
🎯 **Investment Snapshot** (key metrics and immediate takeaway)
📊 **Market Analysis** (current conditions, trends, forecasts)
💰 **Financial Projections** (ROI, cash flow, appreciation estimates)
⚠️ **Risk Assessment** (potential challenges and mitigation strategies)
🎯 **Investment Strategy** (recommended approach and timing)
📋 **Next Steps** (specific actionable items)

Generate a comprehensive investment analysis response that leverages all available context.
        """

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of tools available to the investor agent."""
        return [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
        ]
