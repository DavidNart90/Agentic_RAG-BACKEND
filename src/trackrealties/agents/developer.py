"""
Developer-specific agent for the TrackRealties AI Platform.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from .base import AgentDependencies, BaseAgent, BaseTool
from .prompts import (BASE_SYSTEM_CONTEXT, DEVELOPER_SYSTEM_PROMPT,
                      get_adaptive_analysis_prompt,
                      get_context_enhanced_prompt)
from .tools import (ConstructionCostEstimationTool, FeasibilityAnalysisTool,
                    MarketAnalysisTool, SiteAnalysisTool, ZoningAnalysisTool)


class DeveloperAgent(BaseAgent):
    """Agent specializing in real estate developer tasks with context-aware development analysis."""

    MODEL_PATH = "models/developer_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("developer") if role_models else None
        tools = [
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="developer_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{DEVELOPER_SYSTEM_PROMPT}"

    async def generate_context_aware_response(self, query: str, session_id: str, **kwargs) -> str:
        """
        Generate context-aware response for real estate developers using stored feasibility studies,
        market analysis, and development project data.
        """
        try:
            # Get stored context from session
            context = await self.get_session_context(session_id)

            # Extract developer-specific context
            developer_context = self._extract_developer_context(context)

            # Determine data availability for adaptive analysis
            data_availability = self._assess_data_availability(developer_context)

            # Build enhanced prompt with developer context
            enhanced_prompt = self._build_developer_context_prompt(
                query=query,
                developer_context=developer_context,
                conversation_history=context.messages[-5:] if context else [],
                data_availability=data_availability,
            )

            # Generate response using enhanced prompt
            response = await self.llm_client.generate_response(enhanced_prompt, **kwargs)

            return response

        except Exception:
            # Fallback to standard response generation
            return await super().generate_response(query, session_id, **kwargs)

    def _extract_developer_context(self, context) -> Dict[str, Any]:
        """Extract developer-specific data from session context."""
        if not context or not hasattr(context, "metadata"):
            return {}

        developer_data = {}
        metadata = context.metadata

        # Extract feasibility analysis
        if "feasibility_analysis" in metadata:
            feasibility = metadata["feasibility_analysis"]
            developer_data["feasibility_analysis"] = {
                "project_type": feasibility.get("project_type"),
                "site_location": feasibility.get("site_location"),
                "total_project_cost": feasibility.get("total_project_cost"),
                "projected_revenue": feasibility.get("projected_revenue"),
                "roi_projection": feasibility.get("roi_projection"),
                "feasibility_score": feasibility.get("feasibility_score"),
                "timeline_months": feasibility.get("timeline_months"),
            }

        # Extract zoning analysis
        if "zoning_analysis" in metadata:
            zoning = metadata["zoning_analysis"]
            developer_data["zoning_analysis"] = {
                "current_zoning": zoning.get("current_zoning"),
                "permitted_uses": zoning.get("permitted_uses"),
                "building_restrictions": zoning.get("building_restrictions"),
                "variance_required": zoning.get("variance_required"),
                "approval_probability": zoning.get("approval_probability"),
            }

        # Extract site analysis
        if "site_analysis" in metadata:
            site = metadata["site_analysis"]
            developer_data["site_analysis"] = {
                "site_size": site.get("site_size"),
                "accessibility": site.get("accessibility"),
                "utility_access": site.get("utility_access"),
                "environmental_factors": site.get("environmental_factors"),
                "development_constraints": site.get("development_constraints"),
            }

        # Extract market analysis for development
        if "market_analysis" in metadata:
            market = metadata["market_analysis"]
            developer_data["market_analysis"] = {
                "location": market.get("location"),
                "demand_analysis": market.get("demand_analysis"),
                "absorption_rate": market.get("absorption_rate"),
                "competitive_landscape": market.get("competitive_landscape"),
                "market_timing": market.get("market_timing"),
            }

        # Extract construction cost estimates
        if "construction_costs" in metadata:
            costs = metadata["construction_costs"]
            developer_data["construction_costs"] = {
                "cost_per_sqft": costs.get("cost_per_sqft"),
                "total_construction_cost": costs.get("total_construction_cost"),
                "cost_breakdown": costs.get("cost_breakdown"),
                "contingency_percentage": costs.get("contingency_percentage"),
            }

        return developer_data

    def _assess_data_availability(self, developer_context: Dict[str, Any]) -> str:
        """Assess the richness of available developer context data."""
        data_points = 0

        if developer_context.get("feasibility_analysis", {}).get("feasibility_score"):
            data_points += 3
        if developer_context.get("zoning_analysis", {}).get("current_zoning"):
            data_points += 2
        if developer_context.get("site_analysis", {}).get("site_size"):
            data_points += 2
        if developer_context.get("market_analysis", {}).get("demand_analysis"):
            data_points += 2
        if developer_context.get("construction_costs", {}).get("cost_per_sqft"):
            data_points += 1

        if data_points >= 6:
            return "rich"
        elif data_points >= 3:
            return "mixed"
        else:
            return "limited"

    def _build_developer_context_prompt(
        self, query: str, developer_context: Dict[str, Any], conversation_history: List, data_availability: str
    ) -> str:
        """Build context-aware prompt for developer agent."""

        # Start with enhanced base prompt
        base_prompt = get_context_enhanced_prompt(self.get_role_specific_prompt(), developer_context)

        # Add adaptive analysis instructions
        adaptive_instructions = get_adaptive_analysis_prompt(data_availability)

        # Build context sections
        context_sections = []

        if developer_context.get("feasibility_analysis"):
            feasibility = developer_context["feasibility_analysis"]
            context_sections.append(
                f"""
**🏗️ FEASIBILITY ANALYSIS AVAILABLE:**
- Project Type: {feasibility.get('project_type', 'N/A')}
- Site Location: {feasibility.get('site_location', 'N/A')}
- Total Project Cost: {feasibility.get('total_project_cost', 'N/A')}
- Projected Revenue: {feasibility.get('projected_revenue', 'N/A')}
- ROI Projection: {feasibility.get('roi_projection', 'N/A')}
- Feasibility Score: {feasibility.get('feasibility_score', 'N/A')}
- Timeline: {feasibility.get('timeline_months', 'N/A')} months
            """
            )

        if developer_context.get("zoning_analysis"):
            zoning = developer_context["zoning_analysis"]
            context_sections.append(
                f"""
**📋 ZONING ANALYSIS AVAILABLE:**
- Current Zoning: {zoning.get('current_zoning', 'N/A')}
- Permitted Uses: {zoning.get('permitted_uses', 'N/A')}
- Building Restrictions: {zoning.get('building_restrictions', 'N/A')}
- Variance Required: {zoning.get('variance_required', 'N/A')}
- Approval Probability: {zoning.get('approval_probability', 'N/A')}
            """
            )

        if developer_context.get("site_analysis"):
            site = developer_context["site_analysis"]
            context_sections.append(
                f"""
**📍 SITE ANALYSIS AVAILABLE:**
- Site Size: {site.get('site_size', 'N/A')}
- Accessibility: {site.get('accessibility', 'N/A')}
- Utility Access: {site.get('utility_access', 'N/A')}
- Environmental Factors: {site.get('environmental_factors', 'N/A')}
- Development Constraints: {site.get('development_constraints', 'N/A')}
            """
            )

        if developer_context.get("market_analysis"):
            market = developer_context["market_analysis"]
            context_sections.append(
                f"""
**📈 MARKET ANALYSIS AVAILABLE:**
- Target Location: {market.get('location', 'N/A')}
- Demand Analysis: {market.get('demand_analysis', 'N/A')}
- Absorption Rate: {market.get('absorption_rate', 'N/A')}
- Competitive Landscape: {market.get('competitive_landscape', 'N/A')}
- Market Timing: {market.get('market_timing', 'N/A')}
            """
            )

        if developer_context.get("construction_costs"):
            costs = developer_context["construction_costs"]
            context_sections.append(
                f"""
**💰 CONSTRUCTION COST ANALYSIS AVAILABLE:**
- Cost per Sq Ft: {costs.get('cost_per_sqft', 'N/A')}
- Total Construction Cost: {costs.get('total_construction_cost', 'N/A')}
- Cost Breakdown: {costs.get('cost_breakdown', 'N/A')}
- Contingency: {costs.get('contingency_percentage', 'N/A')}%
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

**📋 AVAILABLE DEVELOPMENT CONTEXT:**
{chr(10).join(context_sections) if context_sections else "No stored development analysis available - provide general development guidance based on market principles."}

{conversation_context}

**🎯 RESPONSE INSTRUCTIONS:**
1. **Reference Specific Analysis**: Use exact data from feasibility studies, zoning analysis, and site evaluations
2. **Provide Development Recommendations**: Based on the comprehensive analysis context provided
3. **Address Regulatory Considerations**: Include permitting, zoning, and compliance guidance
4. **Focus on Financial Viability**: Emphasize ROI, costs, timeline, and market factors
5. **Include Implementation Steps**: Provide specific, actionable development phases and milestones

**📊 RESPONSE FORMAT:**
🏗️ **Development Overview** (project summary and potential)
📍 **Site Analysis** (zoning, access, utilities, constraints)
📈 **Market Opportunity** (demand analysis, competition, timing)
💹 **Financial Feasibility** (costs, revenues, returns, timeline)
📋 **Regulatory Path** (permitting, approvals, compliance requirements)
🛣️ **Implementation Roadmap** (phases, milestones, key decisions)

Generate a comprehensive development analysis response that leverages all available context and analysis data.
        """

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of tools available to the developer agent."""
        return [
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
