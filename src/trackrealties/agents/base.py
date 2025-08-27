"""
Base classes for all agents in the TrackRealties AI Platform.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent as PydanticAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as OpenAI

from ..core.config import settings
from ..models.agent import ValidationResult
from ..models.db import ConversationMessage, MessageRole
from ..rag.optimized_pipeline import EnhancedRAGPipeline
from ..validation.base import ResponseValidator
from .context import ContextManager, ConversationContext
from .intent_classifier import TrackRealtiesIntentClassifier
from .prompts import GREETINGS_PROMPT

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Standard response format for all agents."""

    content: str
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    validation_result: dict[str, Any] | None = None
    confidence_score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str, deps: Optional["AgentDependencies"] = None):
        self.name = name
        self.description = description
        self.dependencies = deps or AgentDependencies()

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        pass

    def as_function(self):
        """Returns the execute method with a unique name for PydanticAI."""
        import inspect

        # Get the original signature and annotations
        original_sig = inspect.signature(self.execute)
        original_annotations = getattr(self.execute, "__annotations__", {})

        # Create a wrapper function that manually preserves everything
        async def wrapper(*args, **kwargs):
            return await self.execute(*args, **kwargs)

        # Manually copy important attributes
        wrapper.__name__ = self.name
        wrapper.__qualname__ = self.name
        wrapper.__signature__ = original_sig
        wrapper.__annotations__ = original_annotations
        wrapper.__doc__ = self.execute.__doc__
        wrapper.__module__ = self.execute.__module__

        return wrapper


class AgentDependencies(BaseModel):
    """Dependencies needed by agents and tools."""

    context_manager: ContextManager = Field(default_factory=ContextManager)
    rag_pipeline: Any = Field(default_factory=EnhancedRAGPipeline)  # Accept any RAG pipeline implementation
    intent_classifier: Optional["TrackRealtiesIntentClassifier"] = None

    class Config:
        arbitrary_types_allowed = True

    async def initialize(self):
        """Initialize all dependencies that require async setup."""
        try:
            # Initialize the RAG pipeline with its tools
            if hasattr(self.rag_pipeline, "initialize"):
                await self.rag_pipeline.initialize()
                logger.info("RAG pipeline initialized successfully")

            # Initialize intent classifier if needed
            if self.intent_classifier is None:
                from .intent_classifier import TrackRealtiesIntentClassifier

                self.intent_classifier = TrackRealtiesIntentClassifier()
                logger.info("Intent classifier initialized")

        except Exception as e:
            logger.error(f"Failed to initialize AgentDependencies: {e}")
            raise


class BaseAgent(ABC):
    """
    The BaseAgent class provides the foundational functionality that all
    role-specific agents inherit.
    """

    def __init__(
        self,
        agent_name: str,
        model: str | OpenAI = None,
        *,
        model_path: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        tools: list[type[BaseTool]] = None,
        deps: AgentDependencies | None = None,
        validator: ResponseValidator | None = None,
        **kwargs,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.dependencies = deps or AgentDependencies()
        self.validator = validator
        self.tools = {tool.name: tool for tool in (tools or [])}

        self.model_path = model_path

        _model = model or settings.DEFAULT_MODEL
        if isinstance(_model, str):
            # Assuming format "provider:model_name" e.g., "openai:gpt-4o"
            provider, model_name = _model.split(":")
            if provider == "openai":
                provider = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.llm = OpenAIModel(model_name, provider=provider)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        else:
            self.llm = _model

        self.agent = PydanticAI(
            model=self.llm,
            system_prompt=self.get_role_specific_prompt(),
            tools=[t.as_function() for t in self.tools.values()],  # Re-enable tools
        )

    @abstractmethod
    def get_role_specific_prompt(self) -> str:
        """Returns the role-specific part of the system prompt."""
        pass

    def build_context_aware_prompt(
        self, query: str, context: "ConversationContext", base_prompt: Optional[str] = None
    ) -> str:
        """
        Build a context-aware prompt that incorporates stored analytics data.

        Args:
            query: The user's current query
            context: Session context containing stored analytics and conversation history
            base_prompt: Optional override for the base prompt

        Returns:
            Enhanced prompt with context-aware information
        """
        # Use provided base prompt or get role-specific prompt
        prompt = base_prompt or self.get_role_specific_prompt()

        # Extract stored analytics from context metadata
        stored_analytics = self._extract_stored_analytics(context)

        # Build context sections
        context_sections = []

        # Add analytics context if available
        if stored_analytics:
            context_sections.append(self._build_analytics_context_section(stored_analytics))

        # Add conversation history context
        recent_messages = context.get_recent_messages(limit=5)
        if len(recent_messages) > 1:  # More than just the current message
            context_sections.append(self._build_conversation_context_section(recent_messages))

        # Add search results context if available
        search_context = context.metadata.get("search_results", [])
        if search_context:
            context_sections.append(self._build_search_context_section(search_context))

        # Combine prompt with context
        if context_sections:
            enhanced_prompt = f"""
{prompt}

**AVAILABLE CONTEXT FOR THIS RESPONSE:**
{chr(10).join(context_sections)}

**CURRENT USER QUERY:** {query}

**INSTRUCTIONS:** Use the available context to provide a comprehensive, data-driven response. Reference specific numbers, calculations, and insights from the stored analytics. 
Make your response highly relevant to the user's query by connecting it to the available context data.
"""
        else:
            enhanced_prompt = f"""
{prompt}

**CURRENT USER QUERY:** {query}

**INSTRUCTIONS:** Provide a comprehensive response based on your role-specific expertise. While no stored analytics context is available for this query, use your knowledge to provide valuable insights.
"""

        return enhanced_prompt

    def _extract_stored_analytics(self, context: "ConversationContext") -> Dict[str, Any]:
        """Extract all stored analytics data from session context."""
        analytics_data = {}

        # Extract various types of stored analytics
        metadata = context.metadata

        if "current_analysis" in metadata:
            analytics_data["investment_analysis"] = metadata["current_analysis"]

        if "market_analysis" in metadata:
            analytics_data["market_analysis"] = metadata["market_analysis"]

        if "roi_analysis" in metadata:
            analytics_data["roi_analysis"] = metadata["roi_analysis"]

        if "risk_analysis" in metadata:
            analytics_data["risk_analysis"] = metadata["risk_analysis"]

        return analytics_data

    def _validate_analytics_data(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic validation checks on analytics data and flag potential issues.

        Returns:
            Dictionary with validation results and flagged issues
        """
        validation_results = {"is_valid": True, "warnings": [], "errors": [], "recommendations": []}

        # Validate Investment Analysis
        if "investment_analysis" in analytics_data:
            investment = analytics_data["investment_analysis"]

            # Check basic financial data consistency
            purchase_price = investment.get("purchase_price", 0)
            monthly_rent = investment.get("monthly_rent", 0)
            annual_expenses = investment.get("annual_expenses", 0)

            if purchase_price > 0 and monthly_rent > 0:
                # Calculate basic cap rate for validation
                annual_rent = monthly_rent * 12
                if annual_expenses > 0:
                    net_operating_income = annual_rent - annual_expenses
                    calculated_cap_rate = (net_operating_income / purchase_price) * 100

                    # Get stored cap rate for comparison
                    stored_cap_rate = investment.get("financial_metrics", {}).get("cap_rate", 0)

                    # Flag if cap rates differ significantly (>1%)
                    if abs(calculated_cap_rate - stored_cap_rate) > 1.0:
                        validation_results["warnings"].append(
                            f"Cap rate discrepancy: Calculated {calculated_cap_rate:.2f}% vs Stored {stored_cap_rate:.2f}%"
                        )

                # Check for unrealistic values
                if monthly_rent > purchase_price * 0.05:  # Rent > 5% of purchase price monthly
                    validation_results["warnings"].append(
                        f"Unusually high rent-to-price ratio: ${monthly_rent:,}/month for ${purchase_price:,} property"
                    )

                if annual_expenses > annual_rent:
                    validation_results["errors"].append(
                        f"Annual expenses (${annual_expenses:,}) exceed annual rent (${annual_rent:,})"
                    )

        # Validate Market Analysis
        if "market_analysis" in analytics_data:
            market = analytics_data["market_analysis"]

            # Check for reasonable price change percentages
            price_change = market.get("market_trends", {}).get("price_change_percent", 0)
            if abs(price_change) > 50:  # More than 50% change is unusual
                validation_results["warnings"].append(
                    f"Extreme price change detected: {price_change:.1f}% - verify data accuracy"
                )

            # Check volatility score reasonableness
            volatility = market.get("market_metrics", {}).get("volatility_score", 0)
            if volatility > 1.0:  # Volatility should typically be < 1.0
                validation_results["warnings"].append(
                    f"High volatility score: {volatility:.3f} - consider increased risk"
                )

        # Validate ROI Analysis
        if "roi_analysis" in analytics_data:
            roi = analytics_data["roi_analysis"]

            # Check for unrealistic ROI values
            annual_roi = roi.get("annual_roi", 0)
            if annual_roi > 50:  # ROI > 50% is highly unusual
                validation_results["warnings"].append(
                    f"Unusually high ROI: {annual_roi:.1f}% - verify calculation methodology"
                )
            elif annual_roi < -20:  # ROI < -20% indicates major loss
                validation_results["errors"].append(f"Negative ROI indicates significant loss: {annual_roi:.1f}%")

        # Set overall validation status
        if validation_results["errors"]:
            validation_results["is_valid"] = False
            validation_results["recommendations"].append("Review and recalculate analytics due to identified errors")
        elif validation_results["warnings"]:
            validation_results["recommendations"].append(
                "Verify unusual values and consider additional market research"
            )

        return validation_results

    def _build_analytics_context_section(self, analytics_data: Dict[str, Any]) -> str:
        """Build the analytics context section for the prompt with validation details."""
        sections = []

        # Investment Analysis Context
        if "investment_analysis" in analytics_data:
            investment = analytics_data["investment_analysis"]
            financial_metrics = investment.get("financial_metrics", {})
            sections.append(
                f"""
**📊 INVESTMENT ANALYSIS AVAILABLE:**
- Analysis Type: {investment.get('type', 'Unknown')}
- Purchase Price: ${investment.get('purchase_price', 'N/A'):,} 
- Monthly Rent: ${investment.get('monthly_rent', 'N/A'):,}
- Annual Expenses: ${investment.get('annual_expenses', 'N/A'):,}
- Cap Rate: {financial_metrics.get('cap_rate', 'N/A')}%
- Cash-on-Cash Return: {financial_metrics.get('cash_on_cash_return', 'N/A')}%
- Monthly Cash Flow: ${financial_metrics.get('monthly_cash_flow', 'N/A'):,}
- Risk Score: {financial_metrics.get('risk_score', 'N/A')}/10
- Properties Analyzed: {investment.get('properties_analyzed', 0)}
- Analysis Date: {investment.get('analysis_timestamp', 'N/A')}
- Data Sources: {investment.get('comparable_properties', 0)} comparable properties

**VALIDATION NOTES:** Verify cap rate = (annual_rent - annual_expenses) / purchase_price. Check if cash flow calculation includes all expenses."""
            )

        # Market Analysis Context
        if "market_analysis" in analytics_data:
            market = analytics_data["market_analysis"]
            market_trends = market.get("market_trends", {})
            market_metrics = market.get("market_metrics", {})
            sections.append(
                f"""
**📈 MARKET ANALYSIS AVAILABLE:**
- Location: {market.get('location', 'N/A')}
- Market Trend Direction: {market_trends.get('trend_direction', 'N/A')}
- Price Change: {market_trends.get('price_change_percent', 'N/A')}%
- Volume Change: {market_trends.get('volume_change_percent', 'N/A')}%
- Trend Strength: {market_trends.get('trend_strength', 'N/A')}
- Volatility Score: {market_metrics.get('volatility_score', 'N/A')}
- Market Health Score: {market_metrics.get('market_health_score', 'N/A')}/100
- Current Median Price: ${market_metrics.get('current_median_price', 'N/A'):,}
- 12-Month Forecast: {market_metrics.get('12_month_forecast', 'N/A')}
- Data Points Analyzed: {market.get('data_sources', 'N/A')}
- Forecast Confidence: {market_trends.get('forecast_confidence', 'N/A')}

**VALIDATION NOTES:** Check if market trends align with property-specific data. Verify if volatility scores match price change patterns."""
            )

        # ROI Analysis Context
        if "roi_analysis" in analytics_data:
            roi = analytics_data["roi_analysis"]
            projections = roi.get("projections", {})
            sections.append(
                f"""
**💰 ROI ANALYSIS AVAILABLE:**
- Annual ROI: {roi.get('annual_roi', projections.get('irr', 'N/A'))}%
- 5-Year Total ROI: {projections.get('five_year_roi', 'N/A')}%
- Monthly Cash Flow: ${roi.get('monthly_cash_flow', projections.get('monthly_cash_flow', 'N/A')):,}
- Annual Cash Flow: ${projections.get('annual_cash_flow', 'N/A'):,}
- Investment Timeline: {roi.get('projection_years', 'N/A')} years
- Break-even Time: {projections.get('break_even_months', 'N/A')} months
- Total Cash Required: ${projections.get('total_cash_required', 'N/A'):,}
- Appreciation Rate Used: {roi.get('appreciation_rate', 'N/A')}%
- ROI Confidence: {roi.get('confidence_score', 'N/A')}

**VALIDATION NOTES:** Verify ROI calculations include appreciation + cash flow. Check if break-even timeline is realistic given cash flow."""
            )

        # Risk Analysis Context
        if "risk_analysis" in analytics_data:
            risk = analytics_data["risk_analysis"]
            sections.append(
                f"""
**⚠️ RISK ANALYSIS AVAILABLE:**
- Overall Risk Score: {risk.get('overall_risk_score', 'N/A')}/10
- Risk Level: {risk.get('risk_level', 'N/A')}
- Market Risk Factors: {risk.get('market_risk_factors', [])}
- Property Risk Factors: {risk.get('property_risk_factors', [])}
- Financial Risk Factors: {risk.get('financial_risk_factors', [])}
- Mitigation Strategies: {risk.get('mitigation_strategies', [])}
- Risk Confidence: {risk.get('confidence_score', 'N/A')}

**VALIDATION NOTES:** Ensure risk level aligns with market volatility and financial metrics. Check if risk factors are relevant to current market conditions."""
            )

        return "\n".join(sections)

    def _build_conversation_context_section(self, recent_messages: List["ConversationMessage"]) -> str:
        """Build conversation history context section."""
        history_items = []
        for msg in recent_messages[-3:]:  # Last 3 messages for context
            role_emoji = "👤" if msg.role == MessageRole.USER else "🤖"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            history_items.append(f"{role_emoji} {msg.role.value}: {content_preview}")

        return f"""
**💬 RECENT CONVERSATION CONTEXT:**
{chr(10).join(history_items)}"""

    def _build_search_context_section(self, search_results: List[Dict[str, Any]]) -> str:
        """Build search results context section."""
        if not search_results:
            return ""

        # Get the most recent search results
        recent_search = search_results[-1] if search_results else {}

        return f"""
**🔍 RECENT SEARCH RESULTS:**
- Search Type: {recent_search.get('search_type', 'N/A')}
- Query: {recent_search.get('query', 'N/A')}
- Results Found: {len(recent_search.get('search_results', []))}
- Confidence: {recent_search.get('rag_result', {}).get('confidence_score', 'N/A')}"""

    def add_tool(self, tool: type[BaseTool]):
        """Adds a tool to the agent."""
        self.tools[tool.name] = tool
        # Re-initialize PydanticAI with the new toolset
        self.agent = PydanticAI(
            model=self.llm,
            system_prompt=self.get_role_specific_prompt(),
            tools=[t.as_function() for t in self.tools.values()],
        )

    async def run(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Process a message with intelligent intent classification and context-aware responses."""
        try:
            # Get or create context
            context = self.dependencies.context_manager.get_or_create_context(
                session_id=session_id or "default",
                user_id=user_id,
                user_role=user_role,
            )
            user_message = ConversationMessage(
                role=MessageRole.USER,
                content=message,
                session_id=session_id or "default",
                created_at=datetime.now(timezone.utc),
            )
            context.add_message(user_message)

            # Initialize intent classifier if not exists
            if not hasattr(self.dependencies, "intent_classifier") or self.dependencies.intent_classifier is None:
                from .intent_classifier import TrackRealtiesIntentClassifier

                self.dependencies.intent_classifier = TrackRealtiesIntentClassifier()

            # Classify intent using LLM
            intent_result = await self.dependencies.intent_classifier.classify_intent(message, user_role)

            # Log the classification result
            logger.info(
                f"Intent Classification Result: {intent_result.intent} (confidence: {intent_result.confidence:.2f}, source: {intent_result.source})"
            )

            # Handle based on classified intent
            if intent_result.intent == "pure_greeting":
                logger.info(f"Processing pure greeting from {user_role or 'user'}")

                # Use existing greeting logic
                greeting_prompt = GREETINGS_PROMPT
                # Add role-specific context to the greeting prompt
                role_context = {
                    "investor": "\n\nThe user is an investor, so mention how you can help with ROI analysis, market trends, and investment opportunities.",
                    "developer": "\n\nThe user is a developer, so mention how you can assist with site analysis, feasibility studies, and development opportunities.",
                    "buyer": "\n\nThe user is a home buyer, so mention how you can help find the perfect property, analyze neighborhoods, and guide through the buying process.",
                    "agent": "\n\nThe user is a real estate agent, so mention how you can provide market intelligence, lead insights, and business growth strategies.",
                }

                if user_role in role_context:
                    greeting_prompt += role_context[user_role]

                # Generate greeting response
                result = await self.agent.run(greeting_prompt)

                # Add assistant message to context
                assistant_msg = ConversationMessage(
                    role=MessageRole.ASSISTANT,
                    content=str(result.output),
                    session_id=session_id or "default",
                    created_at=datetime.now(timezone.utc),
                )
                context.add_message(assistant_msg)

                # Update context
                self.dependencies.context_manager.update_context(session_id or "default", context)

                return AgentResponse(
                    content=str(result.output),
                    confidence_score=1.0,
                    metadata={
                        "intent": "pure_greeting",
                        "skip_search": True,
                        "intent_confidence": intent_result.confidence,
                        "entities": intent_result.entities,
                    },
                )

            elif intent_result.intent == "greeting_with_vague_help":
                logger.info(f"Processing greeting with vague help request from {user_role or 'user'}")

                # Create conversational response asking for more specific information
                role_context = f"as a {user_role}" if user_role else ""
                vague_help_prompt = f"""
                The user has greeted you and asked for general help with real estate {role_context}.
                Respond warmly and ask what specific aspect of real estate they need help with.
                Be conversational and provide examples of things you can help with based on their role.

                User message: {message}
                User role: {user_role or 'general user'}
                """

                # Generate conversational response
                result = await self.agent.run(vague_help_prompt)

                # Add assistant message to context
                assistant_msg = ConversationMessage(
                    role=MessageRole.ASSISTANT,
                    content=str(result.output),
                    session_id=session_id or "default",
                    created_at=datetime.now(timezone.utc),
                )
                context.add_message(assistant_msg)

                # Update context
                self.dependencies.context_manager.update_context(session_id or "default", context)

                return AgentResponse(
                    content=str(result.output),
                    confidence_score=1.0,
                    metadata={
                        "intent": "greeting_with_vague_help",
                        "skip_search": True,
                        "intent_confidence": intent_result.confidence,
                        "entities": intent_result.entities,
                        "response_type": "conversational_clarification",
                    },
                )

            elif intent_result.intent == "greeting_with_query":
                logger.info(
                    f"Processing greeting with business query from {user_role or 'user'}: {intent_result.intent}"
                )

                # Acknowledge greeting but process the business query using PydanticAI agent
                greeting_ack = "Hello! Let me help you with that right away."

                # Add greeting acknowledgment to context
                context.add_message(
                    ConversationMessage(
                        role=MessageRole.ASSISTANT,
                        content=greeting_ack,
                        session_id=session_id or "default",
                        created_at=datetime.now(timezone.utc),
                    )
                )

                # Continue to business processing using PydanticAI agent with tools
                logger.info("Using PydanticAI agent with tools for greeting_with_query business analysis")

                # Build context-aware prompt for the agent
                enhanced_prompt = self.build_context_aware_prompt(
                    query=message,
                    context=context,
                    base_prompt=f"The user has greeted you and asked: {
                        message}. Acknowledge the greeting briefly and then provide comprehensive business analysis using your available tools.",
                )

                # Let PydanticAI agent handle the query using its tools
                agent_result = await self.agent.run(enhanced_prompt)
                business_response = str(agent_result.output)

                # Combine greeting acknowledgment with business response
                combined_response = f"{greeting_ack}\n\n{business_response}"

                # Extract tool usage information
                tools_used = []
                if hasattr(agent_result, "tools_used") and agent_result.tools_used:
                    tools_used = [
                        {"tool_name": tool.__class__.__name__, "args": tool.args} for tool in agent_result.tools_used
                    ]

                # Create agent response
                agent_response = AgentResponse(
                    content=combined_response,
                    tools_used=tools_used,
                    confidence_score=0.9,
                    metadata={
                        "intent": intent_result.intent,
                        "requires_search": intent_result.requires_search,
                        "intent_confidence": intent_result.confidence,
                        "entities": intent_result.entities,
                        "keywords": intent_result.keywords,
                        "response_type": "greeting_with_business_pydantic",
                        "user_role": user_role,
                        "session_id": session_id or "default",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "agent_tools_used": True,
                    },
                )

                # Handle validation - create default since no validation from PydanticAI yet
                combined_result = ValidationResult(
                    is_valid=True,
                    confidence_score=0.9,
                    issues=[],
                    validation_type="pydantic_agent_response",
                    validator_version="1.0",
                    correction_needed=False,
                )

                # Add final response to context
                assistant_msg = ConversationMessage(
                    session_id=session_id or "default",
                    role=MessageRole.ASSISTANT,
                    content=agent_response.content,
                    created_at=datetime.now(timezone.utc),
                )
                context.add_message(assistant_msg)

                # Persist updated context
                self.dependencies.context_manager.update_context(session_id or "default", context)

                # Additional validation if validator is available
                if self.validator:
                    try:
                        extra = await self.validator.validate(agent_response.content, context.dict())
                        combined_result = ValidationResult(
                            is_valid=combined_result.is_valid and extra.is_valid,
                            confidence_score=min(combined_result.confidence_score, extra.confidence_score),
                            issues=combined_result.issues + extra.issues,
                            validation_type="combined",
                            validator_version="1.0",
                            correction_needed=combined_result.correction_needed or extra.correction_needed,
                        )
                    except Exception as e:
                        logger.warning(f"Validator failed: {e}")

                agent_response.validation_result = combined_result.dict()
                agent_response.confidence_score = combined_result.confidence_score

                return agent_response

            else:
                logger.info(f"Processing business query: {intent_result.intent} from {user_role or 'user'}")

                # Use PydanticAI agent with tools instead of bypassing to RAG pipeline
                logger.info("Using PydanticAI agent with specialized tools for business analysis")

                # Build context-aware prompt for the agent
                enhanced_prompt = self.build_context_aware_prompt(query=message, context=context)

                # Let PydanticAI agent handle the query using its tools
                agent_result = await self.agent.run(enhanced_prompt)
                response_content = str(agent_result.output)

                # Extract tool usage information
                tools_used = []
                if hasattr(agent_result, "tools_used") and agent_result.tools_used:
                    tools_used = [
                        {"tool_name": tool.__class__.__name__, "args": tool.args} for tool in agent_result.tools_used
                    ]

                # Create agent response
                agent_response = AgentResponse(
                    content=response_content,
                    tools_used=tools_used,
                    confidence_score=0.9,  # High confidence when using specialized tools
                    metadata={
                        "intent": intent_result.intent,
                        "requires_search": intent_result.requires_search,
                        "intent_confidence": intent_result.confidence,
                        "entities": intent_result.entities,
                        "keywords": intent_result.keywords,
                        "response_type": "pydantic_agent_with_tools",
                        "user_role": user_role,
                        "session_id": session_id or "default",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "agent_tools_used": True,
                    },
                )

                # Handle validation - create default since no validation from PydanticAI yet
                combined_result = ValidationResult(
                    is_valid=True,
                    confidence_score=0.9,
                    issues=[],
                    validation_type="pydantic_agent_response",
                    validator_version="1.0",
                    correction_needed=False,
                )

                # Append assistant message to history
                assistant_msg = ConversationMessage(
                    session_id=session_id or "default",
                    role=MessageRole.ASSISTANT,
                    content=agent_response.content,
                    created_at=datetime.now(timezone.utc),
                )
                context.add_message(assistant_msg)

                # Persist updated context
                self.dependencies.context_manager.update_context(session_id or "default", context)

                # Additional validation if validator is available (business query section)
                if self.validator:
                    try:
                        extra = await self.validator.validate(agent_response.content, context.dict())
                        combined_result = ValidationResult(
                            is_valid=combined_result.is_valid and extra.is_valid,
                            confidence_score=min(combined_result.confidence_score, extra.confidence_score),
                            issues=combined_result.issues + extra.issues,
                            validation_type="combined",
                            validator_version="1.0",
                            correction_needed=combined_result.correction_needed or extra.correction_needed,
                        )
                    except Exception as e:
                        logger.warning(f"Validator failed: {e}")
                        # Keep the original validation result if validator fails

                agent_response.validation_result = combined_result.dict()
                agent_response.confidence_score = combined_result.confidence_score

                return agent_response

        except Exception as e:
            logger.error(f"Error in agent run method: {e}", exc_info=True)
            return AgentResponse(
                content="I apologize, but I encountered an error while processing your request. Please try again.",
                tools_used=[],
                confidence_score=0.0,
                metadata={"error": str(e), "response_type": "error"},
            )

    async def stream(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
        user_role: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams the response from the agent with natural chunking.

        Args:
            message: The user's message
            session_id: The session ID
            user_id: Optional user ID
            user_role: Optional user role

        Yields:
            Chunks of the response text
        """
        try:
            # Get the full response using the existing run method
            response = await self.run(message=message, session_id=session_id, user_id=user_id, user_role=user_role)

            # Extract the content from the response
            response_content = response.content

            # Stream the response in natural chunks
            async for chunk in self._stream_naturally(response_content):
                yield chunk

        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            error_msg = "I apologize, but I encountered an error while processing your request."
            async for chunk in self._stream_naturally(error_msg):
                yield chunk

    async def _stream_naturally(self, text: str) -> AsyncGenerator[str, None]:
        """
        Stream text in natural chunks (words/phrases) for better user experience.

        Args:
            text: The text to stream

        Yields:
            Natural chunks of text
        """
        # Split into sentences for more natural streaming
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for i, sentence in enumerate(sentences):
            # Further split long sentences into phrases at natural break points
            if len(sentence) > 100:
                # Split on commas, semicolons, or other natural breaks
                phrases = re.split(r"(?<=[,;:])\s+", sentence)

                for j, phrase in enumerate(phrases):
                    # Stream phrase word by word
                    words = phrase.split()
                    for k, word in enumerate(words):
                        if k > 0:
                            yield " "  # Add space between words
                        yield word
                        # Small delay to simulate natural typing
                        await asyncio.sleep(0.02)  # 20ms between words

                    # Add punctuation/spacing after phrase if not the last one
                    if j < len(phrases) - 1:
                        yield " "
                        await asyncio.sleep(0.03)  # Slightly longer pause between phrases
            else:
                # Stream short sentences word by word
                words = sentence.split()
                for k, word in enumerate(words):
                    if k > 0:
                        yield " "
                    yield word
                    await asyncio.sleep(0.02)

            # Add spacing between sentences
            if i < len(sentences) - 1:
                yield " "
                await asyncio.sleep(0.05)  # Longer pause between sentences

    async def _generate_context_aware_response(
        self, query: str, rag_response: str, context: ConversationContext, session_id: str
    ) -> str:
        """
        Generate an enhanced response using stored analytics context.

        This method checks if there's stored analytics data in the session context
        and generates a more intelligent response that references specific calculations
        and analysis results.
        """
        try:
            # Extract stored analytics from context
            stored_analytics = self._extract_stored_analytics(context)

            # If no analytics context is available, return the original RAG response
            if not stored_analytics:
                logger.info(f"No stored analytics context found for session {session_id}")
                return rag_response

            # Validate analytics data for consistency and reasonableness
            validation_results = self._validate_analytics_data(stored_analytics)

            # Build context-aware prompt for LLM enhancement with validation instructions
            enhancement_prompt = self.build_context_aware_prompt(
                query=query,
                context=context,
                base_prompt=f"""
You are an expert real estate advisor with access to detailed analytics data. 
Your job is to enhance the response by incorporating analytics context while VALIDATING all calculations and data consistency.

**ORIGINAL RESPONSE:**
{rag_response}

**DATA VALIDATION STATUS:**
- Overall Data Validity: {"✅ VALID" if validation_results["is_valid"] else "❌ ISSUES DETECTED"}
- Warnings: {validation_results["warnings"] if validation_results["warnings"] else "None"}
- Errors: {validation_results["errors"] if validation_results["errors"] else "None"}
- Recommendations: {validation_results["recommendations"] if validation_results["recommendations"] else "Data appears consistent"}

**CRITICAL VALIDATION TASKS:**
1. **Data Verification**: Check if the analytics calculations make sense given the property data
2. **Calculation Accuracy**: Verify ROI percentages, cash flow, and market metrics are reasonable
3. **Data Consistency**: Ensure market trends align with property values and risk assessments
4. **Gap Analysis**: Identify missing data points needed for complete analysis
5. **Reasonableness Check**: Flag any unrealistic numbers (e.g., 500% ROI, negative property values)

**ENHANCEMENT INSTRUCTIONS:**
- IF analytics data is accurate and consistent: Reference specific numbers, calculations, and insights
- IF calculations seem incorrect: Explain the discrepancy and provide corrected analysis
- IF data is insufficient: Clearly state what additional data is needed for complete analysis
- IF market trends don't match property data: Highlight the inconsistency and provide explanation
- ALWAYS validate that ROI calculations align with property price, rent, and expenses

**RESPONSE STRUCTURE:**
1. **Data Validation Summary** (Are the calculations reasonable? Any inconsistencies?)
2. **Enhanced Analysis** (Incorporate verified analytics data with specific numbers)
3. **Calculation Verification** (Confirm or correct any financial projections)
4. **Data Gaps & Recommendations** (What additional analysis might be needed?)

Make your response authoritative by showing you've verified the analytics rather than blindly accepting them.
""",
            )

            # Generate enhanced response using the agent's LLM
            enhanced_result = await self.agent.run(enhancement_prompt)
            enhanced_response = str(enhanced_result.output)

            logger.info(
                f"Generated context-aware response for session {
                    session_id} using {len(stored_analytics)} analytics contexts"
            )

            return enhanced_response

        except Exception as e:
            logger.warning(f"Failed to generate context-aware response: {e}")
            # Fallback to original response if enhancement fails
            return rag_response
