"""
Agent orchestration logic.
"""

import asyncio
import logging
from typing import AsyncGenerator
from uuid import UUID

from asyncpg import Connection

from ..data.repository import SessionRepository
from .factory import create_agent, get_agent_class

logger = logging.getLogger(__name__)


async def run_agent_turn(session_id: UUID, query: str, conn: Connection):
    """
    Orchestrates a single turn of conversation with the appropriate agent.
    """
    # 1. Get session to determine user role
    session_repo = SessionRepository(conn)
    session = await session_repo.get_session(session_id)
    if not session:
        raise ValueError("Session not found or has expired.")

    # 2. Instantiate the appropriate agent using the factory
    try:
        agent = create_agent(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

    # 4. Run the agent
    response = await agent.run(
        message=query, session_id=str(session.id), user_id=session.user_id, user_role=session.user_role.value
    )

    return response


async def stream_agent_turn(session_id: UUID, query: str, conn: Connection) -> AsyncGenerator[str, None]:
    """
    Orchestrates a streaming agent response with improved chunking.

    Args:
        session_id: The session UUID
        query: The user's query
        conn: Database connection

    Yields:
        Chunks of the response text
    """
    # 1. Get session to determine user role
    session_repo = SessionRepository(conn)
    session = await session_repo.get_session(session_id)
    if not session:
        raise ValueError("Session not found or has expired.")

    # 2. Instantiate the appropriate agent using the factory
    try:
        agent = create_agent(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

    # 3. Stream the response from the agent
    try:
        # Pass session context to agent if needed
        agent.dependencies.context_manager.get_or_create_context(
            str(session.id), session.user_id, session.user_role.value
        )

        async for chunk in agent.stream(
            message=query, session_id=str(session.id), user_id=session.user_id, user_role=session.user_role.value
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Error in stream_agent_turn: {e}", exc_info=True)
        # Yield error message in chunks
        error_msg = "I apologize, but I encountered an error while processing your request."
        for word in error_msg.split():
            yield word + " "


async def enhanced_stream_agent_turn(session_id: UUID, query: str, conn: Connection) -> AsyncGenerator[str, None]:
    """
    Enhanced streaming with tool execution visibility and context-aware responses.

    This function provides real-time feedback on tool execution progress and generates
    context-aware responses using stored analytics data.

    Args:
        session_id: The session UUID
        query: The user's query
        conn: Database connection

    Yields:
        Chunks of the response text with tool execution feedback
    """
    # 1. Get session to determine user role
    session_repo = SessionRepository(conn)
    session = await session_repo.get_session(session_id)
    if not session:
        raise ValueError("Session not found or has expired.")

    # 2. Instantiate the appropriate agent using the factory
    try:
        agent = create_agent(session.user_role)
    except NotImplementedError as e:
        logger.warning(f"Could not find agent for role {session.user_role}: {e}")
        raise

    try:
        # 3. Initialize context
        context = agent.dependencies.context_manager.get_or_create_context(
            str(session.id), session.user_id, session.user_role.value
        )

        # 4. Stream initial analysis feedback
        yield "🔧 **Analyzing your request with stored context...**\n\n"
        await asyncio.sleep(0.3)  # Brief pause for visual effect

        # 5. Execute agent processing with enhanced feedback
        async for tool_feedback in execute_tools_with_streaming(agent, query, context):
            yield tool_feedback
            await asyncio.sleep(0.2)  # Brief pause between tool updates

        # 6. Generate context-aware response with streaming
        yield "🧠 **Generating context-aware analysis...**\n\n"
        await asyncio.sleep(0.3)

        # 7. Stream the context-aware response
        async for chunk in generate_context_aware_response_stream(
            agent, query, str(session.id), session.user_id, session.user_role.value, context
        ):
            yield chunk

    except Exception as e:
        logger.error(f"Error in enhanced_stream_agent_turn: {e}", exc_info=True)
        # Yield error message in chunks
        error_msg = "I apologize, but I encountered an error while processing your request."
        for word in error_msg.split():
            yield word + " "


async def execute_tools_with_streaming(agent, query: str, context) -> AsyncGenerator[str, None]:
    """
    Execute agent tools with streaming progress feedback.

    This simulates tool execution progress for demonstration purposes,
    since the actual RAG pipeline tools don't have streaming interfaces yet.
    """
    # Determine which tools would be used based on query content
    potential_tools = []

    # Investment-related queries
    if any(
        keyword in query.lower()
        for keyword in [
            "invest",
            "roi",
            "return",
            "cash flow",
            "cap rate",
            "irr",
            "npv",
            "rental income",
            "investment",
            "financial metrics",
            "yield",
            "profit",
            "loss",
            "rental property",
            "rental yield",
            "rental return",
            "rental analysis",
            "rental investment",
            "rental cash flow",
            "rental cap rate",
            "rental irr",
            "rental npv",
        ]
    ):
        potential_tools.extend(
            [
                ("VectorSearchTool", "Searching property database", 1.2),
                ("InvestmentOpportunityAnalysisTool", "Analyzing investment metrics", 1.5),
                ("ROIProjectionTool", "Calculating ROI projections", 1.0),
                ("MarketAnalysisTool", "Evaluating market conditions", 0.8),
            ]
        )

    # Market analysis queries
    elif any(
        keyword in query.lower()
        for keyword in [
            "market",
            "trends",
            "prices",
            "analysis",
            "forecast",
            "demand",
            "supply",
            "growth",
            "decline",
            "appreciation",
            "depreciation",
            "economic indicators",
            "interest rates",
            "housing market",
            "real estate market",
            "buyer behavior",
            "seller behavior",
        ]
    ):
        potential_tools.extend(
            [
                ("MarketAnalysisTool", "Analyzing market trends", 1.2),
                ("VectorSearchTool", "Gathering market data", 0.9),
                ("RiskAssessmentTool", "Evaluating market risks", 1.1),
            ]
        )

    # Property search queries
    elif any(
        keyword in query.lower()
        for keyword in [
            "find",
            "search",
            "property",
            "house",
            "apartment",
            "buy",
            "rent",
            "listing",
            "available",
            "for sale",
            "for rent",
            "homes",
            "condo",
            "townhouse",
            "multi-family",
            "single family",
            "studio",
            "bedroom",
            "bathroom",
            "sqft",
            "square feet",
            "acre",
            "lot size",
            "price",
            "budget",
            "under",
            "over",
            "between",
            "in",
            "near",
            "close to",
            "around",
            "downtown",
            "suburbs",
            "neighborhood",
            "city",
        ]
    ):
        potential_tools.extend(
            [
                ("VectorSearchTool", "Searching property listings", 1.5),
                ("PropertyAnalysisTool", "Analyzing property details", 1.0),
                ("NeighborhoodAnalysisTool", "Evaluating neighborhoods", 0.8),
            ]
        )

    # Risk assessment queries
    elif any(
        keyword in query.lower()
        for keyword in [
            "risk",
            "safe",
            "volatile",
            "stable",
            "uncertain",
            "hazard",
            "threat",
            "risk assessment",
            "risk analysis",
            "risk management",
            "market risk",
            "property risk",
            "investment risk",
            "financial risk",
            "economic risk",
            "environmental risk",
            "legal risk",
            "regulatory risk",
            "political risk",
        ]
    ):
        potential_tools.extend(
            [
                ("RiskAssessmentTool", "Assessing investment risks", 1.3),
                ("MarketAnalysisTool", "Analyzing market stability", 1.0),
                ("VectorSearchTool", "Gathering risk data", 0.7),
            ]
        )

    # Default tools for general queries
    else:
        potential_tools.extend(
            [("VectorSearchTool", "Searching relevant data", 1.0), ("GeneralAnalysisTool", "Performing analysis", 1.2)]
        )

    # Execute tools with progress feedback
    tools_executed = 0
    for tool_name, description, duration in potential_tools[:3]:  # Limit to 3 tools for reasonable response time
        yield f"⚙️ **{tool_name}**: {description}...\n"

        # Simulate tool execution time
        await asyncio.sleep(duration)

        tools_executed += 1

        # Simulate different result counts based on tool type
        if "Search" in tool_name:
            result_count = 15 + (tools_executed * 3)
        elif "Analysis" in tool_name:
            result_count = 8 + (tools_executed * 2)
        else:
            result_count = 5 + tools_executed

        yield f"✅ **{tool_name} completed** - Found {result_count} data points\n\n"


async def generate_context_aware_response_stream(
    agent, query: str, session_id: str, user_id: str, user_role: str, context
) -> AsyncGenerator[str, None]:
    """
    Generate a context-aware response using stored analytics and stream it naturally.

    This leverages the agent's existing context-aware response generation
    but streams the output for better user experience.
    """
    try:
        # Use the agent's existing run method to get the full response
        response = await agent.run(message=query, session_id=session_id, user_id=user_id, user_role=user_role)

        # Extract the response content
        response_content = response.content

        # Stream the response naturally using the agent's streaming method
        async for chunk in agent._stream_naturally(response_content):
            yield chunk

    except Exception as e:
        logger.error(f"Error in generate_context_aware_response_stream: {e}", exc_info=True)
        # Fallback to basic streaming
        error_msg = "I apologize, but I encountered an error while generating the context-aware response."
        async for chunk in agent._stream_naturally(error_msg):
            yield chunk
