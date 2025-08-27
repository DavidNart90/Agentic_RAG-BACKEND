# TrackRealties AI Agent Framework

The TrackRealties AI Agent Framework provides a robust, type-safe foundation for building intelligent real estate agents using Pydantic AI. This framework enables the creation of role-specific agents that can understand user intent, execute tools, maintain conversation context, and provide validated responses.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TrackRealties AI Platform                │
├─────────────────────────────────────────────────────────────┤
│  Role-Specific Agents                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │   Investor  │ │  Developer  │ │    Buyer    │ │ Agent  │ │
│  │    Agent    │ │    Agent    │ │    Agent    │ │ Agent  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Base Agent Framework                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ BaseAgent: Core functionality, tool management,         │ │
│  │           context handling, response validation         │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Supporting Systems                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │    Tool     │ │   Context   │ │    Role     │ │Validation│ │
│  │   Registry  │ │  Manager    │ │  Detection  │ │ System │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Pydantic AI Core                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ LLM Integration, Streaming, Type Safety, Validation     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. BaseAgent Class

The `BaseAgent` class provides the foundational functionality that all role-specific agents inherit:

- **Tool Registration**: Automatic registration and management of agent tools
- **Context Management**: Conversation state and user preference tracking
- **Response Generation**: Structured response creation with validation
- **Streaming Support**: Real-time response streaming capabilities
- **Error Handling**: Comprehensive error handling and recovery

```python
from pydantic_ai.models import Model
from trackrealties.agents import BaseAgent, AgentContext, AgentDependencies

class MyAgent(BaseAgent):
    def __init__(
        self,
        model: Union[str, Model] = None,
        deps: Optional[AgentDependencies] = None,
        **kwargs
    ):
        super().__init__(
            agent_name="my_agent",
            model=model or "openai:gpt-4o",
            system_prompt="Base system prompt",
            tools=[search_tool, analysis_tool],
            deps=deps
        )
        
    def get_role_specific_prompt(self) -> str:
        return "You are a specialized real estate agent..."
```

### 2. Tool System

The tool system enables agents to interact with external systems and data sources:

```python
from trackrealties.agents import BaseTool, AgentDependencies
from typing import Dict, Any

class PropertySearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="property_search",
            description="Search for properties based on criteria"
        )
    
    async def execute(self, deps: AgentDependencies, **kwargs) -> Dict[str, Any]:
        # Tool implementation
        return {"success": True, "data": search_results}
```

### 3. Context Management

The context management system maintains conversation state across interactions:

```python
from trackrealties.agents import ContextManager, ConversationContext

context_manager = ContextManager()
context = context_manager.get_or_create_context(
    session_id="user_session",
    user_id="user_123",
    user_role="investor"
)
```

### 4. Role Detection

Automatic detection of user roles based on query patterns and conversation history:

```python
from trackrealties.agents import detect_user_role, UserRole

role = detect_user_role("I want to invest in rental properties")
# Returns: UserRole.INVESTOR
```

## User Roles

The framework supports four primary user roles, each with specific characteristics:

### Investor
- **Focus**: ROI, cash flow, portfolio growth
- **Tools**: Market analysis, property recommendations, financial calculations
- **Response Style**: Analytical, detailed financial metrics

### Developer
- **Focus**: Zoning, development potential, feasibility studies
- **Tools**: Regulatory analysis, market demand forecasting, site evaluation
- **Response Style**: Technical, regulatory-focused

### Buyer
- **Focus**: Affordability, neighborhoods, lifestyle factors
- **Tools**: Property search, affordability calculators, area analysis
- **Response Style**: Friendly, lifestyle-oriented

### Agent
- **Focus**: Market intelligence, client service, competitive analysis
- **Tools**: All tools, CMA, lead scoring, marketing strategies
- **Response Style**: Professional, comprehensive

## Available Tools

### Core Tools

1. **VectorSearchTool**: Semantic search across property and market data
2. **GraphSearchTool**: Relationship discovery in the knowledge graph
3. **MarketAnalysisTool**: Market trend analysis and insights
4. **PropertyRecommendationTool**: AI-powered property recommendations

### Tool Registration

Tools are automatically registered with agents and can be invoked by the LLM:

```python
# Tools are registered during agent initialization
agent = MyAgent(
    agent_name="test_agent",
    model="openai:gpt-4o",
    system_prompt="System prompt",
    tools=[VectorSearchTool(), MarketAnalysisTool()]
)

# Or added dynamically
agent.add_tool(PropertyRecommendationTool())
```

## Usage Examples

### Basic Agent Interaction

```python
import asyncio
from trackrealties.agents import BaseAgent

class RealEstateAgent(BaseAgent):
    def get_role_specific_prompt(self) -> str:
        return "You are a helpful real estate AI assistant."

async def main():
    agent = RealEstateAgent(
        agent_name="real_estate_agent",
        model="openai:gpt-4o",
        system_prompt="You help users with real estate questions.",
        tools=[]  # Tools would be imported and added here
    )
    
    response = await agent.run(
        message="I'm looking for investment properties in Austin",
        session_id="user_session_123",
        user_id="user_456",
        user_role="investor"
    )
    
    print(f"Response: {response.content}")
    print(f"Confidence: {response.confidence_score}")
    print(f"Tools used: {response.tools_used}")

asyncio.run(main())
```

### Streaming Responses

```python
async def streaming_example():
    agent = RealEstateAgent(...)
    
    async for chunk in agent.stream(
        message="Analyze the Seattle real estate market",
        session_id="session_123"
    ):
        print(chunk, end="", flush=True)
```

### Context Management

```python
# Get conversation context
context = agent.get_context("session_123")
if context:
    print(f"Messages: {len(context.messages)}")
    print(f"User preferences: {context.user_preferences}")

# Clear context when done
agent.clear_context("session_123")
```

## Validation System

The framework includes a comprehensive validation system to ensure response quality:

```python
from trackrealties.validation.base import ResponseValidator, ValidationResult

class CustomValidator(ResponseValidator):
    async def validate_response(
        self,
        response: str,
        context: Any,
        user_query: str
    ) -> ValidationResult:
        # Custom validation logic
        return ValidationResult(
            is_valid=True,
            confidence_score=0.95,
            issues=[],
            validation_metadata={"validator": "custom"}
        )

# Use with agent
agent = RealEstateAgent(
    ...,
    validator=CustomValidator()
)
```

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/trackrealties
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Agent Configuration
DEFAULT_MODEL=openai:gpt-4o
MAX_CONTEXT_LENGTH=4000
VALIDATION_ENABLED=true
```

### Model Selection

The framework supports multiple LLM providers:

```python
# OpenAI
agent = MyAgent(model="openai:gpt-4o")

# Anthropic
agent = MyAgent(model="anthropic:claude-3-sonnet")

# Local models via Ollama
agent = MyAgent(model="ollama:llama2")
```

## Testing

The framework includes comprehensive test coverage:

```bash
# Run all agent tests
pytest tests/test_base_agent.py -v

# Run specific test categories
pytest tests/test_base_agent.py::TestBaseAgent::test_agent_initialization -v
```

### Test Structure

```python
# Example test
@pytest.mark.asyncio
async def test_agent_execution():
    agent = TestAgent(...)
    response = await agent.run("Test message")
    
    assert isinstance(response, AgentResponse)
    assert response.content is not None
    assert response.confidence_score >= 0.0
```

## Performance Considerations

### Memory Management

- Contexts automatically expire after 24 hours (configurable)
- Periodic cleanup of expired contexts
- Message history truncation for long conversations

### Caching

- Tool results can be cached to improve performance
- Context data is kept in memory for fast access
- Database connections are pooled and reused

### Monitoring

Integration with Pydantic Logfire for comprehensive monitoring:

```python
import logfire

logfire.configure()
agent = MyAgent(..., instrument=True)
```

## Extension Points

### Custom Tools

Create domain-specific tools by extending `BaseTool`:

```python
class ZillowAPITool(BaseTool):
    def __init__(self):
        super().__init__(
            name="zillow_search",
            description="Search Zillow for property listings"
        )
    
    async def execute(self, deps: AgentDependencies, **kwargs):
        # Implement Zillow API integration
        pass
```

### Custom Validators

Implement domain-specific validation logic:

```python
class RealEstateValidator(ResponseValidator):
    async def validate_response(self, response, context, user_query):
        # Check for real estate accuracy
        # Validate price ranges
        # Ensure location consistency
        pass
```

### Role-Specific Agents

Create specialized agents for different user types:

#### InvestorAgent - Real Estate Investment Analysis

The `InvestorAgent` is a comprehensive investment analysis agent with specialized tools for real estate investors:

```python
from trackrealties.agents import InvestorAgent

# Initialize the investor agent
investor_agent = InvestorAgent(
    model="openai:gpt-4o",
    # Additional tools can be added here
)

# Analyze an investment opportunity
response = await investor_agent.run(
    message="Analyze this property: $350K purchase price, expecting $2,800/month rent in Austin, TX",
    user_role="investor",
    session_id="investor_session_123"
)
```

**Specialized Tools:**

1. **Investment Opportunity Analysis Tool**
   - Comprehensive cash flow analysis
   - Cash-on-cash return calculations
   - Cap rate analysis
   - Risk factor assessment
   - Investment recommendations with confidence scores

2. **ROI Projection Tool**
   - Multi-scenario analysis (base, optimistic, pessimistic)
   - Sensitivity analysis for key variables
   - Time horizon projections (1-30 years)
   - Risk-adjusted return calculations
   - Annualized return projections

3. **Risk Assessment Tool**
   - Market risk evaluation
   - Property-specific risk analysis
   - Financial risk assessment
   - Operational risk factors
   - Risk mitigation strategies

**Example Investment Analysis:**

```python
# The agent can perform sophisticated analysis like:
analysis_query = """
I'm considering a $400K duplex in Denver. Each unit rents for $1,400/month. 
I have $80K for down payment. What's the investment potential?
"""

response = await investor_agent.run(
    message=analysis_query,
    user_role="investor"
)

# Response includes:
# - Monthly cash flow projections
# - Cash-on-cash return calculations
# - Risk assessment with specific factors
# - Scenario analysis (best/worst case)
# - Investment recommendation with reasoning
```

**Key Features:**
- **Financial Modeling**: Detailed cash flow, ROI, and risk calculations
- **Scenario Analysis**: Multiple projection scenarios with sensitivity testing
- **Risk Assessment**: Comprehensive evaluation of investment risks
- **Market Integration**: Uses market data for location-specific analysis
- **Professional Insights**: Investment-grade analysis and recommendations

#### DeveloperAgent - Real Estate Development Analysis

The `DeveloperAgent` is a specialized development analysis agent with 4 core tools for real estate developers:

```python
from trackrealties.agents import DeveloperAgent

# Initialize the developer agent
developer_agent = DeveloperAgent(
    model="openai:gpt-4o",
    # Additional tools can be added here
)

# Analyze a development opportunity
response = await developer_agent.run(
    message="I want to develop a 50-unit apartment complex on a 2-acre lot in downtown Austin. What should I consider?",
    user_role="developer",
    session_id="developer_session_123"
)
```

**Specialized Tools:**

1. **Zoning Analysis Tool**
   - Regulatory compliance assessment
   - Permitted uses identification
   - Floor Area Ratio (FAR) calculations
   - Development timeline estimation
   - Constraint identification and mitigation

2. **Construction Cost Estimation Tool**
   - Detailed cost breakdowns per square foot
   - Location-based cost adjustments
   - Quality level considerations
   - Timeline projections
   - Cost factor analysis

3. **Feasibility Analysis Tool**
   - Financial viability assessment
   - ROI and profit margin calculations
   - Risk level evaluation
   - Development yield analysis
   - Strategic recommendations

4. **Site Analysis Tool**
   - Site characteristics evaluation
   - Accessibility scoring
   - Market factor analysis
   - Infrastructure assessment
   - Overall site rating and recommendations

**Example Development Analysis:**

```python
# The agent can perform comprehensive analysis like:
development_query = """
I'm considering developing a mixed-use project with retail on ground floor and 
apartments above on a 1.5-acre downtown lot. Construction budget is $8M. 
What's the feasibility?
"""

response = await developer_agent.run(
    message=development_query,
    user_role="developer"
)

# Response includes:
# - Zoning compliance and permitted uses
# - Detailed construction cost breakdown
# - Financial feasibility with ROI projections
# - Site suitability analysis with scoring
# - Risk assessment and mitigation strategies
```

**Key Features:**
- **Regulatory Expertise**: Comprehensive zoning and compliance analysis
- **Cost Modeling**: Detailed construction cost estimation with location factors
- **Financial Analysis**: Development feasibility with profit projections
- **Site Evaluation**: Multi-factor site assessment and scoring
- **Risk Management**: Development risk identification and mitigation strategies

## Best Practices

### 1. Tool Design

- Keep tools focused on single responsibilities
- Provide comprehensive parameter schemas
- Include proper error handling
- Return structured, validated results

### 2. Context Management

- Set appropriate expiration times for contexts
- Clean up expired contexts regularly
- Store only essential data in context
- Use user preferences to personalize responses

### 3. Validation

- Implement multiple validation layers
- Check for domain-specific accuracy
- Validate against known data sources
- Provide confidence scores

### 4. Error Handling

- Graceful degradation when tools fail
- Informative error messages for users
- Proper logging for debugging
- Fallback strategies for critical failures

## Troubleshooting

### Common Issues

1. **Tool Registration Failures**
   - Check tool parameter schemas
   - Verify tool inheritance from BaseTool
   - Ensure async/await usage

2. **Context Not Persisting**
   - Verify session ID consistency
   - Check context expiration settings
   - Ensure proper context updates

3. **Validation Errors**
   - Review validation logic
   - Check response format compatibility
   - Verify confidence score calculations

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Agent operations will now log detailed information
```

## Contributing

When extending the agent framework:

1. Follow the established patterns for tools and validators
2. Include comprehensive tests for new functionality
3. Update documentation for new features
4. Ensure type safety with proper annotations
5. Add examples demonstrating new capabilities

## Future Enhancements

Planned improvements to the framework:

- **Multi-modal Support**: Image and document analysis capabilities
- **Advanced Context**: Long-term memory and user modeling
- **Tool Composition**: Automatic tool chaining and workflows
- **Performance Optimization**: Caching and response optimization
- **Integration Expansion**: Additional external API integrations