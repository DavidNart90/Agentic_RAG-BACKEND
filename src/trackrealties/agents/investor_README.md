# InvestorAgent - Real Estate Investment Analysis

The InvestorAgent is a specialized AI agent designed specifically for real estate investors. It provides comprehensive investment analysis, financial modeling, risk assessment, and portfolio optimization capabilities to help investors make data-driven decisions.

## Overview

The InvestorAgent extends the base TrackRealties agent framework with investor-specific tools and expertise. It combines quantitative analysis with market intelligence to provide actionable investment insights.

### Key Capabilities

- **Investment Opportunity Analysis**: Comprehensive property evaluation with cash flow projections
- **ROI Projections**: Multi-scenario return analysis with sensitivity testing
- **Risk Assessment**: Detailed risk evaluation across multiple categories
- **Portfolio Optimization**: Diversification and performance analysis
- **Market Timing**: Cycle analysis and timing recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    InvestorAgent                            │
├─────────────────────────────────────────────────────────────┤
│  Specialized Tools                                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐  │
│  │   Investment    │ │  ROI Projection │ │     Risk     │  │
│  │   Opportunity   │ │      Tool       │ │  Assessment  │  │
│  │     Tool        │ │                 │ │     Tool     │  │
│  └─────────────────┘ └─────────────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Base Agent Framework                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Context Management, Tool Registry, Validation System    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Investment Analysis Tools

### 1. Investment Opportunity Tool

Analyzes individual properties for investment potential:

```python
from trackrealties.agents.investor import InvestorAgent

agent = InvestorAgent()

# The agent can analyze properties using natural language
response = await agent.run(
    "Analyze this $350K property in Austin that rents for $2800/month",
    user_role="investor"
)
```

**Analysis Includes:**
- Cash flow projections (monthly and annual)
- Key investment metrics (cash-on-cash return, cap rate)
- Expense breakdown (mortgage, taxes, insurance, maintenance)
- Risk assessment and scoring
- Investment recommendation with confidence level

**Sample Output:**
```
Financial Metrics:
- Monthly Cash Flow: $485.32
- Cash-on-Cash Return: 8.7%
- Cap Rate: 6.4%
- Annual Cash Flow: $5,823.84

Recommendation: Buy (75% confidence)
Reasoning: Good cash-on-cash return of 8.7%. Positive cash flow of $485/month. 
Decent cap rate of 6.4%.
```

### 2. ROI Projection Tool

Projects returns over multiple time horizons with scenario analysis:

```python
# Multi-scenario ROI projection
projection_params = {
    "initial_investment": 87500,
    "annual_cash_flow": 6000,
    "property_value": 350000,
    "appreciation_rate": 4.0,
    "time_horizon": 10
}
```

**Features:**
- **Base, Optimistic, and Pessimistic scenarios**
- **Sensitivity analysis** on key variables
- **Risk-adjusted return calculations**
- **Year-by-year breakdown**
- **Key insights and recommendations**

**Sample Scenarios:**
- Base Case: 11.2% annualized return
- Optimistic: 14.8% annualized return  
- Pessimistic: 7.6% annualized return

### 3. Risk Assessment Tool

Comprehensive risk evaluation across multiple categories:

```python
risk_data = {
    "property_data": {
        "age": 15,
        "condition": "good",
        "neighborhood_grade": "B",
        "type": "single_family"
    },
    "market_data": {
        "price_volatility": 0.12,
        "cycle_phase": "expansion",
        "economic_diversity_score": 8
    },
    "financial_data": {
        "loan_to_value": 75,
        "debt_service_coverage": 1.3,
        "loan_type": "fixed"
    }
}
```

**Risk Categories:**
1. **Market Risk**: Price volatility, economic cycles, population trends
2. **Property Risk**: Age, condition, location, tenant quality
3. **Financial Risk**: Leverage, cash flow coverage, interest rate exposure
4. **Operational Risk**: Management complexity, distance, regulations

**Risk Mitigation Strategies:**
- Specific recommendations for each risk category
- Priority levels and implementation timelines
- Ongoing monitoring recommendations

## Usage Examples

### Basic Investment Analysis

```python
import asyncio
from trackrealties.agents.investor import InvestorAgent

async def analyze_investment():
    agent = InvestorAgent(model="openai:gpt-4o")
    
    response = await agent.run(
        message="I'm looking at a $280K duplex in Dallas that generates $2400/month in rent. Should I invest?",
        session_id="investor_session_1",
        user_role="investor"
    )
    
    print(f"Analysis: {response.content}")
    print(f"Confidence: {response.confidence_score}")
    print(f"Tools used: {response.tools_used}")

asyncio.run(analyze_investment())
```

### Portfolio Analysis

```python
async def portfolio_review():
    agent = InvestorAgent()
    
    response = await agent.run(
        message="I own 3 properties in Austin worth $850K total. How can I optimize my portfolio?",
        session_id="portfolio_session",
        user_role="investor"
    )
    
    print(response.content)

asyncio.run(portfolio_review())
```

### Market Timing Analysis

```python
async def market_timing():
    agent = InvestorAgent()
    
    response = await agent.run(
        message="Is now a good time to invest in Phoenix real estate, or should I wait?",
        session_id="timing_session", 
        user_role="investor"
    )
    
    print(response.content)

asyncio.run(market_timing())
```

## Investment Metrics Calculated

### Cash Flow Analysis
- **Gross Rental Income**: Monthly and annual rental income
- **Operating Expenses**: Property taxes, insurance, maintenance, vacancy
- **Net Operating Income (NOI)**: Income after operating expenses
- **Cash Flow After Debt Service**: NOI minus mortgage payments

### Return Metrics
- **Cash-on-Cash Return**: Annual cash flow / initial cash investment
- **Cap Rate**: NOI / property value
- **Internal Rate of Return (IRR)**: Time-weighted return including appreciation
- **Total Return**: Cash flow + appreciation over holding period

### Risk Metrics
- **Debt Service Coverage Ratio**: NOI / debt service
- **Loan-to-Value Ratio**: Loan amount / property value
- **Risk Score**: Composite risk rating (1-10 scale)
- **Volatility Estimate**: Expected return variability

## Investment Strategies Supported

### Buy and Hold
- Long-term rental property analysis
- Cash flow optimization
- Appreciation projections
- Tax benefit analysis

### Fix and Flip
- Renovation cost estimation
- After-repair value (ARV) analysis
- Holding period optimization
- Market timing considerations

### Commercial Real Estate
- Commercial property analysis
- Tenant risk assessment
- Lease structure evaluation
- Market comparables

### Portfolio Building
- Diversification analysis
- Geographic distribution
- Property type allocation
- Risk-adjusted portfolio optimization

## Configuration

### Model Selection

```python
# Use different models for different analysis types
agent = InvestorAgent(
    model="openai:gpt-4o",  # For complex analysis
    # model="anthropic:claude-3-sonnet",  # Alternative
    # model="openai:gpt-3.5-turbo",  # For faster responses
)
```

### Custom Tools

```python
from trackrealties.agents.tools import BaseTool

class CustomMarketTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_market_analysis",
            description="Custom market analysis tool"
        )
    
    async def execute(self, deps, **kwargs):
        # Custom implementation
        return {"success": True, "data": analysis_result}

# Add to agent
agent = InvestorAgent(tools=[CustomMarketTool()])
```

### Validation Settings

```python
from trackrealties.validation.base import ResponseValidator

class InvestorValidator(ResponseValidator):
    async def validate_response(self, response, context, user_query):
        # Custom validation for investment advice
        return ValidationResult(...)

agent = InvestorAgent(validator=InvestorValidator())
```

## Integration with External Data

### Market Data Sources
- MLS feeds for comparable sales
- Economic indicators (employment, population)
- Interest rate and financing data
- Construction and permit data

### Property Data Sources
- Property records and tax assessments
- Rental market data (rent rolls, vacancy rates)
- Neighborhood demographics and amenities
- Crime statistics and school ratings

### Financial Data Sources
- Mortgage rate feeds
- Insurance cost estimates
- Property management fee schedules
- Maintenance and repair cost databases

## Performance Optimization

### Caching Strategy
- Market analysis results cached for 30 minutes
- Property valuations cached for 1 hour
- Risk assessments cached for 24 hours

### Batch Processing
- Multiple property analysis in single request
- Portfolio-level calculations
- Comparative market analysis

### Response Streaming
```python
async for chunk in agent.stream(
    "Analyze these 5 properties for investment potential",
    session_id="batch_analysis"
):
    print(chunk, end="", flush=True)
```

## Error Handling

### Common Scenarios
- **Missing Property Data**: Graceful degradation with assumptions
- **Market Data Unavailable**: Use regional or national averages
- **Calculation Errors**: Return error with explanation
- **External API Failures**: Fallback to cached data

### Validation Checks
- Property price reasonableness
- Rental income market validation
- Expense ratio verification
- Return metric sanity checks

## Testing

### Unit Tests
```bash
# Run investor agent tests
pytest tests/test_investor_agent.py -v

# Run specific test categories
pytest tests/test_investor_agent.py::TestInvestmentOpportunityTool -v
```

### Integration Tests
```bash
# Test complete investment workflows
pytest tests/test_investor_agent.py::TestInvestorAgentIntegration -v
```

### Performance Tests
```bash
# Test response times and accuracy
pytest tests/test_investor_agent.py -m performance
```

## Monitoring and Analytics

### Key Metrics
- Analysis accuracy vs. actual outcomes
- User satisfaction with recommendations
- Tool usage patterns
- Response time performance

### Logging
```python
import logging

# Enable detailed logging
logging.getLogger("trackrealties.agents.investor").setLevel(logging.DEBUG)
```

### Integration with Pydantic Logfire
```python
import logfire

logfire.configure()
agent = InvestorAgent(instrument=True)
```

## Best Practices

### For Investors
1. **Provide Complete Data**: More data leads to better analysis
2. **Consider Multiple Scenarios**: Don't rely on base case only
3. **Understand Risk Factors**: Review all risk categories
4. **Regular Portfolio Reviews**: Reassess holdings periodically
5. **Market Timing**: Use as guidance, not absolute rules

### For Developers
1. **Validate Inputs**: Always check data quality
2. **Handle Edge Cases**: Plan for unusual scenarios
3. **Cache Appropriately**: Balance freshness vs. performance
4. **Monitor Accuracy**: Track prediction vs. reality
5. **Update Models**: Regularly retrain with new data

## Troubleshooting

### Common Issues

1. **Unrealistic Projections**
   - Check input data accuracy
   - Verify market assumptions
   - Review calculation parameters

2. **High Risk Scores**
   - Examine individual risk factors
   - Consider mitigation strategies
   - Validate property and market data

3. **Low Returns**
   - Review expense assumptions
   - Check financing terms
   - Consider different strategies

### Debug Mode
```python
# Enable debug logging
agent = InvestorAgent(model="openai:gpt-4o")
agent.logger.setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
- **Machine Learning Models**: Predictive analytics for returns and risks
- **Real-Time Data Integration**: Live market feeds and pricing
- **Advanced Portfolio Analytics**: Modern portfolio theory application
- **Tax Optimization**: Detailed tax strategy recommendations
- **Automated Reporting**: Regular portfolio performance reports

### API Integrations
- Direct MLS access
- Real-time financing rates
- Insurance quote APIs
- Property management platforms

The InvestorAgent provides a comprehensive foundation for real estate investment analysis, combining quantitative rigor with practical investment wisdom to help investors make better decisions.