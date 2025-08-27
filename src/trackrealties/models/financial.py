"""Financial analysis models for TrackRealties AI Platform."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from .base import CustomBaseModel as BaseModel
from .base import TimestampMixin


class InvestmentParams(BaseModel):
    """Parameters for investment analysis."""

    # Purchase details
    purchase_price: Decimal = Field(..., gt=0, description="Property purchase price")
    down_payment_percent: float = Field(..., ge=0.0, le=1.0, description="Down payment as percentage of purchase price")
    closing_costs_percent: float = Field(
        default=0.03, ge=0.0, le=0.1, description="Closing costs as percentage of purchase price"
    )

    # Financing details
    loan_interest_rate: float = Field(..., ge=0.0, le=0.2, description="Annual loan interest rate")
    loan_term_years: int = Field(default=30, ge=1, le=50, description="Loan term in years")

    # Rental income
    monthly_rent: Decimal = Field(..., ge=0, description="Expected monthly rental income")
    vacancy_rate: float = Field(default=0.05, ge=0.0, le=0.5, description="Expected vacancy rate")
    annual_rent_increase: float = Field(default=0.03, ge=0.0, le=0.2, description="Expected annual rent increase")

    # Operating expenses
    property_tax_annual: Decimal = Field(..., ge=0, description="Annual property taxes")
    insurance_annual: Decimal = Field(..., ge=0, description="Annual insurance cost")
    maintenance_percent: float = Field(
        default=0.01, ge=0.0, le=0.1, description="Annual maintenance as percentage of property value"
    )
    property_management_percent: float = Field(
        default=0.08, ge=0.0, le=0.2, description="Property management fee as percentage of rent"
    )

    # Other expenses
    hoa_monthly: Decimal = Field(default=Decimal("0"), ge=0, description="Monthly HOA fees")
    utilities_monthly: Decimal = Field(default=Decimal("0"), ge=0, description="Monthly utilities if landlord pays")

    # Analysis parameters
    analysis_years: int = Field(default=10, ge=1, le=50, description="Number of years to analyze")
    property_appreciation_rate: float = Field(
        default=0.03, ge=0.0, le=0.2, description="Expected annual property appreciation"
    )

    @property
    def down_payment_amount(self) -> Decimal:
        """Calculate down payment amount."""
        return self.purchase_price * Decimal(str(self.down_payment_percent))

    @property
    def loan_amount(self) -> Decimal:
        """Calculate loan amount."""
        return self.purchase_price - self.down_payment_amount

    @property
    def closing_costs_amount(self) -> Decimal:
        """Calculate closing costs amount."""
        return self.purchase_price * Decimal(str(self.closing_costs_percent))

    @property
    def total_initial_investment(self) -> Decimal:
        """Calculate total initial cash investment."""
        return self.down_payment_amount + self.closing_costs_amount

    @property
    def effective_monthly_rent(self) -> Decimal:
        """Calculate effective monthly rent after vacancy."""
        return self.monthly_rent * Decimal(str(1 - self.vacancy_rate))


class CashFlowAnalysis(BaseModel, TimestampMixin):
    """Cash flow analysis for rental property investment."""

    # Analysis parameters
    investment_params: InvestmentParams = Field(..., description="Investment parameters used")

    # Monthly cash flow components
    monthly_rental_income: Decimal = Field(..., description="Monthly rental income")
    monthly_mortgage_payment: Decimal = Field(..., description="Monthly mortgage payment (P&I)")
    monthly_property_tax: Decimal = Field(..., description="Monthly property tax")
    monthly_insurance: Decimal = Field(..., description="Monthly insurance")
    monthly_maintenance: Decimal = Field(..., description="Monthly maintenance estimate")
    monthly_property_management: Decimal = Field(..., description="Monthly property management fee")
    monthly_hoa: Decimal = Field(default=Decimal("0"), description="Monthly HOA fees")
    monthly_utilities: Decimal = Field(default=Decimal("0"), description="Monthly utilities")
    monthly_other_expenses: Decimal = Field(default=Decimal("0"), description="Other monthly expenses")

    # Calculated values
    monthly_gross_income: Decimal = Field(..., description="Monthly gross rental income")
    monthly_total_expenses: Decimal = Field(..., description="Total monthly expenses")
    monthly_net_cash_flow: Decimal = Field(..., description="Monthly net cash flow")

    # Annual projections
    annual_cash_flows: List[Decimal] = Field(default_factory=list, description="Annual cash flows for analysis period")

    @property
    def monthly_cash_on_cash_return(self) -> float:
        """Calculate monthly cash-on-cash return."""
        if self.investment_params.total_initial_investment == 0:
            return 0.0
        return float(self.monthly_net_cash_flow / self.investment_params.total_initial_investment)

    @property
    def annual_cash_on_cash_return(self) -> float:
        """Calculate annual cash-on-cash return."""
        return self.monthly_cash_on_cash_return * 12

    @property
    def cap_rate(self) -> float:
        """Calculate capitalization rate."""
        annual_noi = self.monthly_net_cash_flow * 12 + (self.monthly_mortgage_payment * 12)  # Add back mortgage payment
        return float(annual_noi / self.investment_params.purchase_price)

    @property
    def gross_rent_multiplier(self) -> float:
        """Calculate gross rent multiplier."""
        annual_rent = self.monthly_rental_income * 12
        return float(self.investment_params.purchase_price / annual_rent)

    @property
    def debt_service_coverage_ratio(self) -> float:
        """Calculate debt service coverage ratio."""
        annual_noi = (self.monthly_gross_income - (self.monthly_total_expenses - self.monthly_mortgage_payment)) * 12
        annual_debt_service = self.monthly_mortgage_payment * 12
        if annual_debt_service == 0:
            return float("inf")
        return float(annual_noi / annual_debt_service)

    def is_cash_flow_positive(self) -> bool:
        """Check if property has positive cash flow."""
        return self.monthly_net_cash_flow > 0

    def get_break_even_rent(self) -> Decimal:
        """Calculate break-even monthly rent."""
        monthly_expenses_without_management = self.monthly_total_expenses - self.monthly_property_management
        # Solve for rent where: rent * (1 - vacancy) * (1 - mgmt_fee) = expenses
        vacancy_factor = Decimal(str(1 - self.investment_params.vacancy_rate))
        mgmt_factor = Decimal(str(1 - self.investment_params.property_management_percent))

        return monthly_expenses_without_management / (vacancy_factor * mgmt_factor)


class ROIProjection(BaseModel, TimestampMixin):
    """Return on investment projection analysis."""

    # Analysis parameters
    investment_params: InvestmentParams = Field(..., description="Investment parameters used")
    cash_flow_analysis: CashFlowAnalysis = Field(..., description="Cash flow analysis")

    # ROI Components
    total_cash_invested: Decimal = Field(..., description="Total initial cash investment")

    # Year-by-year projections
    yearly_projections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Year-by-year financial projections"
    )

    # Summary metrics
    total_roi_percent: float = Field(..., description="Total ROI percentage over analysis period")
    annualized_roi_percent: float = Field(..., description="Annualized ROI percentage")
    irr_percent: Optional[float] = Field(None, description="Internal rate of return")
    npv: Optional[Decimal] = Field(None, description="Net present value")

    # Risk metrics
    payback_period_years: Optional[float] = Field(None, description="Payback period in years")
    break_even_occupancy: float = Field(..., description="Break-even occupancy rate")

    @property
    def is_profitable(self) -> bool:
        """Check if investment is profitable."""
        return self.total_roi_percent > 0

    @property
    def beats_market_return(self, market_return: float = 0.07) -> bool:
        """Check if investment beats market return."""
        return self.annualized_roi_percent > market_return

    def get_year_projection(self, year: int) -> Optional[Dict[str, Any]]:
        """Get projection for a specific year."""
        if 0 <= year < len(self.yearly_projections):
            return self.yearly_projections[year]
        return None

    def calculate_total_return(self) -> Decimal:
        """Calculate total return over analysis period."""
        if not self.yearly_projections:
            return Decimal("0")

        total_cash_flows = sum(Decimal(str(year_data.get("net_cash_flow", 0))) for year_data in self.yearly_projections)

        final_year = self.yearly_projections[-1]
        property_value = Decimal(str(final_year.get("property_value", 0)))
        remaining_loan_balance = Decimal(str(final_year.get("loan_balance", 0)))
        net_sale_proceeds = property_value - remaining_loan_balance

        return total_cash_flows + net_sale_proceeds - self.total_cash_invested


class RiskAssessment(BaseModel, TimestampMixin):
    """Risk assessment for real estate investment."""

    # Investment context
    investment_params: InvestmentParams = Field(..., description="Investment parameters")
    property_address: str = Field(..., description="Property address")
    market_context: Dict[str, Any] = Field(default_factory=dict, description="Market context data")

    # Risk categories and scores (0-1, where 1 is highest risk)
    market_risk_score: float = Field(..., ge=0.0, le=1.0, description="Market volatility and trend risk")
    location_risk_score: float = Field(..., ge=0.0, le=1.0, description="Location-specific risk factors")
    property_risk_score: float = Field(..., ge=0.0, le=1.0, description="Property-specific risk factors")
    financial_risk_score: float = Field(..., ge=0.0, le=1.0, description="Financial structure risk")
    liquidity_risk_score: float = Field(..., ge=0.0, le=1.0, description="Liquidity and exit risk")

    # Overall risk assessment
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Weighted overall risk score")
    risk_level: Literal["low", "moderate", "high", "very_high"] = Field(..., description="Risk level classification")

    # Risk factors
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Suggested risk mitigation strategies")

    # Sensitivity analysis
    sensitivity_analysis: Dict[str, Any] = Field(default_factory=dict, description="Sensitivity analysis results")

    @field_validator("overall_risk_score")
    @classmethod
    def validate_overall_risk(cls, v: float, info) -> float:
        """Validate overall risk score is consistent with individual scores."""
        # This is a simplified validation - in practice, you'd calculate weighted average
        return v

    @property
    def is_low_risk(self) -> bool:
        """Check if investment is low risk."""
        return self.risk_level == "low"

    @property
    def is_high_risk(self) -> bool:
        """Check if investment is high risk."""
        return self.risk_level in ["high", "very_high"]

    @property
    def risk_adjusted_return_needed(self) -> float:
        """Calculate risk-adjusted return needed."""
        base_return = 0.05  # 5% base return
        risk_premium = self.overall_risk_score * 0.10  # Up to 10% risk premium
        return base_return + risk_premium

    def add_risk_factor(self, factor: str) -> None:
        """Add a risk factor."""
        if factor not in self.risk_factors:
            self.risk_factors.append(factor)

    def add_mitigation_strategy(self, strategy: str) -> None:
        """Add a mitigation strategy."""
        if strategy not in self.mitigation_strategies:
            self.mitigation_strategies.append(strategy)

    def get_risk_summary(self) -> str:
        """Get a summary of the risk assessment."""
        return (
            f"Overall Risk: {self.risk_level.title()} "
            f"(Score: {self.overall_risk_score:.2f}) | "
            f"{len(self.risk_factors)} factors identified | "
            f"{len(self.mitigation_strategies)} mitigation strategies"
        )


class MarketComparison(BaseModel):
    """Market comparison analysis for investment decision."""

    # Property being analyzed
    subject_property: Dict[str, Any] = Field(..., description="Subject property details")

    # Comparable properties
    comparable_properties: List[Dict[str, Any]] = Field(default_factory=list, description="Comparable properties")

    # Market metrics
    market_metrics: Dict[str, Any] = Field(default_factory=dict, description="Market-level metrics")

    # Comparison results
    price_per_sqft_comparison: Dict[str, float] = Field(default_factory=dict, description="Price per sq ft comparison")
    rent_comparison: Dict[str, float] = Field(default_factory=dict, description="Rent comparison")
    cap_rate_comparison: Dict[str, float] = Field(default_factory=dict, description="Cap rate comparison")

    # Valuation
    estimated_value_range: Dict[str, Decimal] = Field(default_factory=dict, description="Estimated value range")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in valuation")

    @property
    def is_good_deal(self) -> bool:
        """Determine if subject property is a good deal."""
        if not self.estimated_value_range:
            return False

        subject_price = Decimal(str(self.subject_property.get("price", 0)))
        estimated_low = self.estimated_value_range.get("low", Decimal("0"))

        return subject_price <= estimated_low

    @property
    def value_score(self) -> float:
        """Calculate value score (0-1, where 1 is best value)."""
        if not self.estimated_value_range:
            return 0.0

        subject_price = float(self.subject_property.get("price", 0))
        estimated_high = float(self.estimated_value_range.get("high", 0))
        estimated_low = float(self.estimated_value_range.get("low", 0))

        if estimated_high == estimated_low:
            return 0.5

        # Score based on where price falls in estimated range
        if subject_price <= estimated_low:
            return 1.0
        elif subject_price >= estimated_high:
            return 0.0
        else:
            return 1.0 - (subject_price - estimated_low) / (estimated_high - estimated_low)

    def add_comparable(self, comparable: Dict[str, Any]) -> None:
        """Add a comparable property."""
        self.comparable_properties.append(comparable)

    def get_comparison_summary(self) -> str:
        """Get a summary of the market comparison."""
        value_score_pct = self.value_score * 100
        return (
            f"Market Comparison: {len(self.comparable_properties)} comps | "
            f"Value Score: {value_score_pct:.0f}% | "
            f"Confidence: {self.confidence_level:.0%}"
        )


class PortfolioAnalysis(BaseModel, TimestampMixin):
    """Portfolio-level analysis for multiple properties."""

    # Portfolio composition
    properties: List[Dict[str, Any]] = Field(default_factory=list, description="Properties in portfolio")
    total_properties: int = Field(default=0, ge=0, description="Total number of properties")

    # Financial summary
    total_investment: Decimal = Field(default=Decimal("0"), description="Total cash invested")
    total_equity: Decimal = Field(default=Decimal("0"), description="Total equity value")
    total_monthly_cash_flow: Decimal = Field(default=Decimal("0"), description="Total monthly cash flow")

    # Performance metrics
    portfolio_roi: float = Field(default=0.0, description="Overall portfolio ROI")
    portfolio_cap_rate: float = Field(default=0.0, description="Portfolio-weighted cap rate")
    portfolio_cash_on_cash: float = Field(default=0.0, description="Portfolio cash-on-cash return")

    # Diversification metrics
    geographic_diversification: Dict[str, int] = Field(default_factory=dict, description="Properties by location")
    property_type_diversification: Dict[str, int] = Field(default_factory=dict, description="Properties by type")

    # Risk metrics
    portfolio_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall portfolio risk")
    concentration_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="Concentration risk score")

    @property
    def is_diversified(self) -> bool:
        """Check if portfolio is well diversified."""
        # Simple diversification check - no single location > 50%
        if not self.geographic_diversification:
            return True

        max_concentration = max(self.geographic_diversification.values()) / self.total_properties
        return max_concentration <= 0.5

    @property
    def average_property_value(self) -> Decimal:
        """Calculate average property value."""
        if self.total_properties == 0:
            return Decimal("0")
        return self.total_equity / self.total_properties

    def add_property(self, property_data: Dict[str, Any]) -> None:
        """Add a property to the portfolio."""
        self.properties.append(property_data)
        self.total_properties += 1
        self._recalculate_metrics()

    def _recalculate_metrics(self) -> None:
        """Recalculate portfolio metrics."""
        # This would contain the logic to recalculate all portfolio metrics
        # Implementation would depend on the specific property data structure
        pass

    def get_portfolio_summary(self) -> str:
        """Get a summary of the portfolio."""
        return (
            f"Portfolio: {self.total_properties} properties | "
            f"Total Investment: ${self.total_investment:,.0f} | "
            f"Monthly Cash Flow: ${self.total_monthly_cash_flow:,.0f} | "
            f"ROI: {self.portfolio_roi:.1%}"
        )
