"""
Enhanced Risk Modeling and Scenario Analysis for ROI Tool
Advanced risk assessment with multiple market conditions and sensitivity analysis
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..models.financial import InvestmentParams, RiskAssessment, ROIProjection
from .financial_engine import FinancialAnalyticsEngine

logger = logging.getLogger(__name__)


@dataclass
class ScenarioParameters:
    """Parameters for different market scenarios."""

    name: str
    appreciation_rate: float
    rent_growth_rate: float
    vacancy_rate: float
    interest_rate_adjustment: float
    maintenance_multiplier: float
    description: str


@dataclass
class ScenarioResult:
    """Results from a specific scenario analysis."""

    scenario_name: str
    irr_percent: float
    cash_on_cash_return: float
    total_roi_percent: float
    monthly_cash_flow: float
    break_even_months: int
    risk_level: str
    key_metrics: Dict[str, float]


@dataclass
class SensitivityAnalysis:
    """Sensitivity analysis results for key variables."""

    rent_sensitivity: Dict[str, float]  # % change in IRR for rent changes
    appreciation_sensitivity: Dict[str, float]  # % change in IRR for appreciation changes
    vacancy_sensitivity: Dict[str, float]  # % change in IRR for vacancy changes
    interest_sensitivity: Dict[str, float]  # % change in IRR for interest rate changes
    worst_case_irr: float
    best_case_irr: float
    base_case_irr: float


class EnhancedRiskModelingEngine:
    """Advanced risk modeling and scenario analysis engine."""

    def __init__(self):
        self.financial_engine = FinancialAnalyticsEngine()
        self.logger = logging.getLogger(__name__)

    def run_scenario_analysis(self, base_params: InvestmentParams) -> List[ScenarioResult]:
        """Run comprehensive scenario analysis with multiple market conditions."""

        scenarios = [
            ScenarioParameters(
                name="optimistic",
                appreciation_rate=0.06,  # 6% appreciation
                rent_growth_rate=0.04,  # 4% rent growth
                vacancy_rate=3.0,  # 3% vacancy
                interest_rate_adjustment=0.0,  # No rate change
                maintenance_multiplier=0.8,  # Lower maintenance
                description="Strong market conditions with high appreciation and low vacancy",
            ),
            ScenarioParameters(
                name="base_case",
                appreciation_rate=0.03,  # 3% appreciation
                rent_growth_rate=0.025,  # 2.5% rent growth
                vacancy_rate=5.0,  # 5% vacancy
                interest_rate_adjustment=0.0,  # No rate change
                maintenance_multiplier=1.0,  # Normal maintenance
                description="Expected market conditions with moderate growth",
            ),
            ScenarioParameters(
                name="conservative",
                appreciation_rate=0.015,  # 1.5% appreciation
                rent_growth_rate=0.015,  # 1.5% rent growth
                vacancy_rate=8.0,  # 8% vacancy
                interest_rate_adjustment=0.01,  # +1% rate increase
                maintenance_multiplier=1.2,  # Higher maintenance
                description="Cautious outlook with slower growth and higher costs",
            ),
            ScenarioParameters(
                name="recession",
                appreciation_rate=-0.02,  # -2% depreciation
                rent_growth_rate=0.0,  # No rent growth
                vacancy_rate=15.0,  # 15% vacancy
                interest_rate_adjustment=0.02,  # +2% rate increase
                maintenance_multiplier=1.5,  # Much higher maintenance
                description="Economic downturn with declining values and high vacancy",
            ),
            ScenarioParameters(
                name="boom",
                appreciation_rate=0.08,  # 8% appreciation
                rent_growth_rate=0.06,  # 6% rent growth
                vacancy_rate=2.0,  # 2% vacancy
                interest_rate_adjustment=-0.01,  # -1% rate decrease
                maintenance_multiplier=0.7,  # Lower maintenance
                description="Hot market with rapid appreciation and tight supply",
            ),
        ]

        results = []

        for scenario in scenarios:
            try:
                # Create modified parameters for this scenario
                scenario_params = self._create_scenario_params(base_params, scenario)

                # Calculate ROI for this scenario
                roi_projection = self.financial_engine.calculate_roi_projection(scenario_params)
                cash_flow_analysis = self.financial_engine.analyze_cash_flow(scenario_params)

                # Calculate additional metrics
                break_even_months = self._calculate_break_even_months(scenario_params)
                risk_level = self._assess_scenario_risk(scenario, roi_projection)

                scenario_result = ScenarioResult(
                    scenario_name=scenario.name,
                    irr_percent=roi_projection.irr_percent,
                    cash_on_cash_return=float(
                        cash_flow_analysis.monthly_net_cash_flow * 12 / roi_projection.total_cash_invested * 100
                    ),
                    total_roi_percent=roi_projection.total_roi_percent,
                    monthly_cash_flow=float(cash_flow_analysis.monthly_net_cash_flow),
                    break_even_months=break_even_months,
                    risk_level=risk_level,
                    key_metrics={
                        "appreciation_rate": scenario.appreciation_rate * 100,
                        "rent_growth_rate": scenario.rent_growth_rate * 100,
                        "vacancy_rate": scenario.vacancy_rate,
                        "cap_rate": self._calculate_cap_rate(scenario_params),
                        "debt_service_ratio": self._calculate_debt_service_ratio(scenario_params),
                    },
                )

                results.append(scenario_result)

            except Exception as e:
                logger.error(f"Scenario analysis failed for {scenario.name}: {e}")
                continue

        return results

    def perform_sensitivity_analysis(self, base_params: InvestmentParams) -> SensitivityAnalysis:
        """Perform sensitivity analysis on key investment variables."""

        # Base case calculation
        base_roi = self.financial_engine.calculate_roi_projection(base_params)
        base_irr = base_roi.irr_percent

        # Sensitivity ranges
        rent_changes = [-20, -10, -5, 0, 5, 10, 20]  # % changes
        appreciation_changes = [-3, -2, -1, 0, 1, 2, 3]  # % point changes
        vacancy_changes = [-5, -3, -1, 0, 2, 5, 10]  # % point changes
        interest_changes = [-2, -1, -0.5, 0, 0.5, 1, 2]  # % point changes

        # Calculate sensitivities
        rent_sensitivity = self._calculate_rent_sensitivity(base_params, base_irr, rent_changes)
        appreciation_sensitivity = self._calculate_appreciation_sensitivity(base_params, base_irr, appreciation_changes)
        vacancy_sensitivity = self._calculate_vacancy_sensitivity(base_params, base_irr, vacancy_changes)
        interest_sensitivity = self._calculate_interest_sensitivity(base_params, base_irr, interest_changes)

        # Calculate worst and best case scenarios
        worst_case_irr = self._calculate_worst_case_irr(base_params)
        best_case_irr = self._calculate_best_case_irr(base_params)

        return SensitivityAnalysis(
            rent_sensitivity=rent_sensitivity,
            appreciation_sensitivity=appreciation_sensitivity,
            vacancy_sensitivity=vacancy_sensitivity,
            interest_sensitivity=interest_sensitivity,
            worst_case_irr=worst_case_irr,
            best_case_irr=best_case_irr,
            base_case_irr=base_irr,
        )

    def assess_advanced_risk_factors(self, params: InvestmentParams, location: Optional[str] = None) -> Dict[str, Any]:
        """Assess advanced risk factors beyond basic financial metrics."""

        risk_assessment = {
            "financial_risks": self._assess_financial_risks(params),
            "market_risks": self._assess_market_risks(params, location),
            "property_risks": self._assess_property_risks(params),
            "macroeconomic_risks": self._assess_macro_risks(params),
            "liquidity_risks": self._assess_liquidity_risks(params),
            "regulatory_risks": self._assess_regulatory_risks(location),
            "overall_risk_score": 0.0,
            "risk_mitigation_strategies": [],
        }

        # Calculate overall risk score (weighted average)
        weights = {
            "financial_risks": 0.25,
            "market_risks": 0.20,
            "property_risks": 0.20,
            "macroeconomic_risks": 0.15,
            "liquidity_risks": 0.10,
            "regulatory_risks": 0.10,
        }

        overall_score = sum(
            risk_assessment[risk_type]["score"] * weight
            for risk_type, weight in weights.items()
            if risk_type in risk_assessment and "score" in risk_assessment[risk_type]
        )

        risk_assessment["overall_risk_score"] = overall_score
        risk_assessment["risk_mitigation_strategies"] = self._generate_mitigation_strategies(risk_assessment)

        return risk_assessment

    def _create_scenario_params(self, base_params: InvestmentParams, scenario: ScenarioParameters) -> InvestmentParams:
        """Create modified parameters for a specific scenario."""

        # Create a copy with modified values
        scenario_params = InvestmentParams(
            purchase_price=base_params.purchase_price,
            monthly_rent=base_params.monthly_rent,
            property_tax_annual=base_params.property_tax_annual,
            insurance_annual=base_params.insurance_annual,
            maintenance_percent=base_params.maintenance_percent * scenario.maintenance_multiplier,
            down_payment_percent=base_params.down_payment_percent,
            loan_interest_rate=base_params.loan_interest_rate + scenario.interest_rate_adjustment,
            loan_term_years=base_params.loan_term_years,
            analysis_years=base_params.analysis_years,
            property_appreciation_rate=scenario.appreciation_rate,
            annual_rent_increase=scenario.rent_growth_rate,
            vacancy_rate=scenario.vacancy_rate,
            property_management_percent=base_params.property_management_percent,
            hoa_monthly=base_params.hoa_monthly,
            utilities_monthly=base_params.utilities_monthly,
        )

        return scenario_params

    def _calculate_break_even_months(self, params: InvestmentParams) -> int:
        """Calculate months until break-even (positive cumulative cash flow)."""

        cash_flow_analysis = self.financial_engine.analyze_cash_flow(params)
        monthly_cash_flow = float(cash_flow_analysis.monthly_net_cash_flow)
        initial_investment = float(params.total_initial_investment)

        if monthly_cash_flow <= 0:
            return 999  # Never breaks even

        break_even_months = int(initial_investment / monthly_cash_flow)
        return min(break_even_months, 999)  # Cap at 999 months

    def _assess_scenario_risk(self, scenario: ScenarioParameters, roi_projection: ROIProjection) -> str:
        """Assess risk level for a specific scenario."""

        irr = roi_projection.irr_percent

        if scenario.name == "recession":
            return "very_high"
        elif scenario.name == "conservative" or irr < 2:
            return "high"
        elif scenario.name == "base_case" or 2 <= irr < 5:
            return "moderate"
        elif scenario.name == "optimistic" or 5 <= irr < 8:
            return "low"
        else:  # boom scenario or very high IRR
            return "very_low"

    def _calculate_rent_sensitivity(
        self, base_params: InvestmentParams, base_irr: float, changes: List[int]
    ) -> Dict[str, float]:
        """Calculate IRR sensitivity to rent changes."""

        sensitivity = {}

        for change in changes:
            try:
                modified_params = InvestmentParams(
                    purchase_price=base_params.purchase_price,
                    monthly_rent=base_params.monthly_rent * (1 + change / 100),
                    property_tax_annual=base_params.property_tax_annual,
                    insurance_annual=base_params.insurance_annual,
                    maintenance_percent=base_params.maintenance_percent,
                    down_payment_percent=base_params.down_payment_percent,
                    loan_interest_rate=base_params.loan_interest_rate,
                    loan_term_years=base_params.loan_term_years,
                    analysis_years=base_params.analysis_years,
                    property_appreciation_rate=base_params.property_appreciation_rate,
                    annual_rent_increase=base_params.annual_rent_increase,
                    vacancy_rate=base_params.vacancy_rate,
                    property_management_percent=base_params.property_management_percent,
                    hoa_monthly=base_params.hoa_monthly,
                    utilities_monthly=base_params.utilities_monthly,
                )

                roi_projection = self.financial_engine.calculate_roi_projection(modified_params)
                irr_change = roi_projection.irr_percent - base_irr
                sensitivity[f"{change:+d}%"] = irr_change

            except Exception as e:
                logger.warning(f"Failed to calculate rent sensitivity for {change}%: {e}")
                sensitivity[f"{change:+d}%"] = 0.0

        return sensitivity

    def _calculate_appreciation_sensitivity(
        self, base_params: InvestmentParams, base_irr: float, changes: List[int]
    ) -> Dict[str, float]:
        """Calculate IRR sensitivity to appreciation rate changes."""

        sensitivity = {}

        for change in changes:
            try:
                new_appreciation = base_params.property_appreciation_rate + (change / 100)

                modified_params = InvestmentParams(
                    purchase_price=base_params.purchase_price,
                    monthly_rent=base_params.monthly_rent,
                    property_tax_annual=base_params.property_tax_annual,
                    insurance_annual=base_params.insurance_annual,
                    maintenance_percent=base_params.maintenance_percent,
                    down_payment_percent=base_params.down_payment_percent,
                    loan_interest_rate=base_params.loan_interest_rate,
                    loan_term_years=base_params.loan_term_years,
                    analysis_years=base_params.analysis_years,
                    property_appreciation_rate=new_appreciation,
                    annual_rent_increase=base_params.annual_rent_increase,
                    vacancy_rate=base_params.vacancy_rate,
                    property_management_percent=base_params.property_management_percent,
                    hoa_monthly=base_params.hoa_monthly,
                    utilities_monthly=base_params.utilities_monthly,
                )

                roi_projection = self.financial_engine.calculate_roi_projection(modified_params)
                irr_change = roi_projection.irr_percent - base_irr
                sensitivity[f"{change:+d}pp"] = irr_change

            except Exception as e:
                logger.warning(f"Failed to calculate appreciation sensitivity for {change}pp: {e}")
                sensitivity[f"{change:+d}pp"] = 0.0

        return sensitivity

    def _calculate_vacancy_sensitivity(
        self, base_params: InvestmentParams, base_irr: float, changes: List[int]
    ) -> Dict[str, float]:
        """Calculate IRR sensitivity to vacancy rate changes."""

        sensitivity = {}

        for change in changes:
            try:
                new_vacancy = max(0, base_params.vacancy_rate + change)

                modified_params = InvestmentParams(
                    purchase_price=base_params.purchase_price,
                    monthly_rent=base_params.monthly_rent,
                    property_tax_annual=base_params.property_tax_annual,
                    insurance_annual=base_params.insurance_annual,
                    maintenance_percent=base_params.maintenance_percent,
                    down_payment_percent=base_params.down_payment_percent,
                    loan_interest_rate=base_params.loan_interest_rate,
                    loan_term_years=base_params.loan_term_years,
                    analysis_years=base_params.analysis_years,
                    property_appreciation_rate=base_params.property_appreciation_rate,
                    annual_rent_increase=base_params.annual_rent_increase,
                    vacancy_rate=new_vacancy,
                    property_management_percent=base_params.property_management_percent,
                    hoa_monthly=base_params.hoa_monthly,
                    utilities_monthly=base_params.utilities_monthly,
                )

                roi_projection = self.financial_engine.calculate_roi_projection(modified_params)
                irr_change = roi_projection.irr_percent - base_irr
                sensitivity[f"{change:+d}pp"] = irr_change

            except Exception as e:
                logger.warning(f"Failed to calculate vacancy sensitivity for {change}pp: {e}")
                sensitivity[f"{change:+d}pp"] = 0.0

        return sensitivity

    def _calculate_interest_sensitivity(
        self, base_params: InvestmentParams, base_irr: float, changes: List[float]
    ) -> Dict[str, float]:
        """Calculate IRR sensitivity to interest rate changes."""

        sensitivity = {}

        for change in changes:
            try:
                new_rate = base_params.loan_interest_rate + (change / 100)

                modified_params = InvestmentParams(
                    purchase_price=base_params.purchase_price,
                    monthly_rent=base_params.monthly_rent,
                    property_tax_annual=base_params.property_tax_annual,
                    insurance_annual=base_params.insurance_annual,
                    maintenance_percent=base_params.maintenance_percent,
                    down_payment_percent=base_params.down_payment_percent,
                    loan_interest_rate=new_rate,
                    loan_term_years=base_params.loan_term_years,
                    analysis_years=base_params.analysis_years,
                    property_appreciation_rate=base_params.property_appreciation_rate,
                    annual_rent_increase=base_params.annual_rent_increase,
                    vacancy_rate=base_params.vacancy_rate,
                    property_management_percent=base_params.property_management_percent,
                    hoa_monthly=base_params.hoa_monthly,
                    utilities_monthly=base_params.utilities_monthly,
                )

                roi_projection = self.financial_engine.calculate_roi_projection(modified_params)
                irr_change = roi_projection.irr_percent - base_irr
                sensitivity[f"{change:+.1f}pp"] = irr_change

            except Exception as e:
                logger.warning(f"Failed to calculate interest sensitivity for {change}pp: {e}")
                sensitivity[f"{change:+.1f}pp"] = 0.0

        return sensitivity

    def _calculate_worst_case_irr(self, base_params: InvestmentParams) -> float:
        """Calculate worst-case scenario IRR."""

        worst_case_params = InvestmentParams(
            purchase_price=base_params.purchase_price,
            monthly_rent=base_params.monthly_rent * 0.8,  # 20% rent reduction
            property_tax_annual=base_params.property_tax_annual * 1.2,  # 20% tax increase
            insurance_annual=base_params.insurance_annual * 1.3,  # 30% insurance increase
            maintenance_percent=base_params.maintenance_percent * 2.0,  # Double maintenance
            down_payment_percent=base_params.down_payment_percent,
            loan_interest_rate=base_params.loan_interest_rate + 0.02,  # +2% interest
            loan_term_years=base_params.loan_term_years,
            analysis_years=base_params.analysis_years,
            property_appreciation_rate=-0.02,  # -2% depreciation
            annual_rent_increase=0.0,  # No rent growth
            vacancy_rate=15.0,  # 15% vacancy
            property_management_percent=base_params.property_management_percent * 1.5,
            hoa_monthly=base_params.hoa_monthly * 1.2,
            utilities_monthly=base_params.utilities_monthly * 1.2,
        )

        try:
            roi_projection = self.financial_engine.calculate_roi_projection(worst_case_params)
            return roi_projection.irr_percent
        except Exception:
            return -50.0  # Assume very poor performance if calculation fails

    def _calculate_best_case_irr(self, base_params: InvestmentParams) -> float:
        """Calculate best-case scenario IRR."""

        best_case_params = InvestmentParams(
            purchase_price=base_params.purchase_price,
            monthly_rent=base_params.monthly_rent * 1.2,  # 20% rent increase
            property_tax_annual=base_params.property_tax_annual,
            insurance_annual=base_params.insurance_annual,
            maintenance_percent=base_params.maintenance_percent * 0.7,  # Lower maintenance
            down_payment_percent=base_params.down_payment_percent,
            loan_interest_rate=base_params.loan_interest_rate - 0.01,  # -1% interest
            loan_term_years=base_params.loan_term_years,
            analysis_years=base_params.analysis_years,
            property_appreciation_rate=0.08,  # 8% appreciation
            annual_rent_increase=0.05,  # 5% rent growth
            vacancy_rate=2.0,  # 2% vacancy
            property_management_percent=base_params.property_management_percent,
            hoa_monthly=base_params.hoa_monthly,
            utilities_monthly=base_params.utilities_monthly,
        )

        try:
            roi_projection = self.financial_engine.calculate_roi_projection(best_case_params)
            return roi_projection.irr_percent
        except Exception:
            return 20.0  # Assume good performance if calculation fails

    def _assess_financial_risks(self, params: InvestmentParams) -> Dict[str, Any]:
        """Assess financial risk factors."""

        cash_flow = self.financial_engine.analyze_cash_flow(params)
        ltv_ratio = (1 - params.down_payment_percent / 100) * 100

        risks = []
        score = 0.0

        # Leverage risk
        if ltv_ratio > 90:
            risks.append("Very high leverage (LTV > 90%)")
            score += 0.3
        elif ltv_ratio > 80:
            risks.append("High leverage (LTV > 80%)")
            score += 0.2

        # Cash flow risk
        monthly_cf = float(cash_flow.monthly_net_cash_flow)
        if monthly_cf < 0:
            risks.append("Negative cash flow")
            score += 0.4
        elif monthly_cf < 200:
            risks.append("Minimal cash flow buffer")
            score += 0.2

        # Debt service coverage
        annual_noi = (
            float(cash_flow.monthly_gross_income - cash_flow.monthly_total_expenses + cash_flow.monthly_mortgage_payment) * 12
        )
        annual_debt_service = float(cash_flow.monthly_mortgage_payment) * 12
        dscr = annual_noi / annual_debt_service if annual_debt_service > 0 else 0

        if dscr < 1.2:
            risks.append("Low debt service coverage ratio")
            score += 0.3

        return {
            "score": min(score, 1.0),
            "risks": risks,
            "metrics": {"ltv_ratio": ltv_ratio, "monthly_cash_flow": monthly_cf, "debt_service_coverage": dscr},
        }

    def _assess_market_risks(self, params: InvestmentParams, location: Optional[str]) -> Dict[str, Any]:
        """Assess market-related risks."""

        risks = []
        score = 0.0

        # Appreciation rate risk
        if params.property_appreciation_rate < 0:
            risks.append("Negative property appreciation expected")
            score += 0.4
        elif params.property_appreciation_rate < 0.02:
            risks.append("Low property appreciation rate")
            score += 0.2

        # Rent growth risk
        if params.annual_rent_increase < 0.02:
            risks.append("Low or no rent growth expected")
            score += 0.2

        # Vacancy risk
        if params.vacancy_rate > 10:
            risks.append("High vacancy rate assumption")
            score += 0.3
        elif params.vacancy_rate > 7:
            risks.append("Elevated vacancy rate")
            score += 0.15

        # Location risk (basic assessment)
        if location and any(term in location.lower() for term in ["rural", "small town"]):
            risks.append("Rural or small market location")
            score += 0.2

        return {
            "score": min(score, 1.0),
            "risks": risks,
            "metrics": {
                "appreciation_rate": params.property_appreciation_rate * 100,
                "rent_growth_rate": params.annual_rent_increase * 100,
                "vacancy_rate": params.vacancy_rate,
            },
        }

    def _assess_property_risks(self, params: InvestmentParams) -> Dict[str, Any]:
        """Assess property-specific risks."""

        risks = []
        score = 0.0

        # Maintenance risk
        if params.maintenance_percent > 0.02:
            risks.append("High maintenance cost assumption")
            score += 0.2

        # Property management risk
        if params.property_management_percent > 10:
            risks.append("High property management fees")
            score += 0.15

        # HOA/utility risk
        total_monthly_fixed = float(params.hoa_monthly + params.utilities_monthly)
        monthly_rent = float(params.monthly_rent)

        if total_monthly_fixed > monthly_rent * 0.15:
            risks.append("High fixed costs relative to rent")
            score += 0.2

        return {
            "score": min(score, 1.0),
            "risks": risks,
            "metrics": {
                "maintenance_percent": params.maintenance_percent * 100,
                "management_percent": params.property_management_percent,
                "fixed_cost_ratio": total_monthly_fixed / monthly_rent * 100,
            },
        }

    def _assess_macro_risks(self, params: InvestmentParams) -> Dict[str, Any]:
        """Assess macroeconomic risks."""

        risks = []
        score = 0.2  # Base macroeconomic risk

        # Interest rate risk
        if params.loan_interest_rate > 0.07:
            risks.append("High interest rate environment")
            score += 0.2
        elif params.loan_interest_rate > 0.06:
            risks.append("Elevated interest rates")
            score += 0.1

        # Economic cycle risk
        risks.append("General economic cycle risk")

        return {"score": min(score, 1.0), "risks": risks, "metrics": {"interest_rate": params.loan_interest_rate * 100}}

    def _assess_liquidity_risks(self, params: InvestmentParams) -> Dict[str, Any]:
        """Assess liquidity-related risks."""

        risks = []
        score = 0.5  # Real estate has inherent liquidity risk

        # High-value property liquidity risk
        purchase_price = float(params.purchase_price)
        if purchase_price > 1000000:
            risks.append("High-value property may have limited buyer pool")
            score += 0.2
        elif purchase_price > 500000:
            risks.append("Above-average price point")
            score += 0.1

        risks.append("Real estate inherent liquidity constraints")

        return {"score": min(score, 1.0), "risks": risks, "metrics": {"purchase_price": purchase_price}}

    def _assess_regulatory_risks(self, location: Optional[str]) -> Dict[str, Any]:
        """Assess regulatory and legal risks."""

        risks = []
        score = 0.1  # Base regulatory risk

        # Location-based regulatory risks
        if location:
            location_lower = location.lower()

            if any(city in location_lower for city in ["san francisco", "new york", "berkeley"]):
                risks.append("High rent control/tenant protection jurisdiction")
                score += 0.3
            elif any(state in location_lower for state in ["california", "new york"]):
                risks.append("Tenant-friendly regulation environment")
                score += 0.2

        risks.append("General regulatory and tax policy risk")

        return {"score": min(score, 1.0), "risks": risks, "metrics": {"location": location or "Unknown"}}

    def _generate_mitigation_strategies(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies based on assessment."""

        strategies = []

        # Financial risk mitigation
        if risk_assessment["financial_risks"]["score"] > 0.3:
            strategies.extend(
                [
                    "Maintain larger cash reserves for negative cash flow periods",
                    "Consider lower leverage or larger down payment",
                    "Establish line of credit for emergency expenses",
                ]
            )

        # Market risk mitigation
        if risk_assessment["market_risks"]["score"] > 0.3:
            strategies.extend(
                [
                    "Diversify across multiple markets and property types",
                    "Focus on recession-resistant rental markets",
                    "Consider shorter-term financing to reduce interest rate risk",
                ]
            )

        # Property risk mitigation
        if risk_assessment["property_risks"]["score"] > 0.3:
            strategies.extend(
                [
                    "Conduct thorough property inspection and due diligence",
                    "Budget additional reserves for maintenance and repairs",
                    "Consider property management to reduce operational burden",
                ]
            )

        # General strategies
        strategies.extend(
            [
                "Regular market monitoring and analysis",
                "Maintain adequate insurance coverage",
                "Screen tenants thoroughly to reduce vacancy risk",
                "Consider professional property management",
            ]
        )

        return list(set(strategies))  # Remove duplicates

    def _calculate_cap_rate(self, params: InvestmentParams) -> float:
        """Calculate capitalization rate."""

        cash_flow = self.financial_engine.analyze_cash_flow(params)
        annual_noi = (
            float(cash_flow.monthly_gross_income - cash_flow.monthly_total_expenses + cash_flow.monthly_mortgage_payment) * 12
        )
        purchase_price = float(params.purchase_price)
        return (annual_noi / purchase_price * 100) if purchase_price > 0 else 0.0

    def _calculate_debt_service_ratio(self, params: InvestmentParams) -> float:
        """Calculate debt service coverage ratio."""

        cash_flow = self.financial_engine.analyze_cash_flow(params)
        annual_noi = (
            float(cash_flow.monthly_gross_income - cash_flow.monthly_total_expenses + cash_flow.monthly_mortgage_payment) * 12
        )
        annual_debt_service = float(cash_flow.monthly_mortgage_payment) * 12
        return (annual_noi / annual_debt_service) if annual_debt_service > 0 else 0.0
