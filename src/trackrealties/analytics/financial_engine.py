import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.config import get_settings
from ..models.financial import (CashFlowAnalysis, InvestmentParams,
                                RiskAssessment, ROIProjection)
from ..models.market import MarketDataPoint
from ..models.property import PropertyListing
from .cma_engine import ComparativeMarketAnalysis
from .financial_metrics import FinancialCalculator
from .market_intelligence import MarketIntelligenceEngine, MarketTrend

logger = logging.getLogger(__name__)
settings = get_settings()


class FinancialAnalyticsEngine:
    """
    Core financial analytics engine for real estate investment analysis.
    Provides ROI calculations, cash flow analysis, and market intelligence.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculator = FinancialCalculator()
        self.market_engine = MarketIntelligenceEngine()
        self.cma_engine = ComparativeMarketAnalysis()

    def calculate_roi_projection(
        self, params: InvestmentParams, market_data: Optional[List[MarketDataPoint]] = None
    ) -> ROIProjection:
        """Calculate comprehensive ROI projection with multiple scenarios."""
        try:
            # Base calculations
            purchase_price = params.purchase_price
            loan_amount = params.loan_amount  # Use property method
            total_investment = params.total_initial_investment  # Use property method

            # Calculate monthly mortgage payment
            monthly_payment = self.calculator.calculate_mortgage_payment(
                loan_amount, params.loan_interest_rate, params.loan_term_years
            )

            # Annual income and expenses
            effective_rental_income = params.effective_monthly_rent * 12  # Use property method

            annual_expenses = (
                params.property_tax_annual + params.insurance_annual + (purchase_price * Decimal(str(params.maintenance_percent))) + (params.utilities_monthly * 12) + (params.hoa_monthly * 12) + (effective_rental_income * Decimal(str(params.property_management_percent)) / Decimal("100"))  
                # Fix decimal/float mixing
            )

            # Net Operating Income
            noi = effective_rental_income - annual_expenses
            annual_debt_service = monthly_payment * 12
            annual_cash_flow = noi - annual_debt_service

            # Multi-year projections
            projections = self._calculate_multi_year_projections(
                params, noi, annual_cash_flow, purchase_price, loan_amount
            )

            # Calculate IRR - include property sale proceeds in final year
            cash_flows = [-total_investment]

            # Add annual cash flows for years 1-4
            for proj in projections[:4]:
                cash_flows.append(proj["cash_flow"])

            # For final year (year 5), add cash flow + equity from property sale
            if len(projections) >= 5:
                final_year_cash_flow = projections[4]["cash_flow"]
                final_year_equity = projections[4]["equity"]  # Property value - remaining loan
                final_year_total = final_year_cash_flow + final_year_equity
                cash_flows.append(final_year_total)

            irr = self.calculator.calculate_irr(cash_flows) * 100

            # Create cash flow analysis first
            cash_flow_analysis = self.analyze_cash_flow(params)

            # Create yearly projections in the expected format
            yearly_projections_formatted = []
            for proj in projections:
                yearly_projections_formatted.append(
                    {
                        "year": proj["year"],
                        "property_value": proj["property_value"],
                        "net_cash_flow": proj["cash_flow"],
                        "loan_balance": proj["remaining_loan"],
                        "equity": proj["equity"],
                        "total_return": proj["total_return"],
                    }
                )

            # Calculate total ROI metrics
            total_roi_percent = (
                (float(Decimal(str(projections[-1]["total_return"])) / total_investment) * 100) if projections else 0.0
            )
            annualized_roi_percent = total_roi_percent / len(projections) if projections else 0.0

            return ROIProjection(
                investment_params=params,
                cash_flow_analysis=cash_flow_analysis,
                total_cash_invested=total_investment,
                yearly_projections=yearly_projections_formatted,
                total_roi_percent=total_roi_percent,
                annualized_roi_percent=annualized_roi_percent,
                irr_percent=float(irr),
                break_even_occupancy=50.0,  # Placeholder calculation
            )

        except Exception as e:
            self.logger.error(f"ROI calculation failed: {e}")
            raise

    def analyze_cash_flow(self, params: InvestmentParams) -> CashFlowAnalysis:
        """Perform detailed cash flow analysis."""
        try:
            # Calculate all components
            purchase_price = params.purchase_price
            loan_amount = params.loan_amount  # Use property method

            # Monthly mortgage payment
            monthly_mortgage = self.calculator.calculate_mortgage_payment(
                loan_amount, params.loan_interest_rate, params.loan_term_years
            )

            # Monthly income
            monthly_rental_income = params.monthly_rent
            effective_monthly_income = params.effective_monthly_rent  # Use property method

            # Monthly expenses
            monthly_property_tax = params.property_tax_annual / Decimal("12")
            monthly_insurance = params.insurance_annual / Decimal("12")
            monthly_maintenance = purchase_price * Decimal(str(params.maintenance_percent)) / Decimal("12")
            monthly_management = (
                effective_monthly_income * Decimal(str(params.property_management_percent)) / Decimal("100")
            )  # Fix decimal/float mixing

            total_monthly_expenses = (
                monthly_property_tax + monthly_insurance + monthly_maintenance + params.utilities_monthly + params.hoa_monthly + monthly_management
            )

            # Net cash flow
            monthly_gross_income = monthly_rental_income
            monthly_total_expenses = total_monthly_expenses + monthly_mortgage
            monthly_net_cash_flow = effective_monthly_income - monthly_total_expenses

            return CashFlowAnalysis(
                investment_params=params,
                monthly_rental_income=monthly_rental_income,
                monthly_mortgage_payment=monthly_mortgage,
                monthly_property_tax=monthly_property_tax,
                monthly_insurance=monthly_insurance,
                monthly_maintenance=monthly_maintenance,
                monthly_property_management=monthly_management,
                monthly_hoa=params.hoa_monthly,
                monthly_utilities=params.utilities_monthly,
                monthly_other_expenses=Decimal("0"),
                monthly_gross_income=monthly_gross_income,
                monthly_total_expenses=monthly_total_expenses,
                monthly_net_cash_flow=monthly_net_cash_flow,
            )

        except Exception as e:
            self.logger.error(f"Cash flow analysis failed: {e}")
            raise

    def assess_investment_risk(
        self,
        params: InvestmentParams,
        location: Optional[str] = None,
        market_data: Optional[List[MarketDataPoint]] = None,
    ) -> RiskAssessment:
        """Assess investment risk factors."""
        try:
            risk_factors = []
            risk_score = 0  # 0-100 scale, higher = riskier

            # Leverage risk
            ltv_ratio = (1 - params.down_payment_percent / 100) * 100  # This is float arithmetic, should be fine
            if ltv_ratio > 80:
                risk_factors.append("High leverage (LTV > 80%)")
                risk_score += 15
            elif ltv_ratio > 90:
                risk_factors.append("Very high leverage (LTV > 90%)")
                risk_score += 25

            # Cash flow risk
            cash_flow_analysis = self.analyze_cash_flow(params)
            if cash_flow_analysis.monthly_net_cash_flow < 0:
                risk_factors.append("Negative cash flow")
                risk_score += 30
            elif cash_flow_analysis.monthly_net_cash_flow < 200:
                risk_factors.append("Minimal cash flow buffer")
                risk_score += 15

            # Vacancy risk
            if params.vacancy_rate > 10:
                risk_factors.append("High vacancy assumption (>10%)")
                risk_score += 10

            # Market risk (if market data available)
            market_volatility = None
            if market_data:
                market_volatility = self.market_engine.calculate_market_volatility(market_data)
                if market_volatility > 0.15:
                    risk_factors.append("High market volatility")
                    risk_score += 20

            # Calculate individual risk scores (0-1 scale)
            market_risk_score = min(risk_score / 100.0, 1.0)
            location_risk_score = min((risk_score + 10) / 100.0, 1.0)  # Default location risk
            property_risk_score = min(risk_score / 100.0, 1.0)
            financial_risk_score = min(ltv_ratio / 100.0, 1.0)
            liquidity_risk_score = 0.5  # Default moderate liquidity risk

            # Calculate overall risk score as weighted average
            overall_risk_score = (
                market_risk_score * 0.25 + location_risk_score * 0.20 + property_risk_score * 0.20 + financial_risk_score * 0.25 + liquidity_risk_score * 0.10
            )

            # Determine risk level based on overall score
            if overall_risk_score <= 0.3:
                risk_level = "low"
            elif overall_risk_score <= 0.6:
                risk_level = "moderate"
            elif overall_risk_score <= 0.8:
                risk_level = "high"
            else:
                risk_level = "very_high"

            return RiskAssessment(
                investment_params=params,
                property_address=location or "Unknown Location",
                market_context={
                    "market_score": market_risk_score,
                    "comps_count": len(market_data) if market_data else 0,
                },
                market_risk_score=market_risk_score,
                location_risk_score=location_risk_score,
                property_risk_score=property_risk_score,
                financial_risk_score=financial_risk_score,
                liquidity_risk_score=liquidity_risk_score,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                risk_factors=risk_factors,
                mitigation_strategies=["Diversify portfolio", "Regular maintenance", "Market monitoring"],
                sensitivity_analysis={"rent_sensitivity": 0.1, "vacancy_sensitivity": 0.05},
            )

        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            raise

    def analyze_market_trends(self, market_data: List[MarketDataPoint], timeframe_days: int = 90) -> MarketTrend:
        """Analyze market trends from historical data."""
        return self.market_engine.analyze_market_trends(market_data, timeframe_days)

    def forecast_property_value(
        self, current_value: float, market_data: List[MarketDataPoint], forecast_months: int = 12
    ) -> dict:
        """Forecast future property value based on market trends."""
        return self.market_engine.forecast_property_value(current_value, market_data, forecast_months)

    def _calculate_multi_year_projections(
        self,
        params: InvestmentParams,
        initial_noi: Decimal,
        initial_cash_flow: Decimal,
        purchase_price: Decimal,
        loan_amount: Decimal,
    ) -> List[Dict[str, Any]]:
        """Calculate multi-year financial projections."""
        projections = []
        current_noi = initial_noi
        current_property_value = purchase_price
        remaining_loan = loan_amount

        annual_mortgage = (
            self.calculator.calculate_mortgage_payment(loan_amount, params.loan_interest_rate, params.loan_term_years) * 12
        )

        for year in range(1, 11):  # 10-year projection
            # Apply appreciation and rent growth - fix Decimal/float mixing
            appreciation_factor = Decimal("1") + Decimal(str(params.property_appreciation_rate))
            rent_growth_factor = Decimal("1") + Decimal(str(params.annual_rent_increase))

            current_property_value *= appreciation_factor
            current_noi *= rent_growth_factor

            # Calculate principal paydown
            interest_payment = remaining_loan * Decimal(str(params.loan_interest_rate))
            principal_payment = annual_mortgage - interest_payment
            remaining_loan = max(Decimal("0"), remaining_loan - principal_payment)

            # Calculate cash flow and equity
            annual_cash_flow = current_noi - annual_mortgage
            current_equity = current_property_value - remaining_loan

            # Total return (cash flow + equity gain)
            equity_gain = current_property_value - purchase_price
            total_return = (Decimal(str(year)) * annual_cash_flow) + equity_gain

            projections.append(
                {
                    "year": year,
                    "property_value": round(float(current_property_value), 2),
                    "noi": round(float(current_noi), 2),
                    "cash_flow": round(float(annual_cash_flow), 2),
                    "equity": round(float(current_equity), 2),
                    "total_return": round(float(total_return), 2),
                    "remaining_loan": round(float(remaining_loan), 2),
                }
            )

        return projections
