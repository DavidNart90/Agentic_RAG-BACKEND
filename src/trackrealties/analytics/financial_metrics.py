"""
Financial Metrics Calculator Module

This module provides core financial calculation utilities for real estate investments.
Keeps calculations separate from the main engine for better modularity.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Union

# Set decimal precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """Container for financial analysis results."""

    roi: float
    irr: float
    npv: float
    cash_on_cash_return: float
    cap_rate: float
    debt_service_coverage_ratio: float
    break_even_occupancy: float
    payback_period: int


class FinancialCalculator:
    """Core financial calculations for real estate investments."""

    @staticmethod
    def calculate_mortgage_payment(
        loan_amount: Union[float, Decimal], annual_rate: Union[float, Decimal], term_years: int
    ) -> Decimal:
        """Calculate monthly mortgage payment."""
        # Convert inputs to Decimal for precision
        loan_amount = Decimal(str(loan_amount))
        annual_rate = Decimal(str(annual_rate))

        if loan_amount <= 0 or term_years <= 0:
            return Decimal("0")

        monthly_rate = annual_rate / Decimal("100") / Decimal("12")
        num_payments = Decimal(str(term_years * 12))

        if monthly_rate == 0:
            return loan_amount / num_payments

        payment = (
            loan_amount * (monthly_rate * (Decimal("1") + monthly_rate) ** num_payments) / ((Decimal("1") + monthly_rate) ** num_payments - Decimal("1"))
        )
        return payment

    @staticmethod
    def calculate_irr(cash_flows: List[Union[float, Decimal]], max_iterations: int = 100) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        try:
            # Convert to float for IRR calculation (complex math operations)
            float_cash_flows = [float(cf) for cf in cash_flows]

            # Initial guess
            rate = 0.1

            for _ in range(max_iterations):
                npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(float_cash_flows))
                npv_derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(float_cash_flows))

                if abs(npv_derivative) < 1e-10:
                    break

                new_rate = rate - npv / npv_derivative

                if abs(new_rate - rate) < 1e-10:
                    break

                rate = new_rate

            return rate

        except (ZeroDivisionError, OverflowError):
            return 0.0

    @staticmethod
    def calculate_npv(
        initial_investment: Union[float, Decimal],
        cash_flows: List[Union[float, Decimal]],
        discount_rate: Union[float, Decimal],
    ) -> Decimal:
        """Calculate Net Present Value."""
        # Convert to Decimal for precision
        initial_investment = Decimal(str(initial_investment))
        discount_rate = Decimal(str(discount_rate))

        npv = -initial_investment
        for i, cf in enumerate(cash_flows):
            cf_decimal = Decimal(str(cf))
            npv += cf_decimal / ((Decimal("1") + discount_rate) ** (i + 1))
        return npv

    @staticmethod
    def calculate_cap_rate(noi: Union[float, Decimal], property_value: Union[float, Decimal]) -> float:
        """Calculate Capitalization Rate."""
        noi = Decimal(str(noi))
        property_value = Decimal(str(property_value))

        if property_value <= 0:
            return 0.0
        return float((noi / property_value) * Decimal("100"))

    @staticmethod
    def calculate_cash_on_cash_return(
        annual_cash_flow: Union[float, Decimal], total_cash_invested: Union[float, Decimal]
    ) -> float:
        """Calculate Cash-on-Cash Return."""
        annual_cash_flow = Decimal(str(annual_cash_flow))
        total_cash_invested = Decimal(str(total_cash_invested))

        if total_cash_invested <= 0:
            return 0.0
        return float((annual_cash_flow / total_cash_invested) * Decimal("100"))

    @staticmethod
    def calculate_dscr(noi: Union[float, Decimal], debt_service: Union[float, Decimal]) -> float:
        """Calculate Debt Service Coverage Ratio."""
        noi = Decimal(str(noi))
        debt_service = Decimal(str(debt_service))

        if debt_service <= 0:
            return 0.0
        return float(noi / debt_service)
