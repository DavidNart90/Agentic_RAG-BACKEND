# src/trackrealties/analytics/investor.py

"""
This module provides analytics tools for the Investor Agent, including ROI calculations.
"""


class InvestorAnalyticsTools:
    """
    A collection of static methods for performing investment-related analytics.
    """

    @staticmethod
    def calculate_roi(
        purchase_price: float, rental_income: float, operating_expenses: float, property_value_appreciation: float
    ) -> dict:
        """
        Calculates the annual Return on Investment (ROI) for a property.

        Args:
            purchase_price (float): The total purchase price of the property.
            rental_income (float): The monthly rental income.
            operating_expenses (float): The monthly operating expenses.
            property_value_appreciation (float): The estimated annual appreciation in property value.

        Returns:
            dict: A dictionary containing the calculated 'roi_percentage' and a brief 'analysis'.
        """
        if purchase_price <= 0:
            return {"roi_percentage": 0, "analysis": "Purchase price must be positive to calculate ROI."}

        annual_net_income = (rental_income * 12) - (operating_expenses * 12)
        total_return = annual_net_income + property_value_appreciation
        roi_percentage = (total_return / purchase_price) * 100

        analysis = "This represents a strong return on investment."
        if roi_percentage < 5:
            analysis = "This investment shows a modest return."
        elif roi_percentage < 0:
            analysis = "This investment may result in a loss."

        return {"roi_percentage": round(roi_percentage, 2), "analysis": analysis}

    @staticmethod
    def project_cash_flow(rental_income: float, operating_expenses: float, years: int = 5) -> dict:
        """
        Projects the cash flow for a property over a specified number of years.

        Args:
            rental_income (float): The monthly rental income.
            operating_expenses (float): The monthly operating expenses.
            years (int): The number of years to project (default is 5).

        Returns:
            dict: A dictionary containing 'annual_cash_flow', 'total_cash_flow', and a brief 'analysis'.
        """
        if rental_income < 0 or operating_expenses < 0:
            return {
                "annual_cash_flow": 0,
                "total_cash_flow": 0,
                "analysis": "Income and expenses must be non-negative.",
            }

        annual_cash_flow = (rental_income * 12) - (operating_expenses * 12)
        total_cash_flow = annual_cash_flow * years

        analysis = f"Projected cash flow over {years} years is positive."
        if annual_cash_flow < 0:
            analysis = f"The property is projected to have a negative cash flow over {years} years."

        return {
            "annual_cash_flow": round(annual_cash_flow, 2),
            "total_cash_flow": round(total_cash_flow, 2),
            "analysis": analysis,
        }
