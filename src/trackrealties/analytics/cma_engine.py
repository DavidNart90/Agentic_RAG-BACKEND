"""
Comparative Market Analysis (CMA) Engine for the TrackRealties AI Platform.

This module provides the ComparativeMarketAnalysis class, which is responsible
for generating comprehensive CMA reports for property valuation.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models.market import MarketDataPoint
from ..models.property import PropertyListing


class ComparativeMarketAnalysis:
    """Comparative Market Analysis (CMA) engine for property valuation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_cma(
        self,
        subject_property: PropertyListing,
        comparable_properties: List[PropertyListing],
        market_data: Optional[List[MarketDataPoint]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive CMA report."""
        try:
            if not comparable_properties:
                raise ValueError("No comparable properties provided for CMA generation.")

            # Calculate adjustments for each comparable
            adjusted_comps = []
            for comp in comparable_properties:
                adjustments = self._calculate_adjustments(subject_property, comp)
                adjusted_price = comp.price + sum(adjustments.values())

                adjusted_comps.append(
                    {
                        "property": comp,
                        "original_price": comp.price,
                        "adjustments": adjustments,
                        "adjusted_price": adjusted_price,
                        "price_per_sqft": (
                            adjusted_price / comp.squareFootage if comp.squareFootage and comp.squareFootage > 0 else 0
                        ),
                    }
                )

            # Calculate estimated value range
            adjusted_prices = [comp["adjusted_price"] for comp in adjusted_comps]
            estimated_value = sum(adjusted_prices) / len(adjusted_prices)
            value_range = {"low": min(adjusted_prices), "high": max(adjusted_prices), "average": estimated_value}

            # Market position analysis
            market_position = self._analyze_market_position(subject_property, adjusted_comps, market_data)

            # Suggested listing price
            suggested_price = self.suggest_listing_price(estimated_value, market_position)

            return {
                "subject_property": subject_property,
                "comparable_properties": adjusted_comps,
                "estimated_value": round(estimated_value, 2),
                "value_range": value_range,
                "market_position": market_position,
                "suggested_listing_price": suggested_price,
                "confidence_score": self._calculate_confidence_score(adjusted_comps),
                "analysis_date": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"CMA generation failed: {e}")
            raise

    def _calculate_adjustments(self, subject: PropertyListing, comparable: PropertyListing) -> Dict[str, float]:
        """Calculate price adjustments between subject and comparable properties."""
        adjustments = {}

        # Square footage adjustment
        if subject.squareFootage and comparable.squareFootage and comparable.price and comparable.squareFootage > 0:
            sqft_diff = subject.squareFootage - comparable.squareFootage
            price_per_sqft = comparable.price / comparable.squareFootage
            adjustments["squareFootage"] = sqft_diff * price_per_sqft * 0.8  # 80% adjustment factor

        # Bedroom adjustment
        if subject.bedrooms and comparable.bedrooms:
            bedroom_diff = subject.bedrooms - comparable.bedrooms
            adjustments["bedrooms"] = bedroom_diff * 15000  # $15k per bedroom

        # Bathroom adjustment
        if subject.bathrooms and comparable.bathrooms:
            bathroom_diff = subject.bathrooms - comparable.bathrooms
            adjustments["bathrooms"] = bathroom_diff * 10000  # $10k per bathroom

        # Age adjustment
        if subject.yearBuilt and comparable.yearBuilt:
            age_diff = comparable.yearBuilt - subject.yearBuilt  # Newer is better
            adjustments["age"] = age_diff * 1000  # $1k per year

        # Location adjustment (simplified)
        if hasattr(subject, "zipCode") and hasattr(comparable, "zipCode") and subject.zipCode != comparable.zipCode:
            adjustments["location"] = 0  # Would need market data for proper adjustment

        return adjustments

    def _analyze_market_position(
        self, subject: PropertyListing, comps: List[Dict], market_data: Optional[List[MarketDataPoint]]
    ) -> Dict[str, Any]:
        """Analyze subject property's position in the market."""
        if not comps:
            return {}

        avg_price_per_sqft = (
            sum(comp["price_per_sqft"] for comp in comps if comp["price_per_sqft"] > 0) / len(comps)
            if len(comps) > 0
            else 0
        )

        position = {
            "relative_to_comps": "average",
            "pricing_recommendation": "market_value",
            "days_on_market_estimate": 45,  # Default estimate
        }

        # Determine relative position
        subject_price_per_sqft = (
            subject.price / subject.squareFootage if subject.squareFootage and subject.squareFootage > 0 else 0
        )
        if avg_price_per_sqft > 0:
            if subject_price_per_sqft > avg_price_per_sqft * 1.1:
                position["relative_to_comps"] = "above_market"
                position["pricing_recommendation"] = "consider_price_reduction"
            elif subject_price_per_sqft < avg_price_per_sqft * 0.9:
                position["relative_to_comps"] = "below_market"
                position["pricing_recommendation"] = "opportunity_for_increase"

        return position

    def suggest_listing_price(self, estimated_value: float, market_position: Dict[str, Any]) -> float:
        """Suggest a listing price based on CMA and market position."""
        recommendation = market_position.get("pricing_recommendation", "market_value")

        if recommendation == "consider_price_reduction":
            return round(estimated_value * 0.98, -2)  # Suggest 2% lower, rounded to nearest 100
        elif recommendation == "opportunity_for_increase":
            return round(estimated_value * 1.02, -2)  # Suggest 2% higher
        else:
            return round(estimated_value, -2)  # Round to nearest 100

    def _calculate_confidence_score(self, comps: List[Dict]) -> float:
        """Calculate confidence score for the CMA."""
        if not comps or len(comps) < 3:
            return 0.6  # Lower confidence with fewer comps

        # Calculate price variance
        prices = [comp["adjusted_price"] for comp in comps]
        avg_price = sum(prices) / len(prices)
        if avg_price == 0:
            return 0.5

        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        cv = (variance**0.5) / avg_price if avg_price > 0 else 1

        # Higher variance = lower confidence
        confidence = max(0.5, 1 - cv)
        return round(confidence, 2)
