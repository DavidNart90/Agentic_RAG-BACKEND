"""
Market Intelligence Module

This module provides market trend analysis and intelligence capabilities
for real estate market data analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..models.market import MarketDataPoint

logger = logging.getLogger(__name__)


@dataclass
class MarketTrend:
    """Container for market trend analysis."""

    trend_direction: str  # 'up', 'down', 'stable'
    trend_strength: float  # 0-1 scale
    price_change_percent: float
    volume_change_percent: float
    forecast_confidence: float


class MarketIntelligenceEngine:
    """Market intelligence and trend analysis engine."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_market_trends(self, market_data: List[MarketDataPoint], timeframe_days: int = 90) -> MarketTrend:
        """Analyze market trends from historical data."""
        try:
            if len(market_data) < 2:
                return MarketTrend(
                    trend_direction="stable",
                    trend_strength=0.0,
                    price_change_percent=0.0,
                    volume_change_percent=0.0,
                    forecast_confidence=0.0,
                )

            # Filter recent data
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
            recent_data = [d for d in market_data if hasattr(d, "date") and d.date >= cutoff_date]

            if len(recent_data) < 2:
                recent_data = (
                    sorted(market_data, key=lambda x: x.date)[-10:]
                    if hasattr(market_data[0], "date")
                    else market_data[-10:]
                )

            # Calculate price trend - convert Decimal to float for mathematical operations
            prices = [
                float(d.median_price) for d in recent_data if hasattr(d, "median_price") and d.median_price is not None
            ]
            price_change = 0.0
            if len(prices) >= 2:
                price_change = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] != 0 else 0

            # Calculate volume trend
            volumes = [
                float(d.sales_volume) for d in recent_data if hasattr(d, "sales_volume") and d.sales_volume is not None
            ]
            volume_change = 0.0
            if len(volumes) >= 2:
                volume_change = (volumes[-1] - volumes[0]) / volumes[0] * 100 if volumes[0] != 0 else 0

            # Determine trend direction and strength
            if abs(price_change) < 2:
                trend_direction = "stable"
                trend_strength = 0.3
            elif price_change > 0:
                trend_direction = "up"
                trend_strength = min(abs(price_change) / 10, 1.0)
            else:
                trend_direction = "down"
                trend_strength = min(abs(price_change) / 10, 1.0)

            # Calculate forecast confidence
            data_points = len(recent_data)
            confidence = min(data_points / 20, 1.0) * 0.8  # Max 80% confidence

            return MarketTrend(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                price_change_percent=price_change,
                volume_change_percent=volume_change,
                forecast_confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Market trend analysis failed: {e}")
            raise

    def calculate_market_volatility(self, market_data: List[MarketDataPoint]) -> float:
        """Calculate market volatility with enhanced real estate market logic."""
        if len(market_data) < 2:
            # Use historical real estate market volatility estimates
            return self._estimate_real_estate_volatility()

        # Convert Decimal prices to float for mathematical operations
        prices = [
            float(d.median_price) for d in market_data if hasattr(d, "median_price") and d.median_price is not None
        ]
        if len(prices) < 2:
            return self._estimate_real_estate_volatility()

        # Check if all prices are identical (common with synthetic data)
        if len(set(prices)) == 1:
            self.logger.warning("All market data prices are identical - using estimated volatility")
            return self._estimate_real_estate_volatility()

        returns = [(prices[i] / prices[i - 1] - 1) for i in range(1, len(prices)) if prices[i - 1] != 0]

        if not returns:
            return self._estimate_real_estate_volatility()

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        calculated_volatility = variance**0.5

        # Only apply constraints if we have insufficient data (less than 5 data points)
        # With enough data, trust the calculated volatility even if it's high
        if len(market_data) < 5:
            self.logger.info(
                f"Insufficient data ({len(market_data)} points) - applying real estate constraints to volatility"
            )
            return self._apply_real_estate_constraints(calculated_volatility, prices)
        else:
            self.logger.info(
                f"Sufficient data ({len(market_data)} points) - using calculated volatility without capping"
            )
            # Only apply minimum volatility constraint when we have enough data
            min_real_estate_volatility = 0.005  # 0.5% monthly minimum
            if calculated_volatility < min_real_estate_volatility:
                calculated_volatility = min_real_estate_volatility
            return calculated_volatility

    def _estimate_real_estate_volatility(self) -> float:
        """Estimate realistic real estate market volatility based on market conditions."""
        # Real estate markets typically have lower volatility than stock markets
        # Based on historical data, residential real estate has annual volatility of 2-15%
        # Monthly volatility would be approximately annual_volatility / sqrt(12)

        # Base volatility estimates for different market conditions
        base_volatilities = {
            "stable_market": 0.02,  # 2% annual → ~0.6% monthly
            "moderate_market": 0.05,  # 5% annual → ~1.4% monthly
            "volatile_market": 0.10,  # 10% annual → ~2.9% monthly
            "crisis_market": 0.15,  # 15% annual → ~4.3% monthly
        }

        # For now, use moderate market as default
        # In a real system, this could be determined by economic indicators
        estimated_annual_volatility = base_volatilities["moderate_market"]

        # Convert to monthly volatility (assuming monthly data points)
        monthly_volatility = estimated_annual_volatility / (12**0.5)

        self.logger.info(f"Using estimated real estate volatility: {monthly_volatility:.4f}")
        return monthly_volatility

    def _apply_real_estate_constraints(self, calculated_volatility: float, prices: List[float]) -> float:
        """Apply real estate market constraints to calculated volatility."""
        # Real estate markets have different characteristics than financial markets

        # 1. Cap maximum volatility (real estate is less volatile than stocks)
        max_real_estate_volatility = 0.20  # 20% monthly is extremely high for real estate
        if calculated_volatility > max_real_estate_volatility:
            self.logger.warning(
                f"Calculated volatility {calculated_volatility:.4f} exceeds real estate maximum, capping at {max_real_estate_volatility}"
            )
            calculated_volatility = max_real_estate_volatility

        # 2. Set minimum volatility (markets are never completely stable)
        min_real_estate_volatility = 0.005  # 0.5% monthly minimum
        if calculated_volatility < min_real_estate_volatility:
            calculated_volatility = min_real_estate_volatility

        # 3. Adjust based on price level (higher priced markets tend to be more stable)
        avg_price = sum(prices) / len(prices)
        if avg_price > 1000000:  # Luxury market
            calculated_volatility *= 0.8  # 20% reduction for luxury markets
        elif avg_price < 200000:  # Lower-priced market
            calculated_volatility *= 1.2  # 20% increase for lower-priced markets

        return calculated_volatility

    def forecast_property_value(
        self, current_value: float, market_data: List[MarketDataPoint], forecast_months: int = 12
    ) -> dict:
        """Forecast future property value based on market trends."""
        try:
            if not market_data or len(market_data) < 3:
                return {"forecasted_value": current_value, "confidence": 0.0, "trend": "insufficient_data"}

            # Calculate historical appreciation rate
            sorted_data = sorted(market_data, key=lambda x: x.date) if hasattr(market_data[0], "date") else market_data
            prices = [
                float(d.median_price) for d in sorted_data if hasattr(d, "median_price") and d.median_price is not None
            ]
            if len(prices) < 2:
                return {"forecasted_value": current_value, "confidence": 0.0, "trend": "insufficient_data"}

            # Calculate monthly appreciation rate
            first_date = (
                sorted_data[0].date
                if hasattr(sorted_data[0], "date")
                else datetime.now() - timedelta(days=len(sorted_data))
            )
            last_date = sorted_data[-1].date if hasattr(sorted_data[-1], "date") else datetime.now()
            months_diff = max(1, (last_date.year - first_date.year) * 12 + last_date.month - first_date.month)

            total_appreciation = (prices[-1] / prices[0]) - 1 if prices[0] != 0 else 0
            monthly_appreciation_rate = ((1 + total_appreciation) ** (1 / months_diff)) - 1

            # Apply forecast
            forecasted_value = current_value * ((1 + monthly_appreciation_rate) ** forecast_months)

            # Calculate confidence based on data quality
            confidence = min(len(market_data) / 24, 1.0) * 0.9  # Max 90% confidence

            # Determine trend
            if monthly_appreciation_rate > 0.002:  # > 0.2% monthly
                trend = "appreciating"
            elif monthly_appreciation_rate < -0.002:  # < -0.2% monthly
                trend = "depreciating"
            else:
                trend = "stable"

            return {
                "forecasted_value": round(forecasted_value, 2),
                "monthly_rate": round(monthly_appreciation_rate * 100, 3),
                "annual_rate": round(((1 + monthly_appreciation_rate) ** 12 - 1) * 100, 2),
                "confidence": round(confidence, 2),
                "trend": trend,
            }

        except Exception as e:
            self.logger.error(f"Property value forecast failed: {e}")
            return {"forecasted_value": current_value, "confidence": 0.0, "trend": "error"}

    def generate_market_summary(self, market_data: List[MarketDataPoint]) -> Dict[str, Any]:
        """Generate a high-level summary of market conditions."""
        if not market_data:
            return {"summary": "No market data available to generate a summary."}

        latest_data = (
            sorted(market_data, key=lambda x: x.date, reverse=True)[0]
            if hasattr(market_data[0], "date")
            else market_data[-1]
        )
        trends = self.analyze_market_trends(market_data)

        summary = {
            "location": latest_data.location if hasattr(latest_data, "location") else "N/A",
            "latest_date": latest_data.date.isoformat() if hasattr(latest_data, "date") else "N/A",
            "median_price": latest_data.median_price if hasattr(latest_data, "median_price") else "N/A",
            "inventory_count": latest_data.inventory_count if hasattr(latest_data, "inventory_count") else "N/A",
            "days_on_market": latest_data.days_on_market if hasattr(latest_data, "days_on_market") else "N/A",
            "trend_direction": trends.trend_direction,
            "price_change_percent": round(trends.price_change_percent, 2),
            "forecast_confidence": round(trends.forecast_confidence, 2),
        }

        return summary
