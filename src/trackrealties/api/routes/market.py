"""
Market data API routes for TrackRealties AI Platform.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from asyncpg import Connection
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ...core.database import db_pool
from ...models.market import (MarketDataPoint, MarketDataResponse,
                              MarketSearchCriteria, MarketSearchResponse)
from ..dependencies import get_db_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{region_id}", response_model=MarketDataResponse)
async def get_market_data(
    region_id: str,
    db: Connection = Depends(get_db_connection),
    date: Optional[str] = Query(None, description="Date for market data (YYYY-MM-DD)"),
):
    """Get market data for a specific region."""
    try:
        query = select(MarketDataPoint).where(MarketDataPoint.region_id == region_id)
        if date:
            target_date = datetime.fromisoformat(date)
            query = query.where(MarketDataPoint.period_start <= target_date, MarketDataPoint.period_end >= target_date)

        market_data = (await db.execute(query.order_by(MarketDataPoint.period_end.desc()))).scalars().first()

        if not market_data:
            raise HTTPException(status_code=404, detail="Market data not found for the specified region and date")

        return MarketDataResponse.model_validate(market_data, from_attributes=True)

    except Exception as e:
        logger.error(f"Failed to get market data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve market data")


@router.post("/search", response_model=MarketSearchResponse)
async def search_market_data(search_request: MarketSearchCriteria, db: Connection = Depends(get_db_connection)):
    """Search for market data based on criteria."""
    try:
        where_clause, params = search_request.to_sql_filters()

        # Build the query
        query = select(MarketDataPoint).where(text(where_clause))

        # Count total results
        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query, params)).scalar_one()

        # Get paginated results
        results_query = query.limit(search_request.limit).offset(search_request.offset)
        results = (await db.execute(results_query, params)).scalars().all()

        response_results = [MarketDataResponse.model_validate(p, from_attributes=True) for p in results]

        return MarketSearchResponse(
            results=response_results,
            total=total,
            limit=search_request.limit,
            offset=search_request.offset,
            filters_applied=search_request.model_dump(exclude_none=True),
        )

    except Exception as e:
        logger.error(f"Market data search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to search market data")


@router.get("/{region_id}/trends")
async def get_market_trends(
    region_id: str,
    db: Connection = Depends(get_db_connection),
    period: str = Query("1y", description="Time period (1m, 3m, 6m, 1y, 5y)"),
):
    """Get market trends for a specific region."""
    try:
        # Define time period
        end_date = datetime.now(timezone.utc)
        if period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "5y":
            start_date = end_date - timedelta(days=365 * 5)
        else:
            raise HTTPException(status_code=400, detail="Invalid period specified")

        # Query for trend data
        query = (
            select(MarketDataPoint)
            .where(MarketDataPoint.region_id == region_id, MarketDataPoint.period_end.between(start_date, end_date))
            .order_by(MarketDataPoint.period_end.asc())
        )

        trend_data = (await db.execute(query)).scalars().all()

        if not trend_data:
            raise HTTPException(status_code=404, detail="No trend data found for the specified period")

        # Calculate trend summary
        first_point = trend_data[0]
        last_point = trend_data[-1]

        price_change = (
            (last_point.median_price - first_point.median_price) / first_point.median_price * 100
            if first_point.median_price
            else 0
        )
        inventory_change = (
            (last_point.inventory_count - first_point.inventory_count) / first_point.inventory_count * 100
            if first_point.inventory_count
            else 0
        )
        dom_change = (
            (last_point.days_on_market - first_point.days_on_market) / first_point.days_on_market * 100
            if first_point.days_on_market
            else 0
        )

        return {
            "region_id": region_id,
            "period": period,
            "data_points": [MarketDataResponse.model_validate(p, from_attributes=True) for p in trend_data],
            "trends": {
                "median_price": {"change": round(price_change, 2), "direction": "up" if price_change > 0 else "down"},
                "inventory": {
                    "change": round(inventory_change, 2),
                    "direction": "up" if inventory_change > 0 else "down",
                },
                "days_on_market": {"change": round(dom_change, 2), "direction": "up" if dom_change > 0 else "down"},
            },
        }

    except Exception as e:
        logger.error(f"Failed to get market trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve market trends")
