"""
API routes for analytics endpoints.
"""

import logging
from typing import List, Optional

from asyncpg import Connection
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text

from ...analytics.cma_engine import ComparativeMarketAnalysis
from ...analytics.search import search_analytics
from ...models.property import PropertyListing
from ..dependencies import get_cma_engine, get_db_connection

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/cma/{property_id}", summary="Generate Comparative Market Analysis")
async def generate_cma_endpoint(
    property_id: str,
    radius: float = Query(5.0, description="Radius in miles for finding comparables"),
    db: Connection = Depends(get_db_connection),
    cma_engine: ComparativeMarketAnalysis = Depends(get_cma_engine),
):
    """
    Generate a Comparative Market Analysis (CMA) for a given property.
    """
    # Note: This is a simplified example. A real implementation would need a more robust
    # way to fetch and interact with data using asyncpg. The `db.get` and `db.execute`
    # methods used here are illustrative and assume an SQLAlchemy-like async interface
    # that doesn't exist by default with asyncpg.

    # Placeholder for fetching the subject property
    subject_property_record = await db.fetchrow("SELECT * FROM property_listings WHERE id = $1", property_id)
    if not subject_property_record:
        raise HTTPException(status_code=404, detail="Property not found")
    subject_property = PropertyListing(**dict(subject_property_record))

    # Placeholder for fetching comparable properties
    query = "SELECT * FROM property_listings WHERE id != $1 LIMIT 5"
    comparable_properties_records = await db.fetch(query, property_id)

    if not comparable_properties_records:
        raise HTTPException(status_code=404, detail="No comparable properties found within the specified radius")

    comparable_properties = [PropertyListing(**dict(rec)) for rec in comparable_properties_records]

    try:
        cma_report = await cma_engine.generate_cma(
            subject_property=subject_property,
            comparable_properties=comparable_properties,
        )
        return cma_report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating CMA report: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/search-report", summary="Get aggregated search analytics")
async def get_search_report():
    """Return aggregated search analytics metrics."""
    return await search_analytics.generate_performance_report()
