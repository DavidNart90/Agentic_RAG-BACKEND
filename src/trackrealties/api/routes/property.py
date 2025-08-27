"""
Property listing API routes for TrackRealties AI Platform.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from asyncpg import Connection
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...core.database import db_pool
from ...models.property import (PropertyListing, PropertyListingRequest,
                                PropertyListingResponse,
                                PropertySearchCriteria, PropertySearchResponse)
from ..dependencies import get_db_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{property_id}", response_model=PropertyListingResponse)
async def get_property(
    property_id: str,
    db: Connection = Depends(get_db_connection),
    include_history: bool = Query(False, description="Include property history"),
):
    """Get property listing by ID."""
    try:
        result = await db.execute(select(PropertyListing).where(PropertyListing.id == property_id))
        property_listing = result.scalar_one_or_none()
        if not property_listing:
            raise HTTPException(status_code=404, detail="Property not found")

        response = PropertyListingResponse.model_validate(property_listing, from_attributes=True)
        response.url = f"https://example.com/property/{property_listing.id}"
        return response

    except Exception as e:
        logger.error(f"Failed to get property: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve property")


@router.post("/search", response_model=PropertySearchResponse)
async def search_properties(search_request: PropertySearchCriteria, db: Connection = Depends(get_db_connection)):
    """Search for properties based on criteria."""
    try:
        query = select(PropertyListing)

        # Apply filters from search criteria
        if search_request.city:
            query = query.where(PropertyListing.city.ilike(f"%{search_request.city}%"))
        if search_request.state:
            query = query.where(PropertyListing.state == search_request.state)
        if search_request.zipCode:
            query = query.where(PropertyListing.zipCode == search_request.zipCode)
        if search_request.min_price:
            query = query.where(PropertyListing.price >= search_request.min_price)
        if search_request.max_price:
            query = query.where(PropertyListing.price <= search_request.max_price)
        if search_request.min_bedrooms:
            query = query.where(PropertyListing.bedrooms >= search_request.min_bedrooms)
        if search_request.max_bedrooms:
            query = query.where(PropertyListing.bedrooms <= search_request.max_bedrooms)
        if search_request.propertyTypes:
            query = query.where(PropertyListing.propertyType.in_(search_request.propertyTypes))
        if search_request.statuses:
            query = query.where(PropertyListing.status.in_(search_request.statuses))

        # Count total results
        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar_one()

        # Get paginated results
        results_query = query.limit(search_request.limit).offset(search_request.offset)
        results = (await db.execute(results_query)).scalars().all()

        response_results = [PropertyListingResponse.model_validate(p, from_attributes=True) for p in results]
        for r in response_results:
            r.url = f"https://example.com/property/{r.id}"

        return PropertySearchResponse(
            results=response_results,
            total=total,
            limit=search_request.limit,
            offset=search_request.offset,
            filters_applied=search_request.model_dump(exclude_none=True),
        )

    except Exception as e:
        logger.error(f"Property search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to search properties")


@router.post("/", status_code=201)
async def create_or_update_property(
    property_request: PropertyListingRequest, db: Connection = Depends(get_db_connection)
):
    """Create or update a property listing."""
    try:
        if not property_request.property_id:
            raise HTTPException(status_code=400, detail="property_id is required")

        result = await db.execute(select(PropertyListing).where(PropertyListing.id == property_request.property_id))
        property_listing = result.scalar_one_or_none()

        if property_listing:
            # Update
            update_data = property_request.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(property_listing, key, value)
            message = "Property listing updated successfully"
        else:
            # Create
            property_listing = PropertyListing(**property_request.model_dump(exclude_unset=True))
            db.add(property_listing)
            message = "Property listing created successfully"

        await db.commit()

        return {"property_id": property_request.property_id, "message": message, "status": "success"}

    except Exception as e:
        logger.error(f"Failed to create/update property: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create/update property")


@router.post("/{property_id}/analyze")
async def analyze_property(
    property_id: str,
    db: Connection = Depends(get_db_connection),
    analysis_type: str = Query("investment", description="Type of analysis to perform"),
):
    """Analyze a property for investment, development, or purchase."""
    try:
        result = await db.execute(select(PropertyListing).where(PropertyListing.id == property_id))
        property_listing = result.scalar_one_or_none()
        if not property_listing:
            raise HTTPException(status_code=404, detail="Property not found")

        # Analysis logic would go here.
        # For now, returning mock analysis based on the fetched property.

        return {
            "property_id": property_id,
            "analysis_type": analysis_type,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "results": {
                "estimated_value": float(property_listing.price) * 1.05,
                "estimated_rent": float(property_listing.price) / 200,
                "cap_rate": 0.06,
                "cash_on_cash_return": 0.08,
                "roi_5yr": 0.35,
            },
            "recommendations": [
                "Property shows good potential for long-term investment",
                "Consider negotiating purchase price to improve cap rate",
                "Rental demand is strong in this area",
            ],
        }

    except Exception as e:
        logger.error(f"Property analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze property")
