"""
Field mapping utilities for data normalization.

This module provides functions to normalize field names from external data sources
to match the internal field naming conventions used throughout the platform.
"""

from typing import Any, Dict, List, Optional


def normalize_property_data(property_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize property data field names to internal standards.

    This function maps external field names (like "id") to internal field names
    (like "property_id") to ensure consistency across the platform.

    Args:
        property_data: Property data dictionary with external field names

    Returns:
        Normalized property data dictionary with internal field names
    """
    if not property_data:
        return {}

    normalized_data = property_data.copy()

    # Map ID field
    if "id" in normalized_data and "property_id" not in normalized_data:
        normalized_data["property_id"] = normalized_data["id"]

    # Map other common fields
    field_mappings = {
        "propertyType": "property_type",
        "squareFootage": "square_footage",
        "zipCode": "zip_code",
        "yearBuilt": "year_built",
        "daysOnMarket": "days_on_market",
        "formattedAddress": "formatted_address",
        "addressLine1": "address_line1",
        "addressLine2": "address_line2",
        "listingAgent": "listing_agent",
        "listingOffice": "listing_office",
    }

    for external_field, internal_field in field_mappings.items():
        if external_field in normalized_data and internal_field not in normalized_data:
            normalized_data[internal_field] = normalized_data[external_field]

    return normalized_data


def normalize_market_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize market data field names to internal standards.

    Args:
        market_data: Market data dictionary with external field names

    Returns:
        Normalized market data dictionary with internal field names
    """
    if not market_data:
        return {}

    normalized_data = market_data.copy()

    # Map ID field
    if "id" in normalized_data and "region_id" not in normalized_data:
        normalized_data["region_id"] = normalized_data["id"]

    # Map other common fields
    field_mappings = {
        "regionName": "region_name",
        "regionType": "region_type",
        "periodStart": "period_start",
        "periodEnd": "period_end",
        "dataQualityScore": "data_quality_score",
        "sampleSize": "sample_size",
    }

    for external_field, internal_field in field_mappings.items():
        if external_field in normalized_data and internal_field not in normalized_data:
            normalized_data[internal_field] = normalized_data[external_field]

    return normalized_data


def normalize_batch_data(data_items: List[Dict[str, Any]], data_type: str = "property") -> List[Dict[str, Any]]:
    """
    Normalize a batch of data items.

    Args:
        data_items: List of data dictionaries to normalize
        data_type: Type of data ("property" or "market")

    Returns:
        List of normalized data dictionaries
    """
    if not data_items:
        return []

    if data_type.lower() == "property":
        return [normalize_property_data(item) for item in data_items]
    elif data_type.lower() == "market":
        return [normalize_market_data(item) for item in data_items]
    else:
        # For unknown data types, return as is
        return data_items
