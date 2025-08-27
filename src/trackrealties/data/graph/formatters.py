"""
Formatters for graph content to improve readability in Neo4j.

This module provides utility functions to format content for better
readability in Neo4j graph visualizations and queries.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def format_property_content(property_data: Dict[str, Any]) -> str:
    """
    Format property data for better readability in graph.

    Args:
        property_data: Property data to format

    Returns:
        Formatted property content as a string
    """
    address = property_data.get("formattedAddress", "Unknown Address")
    property_type = property_data.get("propertyType", "Unknown Type")
    price = property_data.get("price")
    price_str = f"${price:,.2f}" if price else "Price not available"

    content = f"{address} - {property_type} - {price_str}\n\n"

    # Add key details
    content += "Key Details:\n"

    if property_data.get("bedrooms") is not None:
        content += f"• {property_data['bedrooms']} bed"
        if property_data.get("bathrooms") is not None:
            content += f" / {property_data['bathrooms']} bath"
        content += "\n"

    if property_data.get("squareFootage") is not None:
        content += f"• {property_data['squareFootage']:,} sq ft"
        if property_data.get("lotSize") is not None:
            content += f" on {property_data['lotSize']:,} sq ft lot"
        content += "\n"

    if property_data.get("yearBuilt") is not None:
        content += f"• Built in {property_data['yearBuilt']}\n"

    if property_data.get("daysOnMarket") is not None:
        content += f"• {property_data['daysOnMarket']} days on market\n"

    # Add listing type and date
    if property_data.get("listingType"):
        content += f"• {property_data['listingType']} listing\n"

    if property_data.get("listedDate"):
        content += f"• Listed on {property_data['listedDate']}\n"

    return content


def format_market_content(market_data: Dict[str, Any]) -> str:
    """
    Format market data for better readability in graph.

    Args:
        market_data: Market data to format

    Returns:
        Formatted market content as a string
    """
    location = market_data.get("location", "Unknown Location")
    date = market_data.get("date", "Unknown Date")
    region_type = market_data.get("region_type", "Unknown Type")

    content = f"{location} Market Report ({region_type})\n"
    content += f"Date: {date}\n\n"

    # Add key metrics
    content += "Key Metrics:\n"

    if market_data.get("median_price") is not None:
        content += f"• Median Price: ${market_data['median_price']:,.2f}\n"

    if market_data.get("inventory_count") is not None:
        content += f"• Inventory Count: {market_data['inventory_count']:,}\n"

    if market_data.get("sales_volume") is not None:
        content += f"• Sales Volume: {market_data['sales_volume']:,}\n"

    if market_data.get("days_on_market") is not None:
        content += f"• Days on Market: {market_data['days_on_market']}\n"

    if market_data.get("months_supply") is not None:
        content += f"• Months Supply: {market_data['months_supply']:.2f}\n"

    if market_data.get("price_per_sqft") is not None:
        content += f"• Price per Sq Ft: ${market_data['price_per_sqft']:.2f}\n"

    # Add source and date information
    if market_data.get("source"):
        content += f"\nSource: {market_data['source']}"

    if market_data.get("last_updated"):
        content += f"\nLast Updated: {market_data['last_updated']}"

    return content


def format_agent_content(agent_data: Dict[str, Any]) -> str:
    """
    Format agent data for better readability in graph.

    Args:
        agent_data: Agent data to format

    Returns:
        Formatted agent content as a string
    """
    name = agent_data.get("name", "Unknown Agent")

    content = f"Agent: {name}\n\n"

    if agent_data.get("phone"):
        content += f"Phone: {agent_data['phone']}\n"

    if agent_data.get("email"):
        content += f"Email: {agent_data['email']}\n"

    if agent_data.get("website"):
        content += f"Website: {agent_data['website']}\n"

    return content


def format_location_content(location_data: Dict[str, Any]) -> str:
    """
    Format location data for better readability in graph.

    Args:
        location_data: Location data to format

    Returns:
        Formatted location content as a string
    """
    city = location_data.get("city", "Unknown City")
    state = location_data.get("state", "Unknown State")

    content = f"{city}, {state}\n\n"

    if location_data.get("zipCode"):
        content += f"ZIP: {location_data['zipCode']}\n"

    if location_data.get("county"):
        content += f"County: {location_data['county']}\n"

    if location_data.get("latitude") is not None and location_data.get("longitude") is not None:
        content += f"Coordinates: {location_data['latitude']:.6f}, {location_data['longitude']:.6f}\n"

    return content


def format_relationship_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format relationship properties for better readability in graph.

    Args:
        properties: Relationship properties to format

    Returns:
        Formatted relationship properties
    """
    formatted_props = {}

    for key, value in properties.items():
        # Format dates
        if isinstance(value, datetime):
            formatted_props[key] = value.isoformat()
        # Format complex objects
        elif not isinstance(value, (str, int, float, bool, type(None))):
            formatted_props[key] = json.dumps(value)
        # Keep simple values as is
        else:
            formatted_props[key] = value

    return formatted_props
