"""
Data transformation utilities for the TrackRealties AI Platform.

This module provides utilities for transforming data between different formats
and models, including field mapping, type conversion, and data normalization.
"""

import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models.market import MarketDataPoint, MarketMetrics
from ..models.property import (ContactInfo, HOAInfo, PropertyEvent,
                               PropertyListing)

logger = logging.getLogger(__name__)


class ModelTransformer:
    """
    Transforms data into Pydantic models.

    This class provides methods for transforming raw data into Pydantic models,
    including validation and error handling.
    """

    async def transform_to_property_listing(self, data: Dict[str, Any]) -> Tuple[Optional[PropertyListing], List[str]]:
        """
        Transform raw property data into a PropertyListing model.

        Args:
            data: Raw property data

        Returns:
            Tuple of (PropertyListing model, list of validation errors)
        """
        errors = []

        try:
            # Process nested objects
            if "listingAgent" in data and data["listingAgent"]:
                try:
                    data["listingAgent"] = ContactInfo(**data["listingAgent"])
                except Exception as e:
                    errors.append(f"Invalid listing agent data: {str(e)}")
                    data["listingAgent"] = None

            if "listingOffice" in data and data["listingOffice"]:
                try:
                    data["listingOffice"] = ContactInfo(**data["listingOffice"])
                except Exception as e:
                    errors.append(f"Invalid listing office data: {str(e)}")
                    data["listingOffice"] = None

            # Process history events
            if "history" in data and data["history"]:
                history_dict = {}
                for date_key, event_data in data["history"].items():
                    try:
                        history_dict[date_key] = PropertyEvent(**event_data)
                    except Exception as e:
                        errors.append(f"Invalid history event for {date_key}: {str(e)}")
                data["history"] = history_dict

            # Convert date fields
            date_fields = ["listedDate", "removedDate", "createdDate", "lastSeenDate"]
            for field in date_fields:
                if field in data and data[field] and isinstance(data[field], str):
                    try:
                        data[field] = datetime.fromisoformat(data[field].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        errors.append(f"Failed to parse date for {field}: {data[field]}")

            # Create the model
            model = PropertyListing(**data)
            return model, errors

        except Exception as e:
            errors.append(f"Failed to create PropertyListing model: {str(e)}")
            return None, errors

    async def transform_to_market_data(self, data: Dict[str, Any]) -> Tuple[Optional[MarketDataPoint], List[str]]:
        """
        Transform raw market data into a MarketDataPoint model.

        Args:
            data: Raw market data

        Returns:
            Tuple of (MarketDataPoint model, list of validation errors)
        """
        errors = []

        try:
            logger.info(f"Starting market data transformation for data: {data}")

            # Convert date fields
            date_fields = ["date", "last_updated"]
            for field in date_fields:
                if field in data and data[field] and isinstance(data[field], str):
                    try:
                        data[field] = datetime.fromisoformat(data[field].replace("Z", "+00:00"))
                    except (ValueError, TypeError) as e:
                        errors.append(f"Failed to parse date for {field}: {data[field]} - Error: {e}")

            # Set period start and end based on date
            if "date" in data and data["date"]:
                if isinstance(data["date"], str):
                    try:
                        data["date"] = datetime.fromisoformat(data["date"].replace("Z", "+00:00"))
                    except (ValueError, TypeError) as e:
                        errors.append(f"Failed to parse date for 'date' field: {data['date']} - Error: {e}")

                if isinstance(data["date"], datetime):
                    data["period_start"] = data["date"]
                    data["period_end"] = data["date"] + timedelta(days=1)
                else:
                    errors.append(f"Date field is not a valid datetime object: {data['date']}")

            logger.info(f"Data after date transformation: {data}")

            # Convert numeric types to Decimal where needed
            for field in ["median_price", "price_per_sqft"]:
                if field in data and data[field] is not None:
                    try:
                        data[field] = Decimal(str(data[field]))
                    except (ValueError, TypeError) as e:
                        errors.append(f"Invalid {field} value: {data[field]} - Error: {e}")

            logger.info(f"Data after numeric transformation: {data}")

            # Create metrics model from flat fields
            metrics_data = {
                "median_sale_price": data.get("median_price"),
                "active_listings": data.get("inventory_count"),
                "homes_sold": data.get("sales_volume"),
                "new_listings": data.get("new_listings"),
                "days_on_market": data.get("days_on_market"),
                "months_of_supply": data.get("months_supply"),
                "median_sale_ppsf": data.get("price_per_sqft"),
            }
            try:
                data["metrics"] = MarketMetrics(**metrics_data)
            except Exception as e:
                errors.append(f"Invalid metrics data: {str(e)}")
                data["metrics"] = MarketMetrics()

            logger.info(f"Data after metrics transformation: {data}")

            # Create the model
            model = MarketDataPoint(**data)
            logger.info(f"Successfully created MarketDataPoint model: {model}")

            # Sync flat and nested metrics
            model.sync_flat_and_nested_metrics()
            logger.info("Successfully synced flat and nested metrics")

            return model, errors

        except Exception as e:
            logger.error(f"Error in transform_to_market_data: {e}", exc_info=True)
            errors.append(f"Failed to create MarketDataPoint model: {str(e)}")
            return None, errors
