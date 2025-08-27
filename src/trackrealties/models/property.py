"""Property data models for TrackRealties AI Platform."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import EmailStr, Field, computed_field, field_validator

from .base import CustomBaseModel as BaseModel
from .base import SourceMixin, TimestampMixin, ValidationMixin


class HOAInfo(BaseModel):
    """Homeowners Association information."""

    fee: Optional[int] = Field(None, description="Monthly HOA fee")
    frequency: Optional[str] = Field(None, description="Fee frequency (monthly, quarterly, annual)")
    amenities: Optional[List[str]] = Field(None, description="HOA amenities")

    @field_validator("fee")
    @classmethod
    def validate_fee(cls, v: Optional[int]) -> Optional[int]:
        """Validate that fee is reasonable."""
        if v is not None and (v < 0 or v > 10000):
            raise ValueError("HOA fee must be between 0 and 10000")
        return v


class ContactInfo(BaseModel):
    """Contact information for agents and offices."""

    name: str = Field(..., description="Contact name")
    phone: Optional[str] = Field(None, description="Phone number")
    email: Optional[EmailStr] = Field(None, description="Email address")
    website: Optional[str] = Field(None, description="Website URL")

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """Basic phone number validation."""
        if v and not any(c.isdigit() for c in v):
            raise ValueError("Phone number must contain at least one digit")
        return v

    @field_validator("website")
    @classmethod
    def validate_website(cls, v: Optional[str]) -> Optional[str]:
        """Basic website URL validation."""
        if v and not (v.startswith("http://") or v.startswith("https://")):
            return f"https://{v}"
        return v


class PropertyEvent(BaseModel):
    """Property history event."""

    event: str = Field(..., description="Event type (Sale Listing, Rental Listing, Price Change, etc.)")
    price: Optional[Decimal] = Field(None, description="Price at time of event")
    listing_type: Optional[str] = Field(None, description="Listing type at time of event")
    listed_date: Optional[datetime] = Field(None, description="Date listed")
    removed_date: Optional[datetime] = Field(None, description="Date removed")
    days_on_market: Optional[int] = Field(None, description="Days on market at time of event")

    @field_validator("event")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type."""
        valid_events = ["Sale Listing", "Rental Listing", "Price Change", "Status Change", "Removed"]
        if v not in valid_events and not v.endswith(" Listing"):
            raise ValueError(f"Invalid event type: {v}")
        return v

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate that price is positive."""
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v


class PropertyListing(BaseModel, TimestampMixin, SourceMixin, ValidationMixin):
    """Property listing data."""

    # Unique identifiers
    id: str = Field(..., description="Unique property identifier")
    mlsNumber: Optional[str] = Field(None, description="MLS listing number")

    # Address information
    formattedAddress: str = Field(..., description="Full formatted address")
    addressLine1: Optional[str] = Field(None, description="Street address line 1")
    addressLine2: Optional[str] = Field(None, description="Street address line 2")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State/Province")
    zipCode: Optional[str] = Field(None, description="ZIP/Postal code")
    county: Optional[str] = Field(None, description="County")

    # Geographic coordinates
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")

    # Property characteristics
    propertyType: str = Field(..., description="Type of property (Single Family, Multi-Family, Condo, etc.)")
    bedrooms: Optional[int] = Field(None, ge=0, description="Number of bedrooms")
    bathrooms: Optional[Decimal] = Field(None, ge=0, description="Number of bathrooms")
    squareFootage: Optional[int] = Field(None, ge=0, description="Interior square footage")
    lotSize: Optional[int] = Field(None, ge=0, description="Lot size in square feet")
    yearBuilt: Optional[int] = Field(None, ge=1800, le=2030, description="Year property was built")

    # HOA information
    hoa: Optional[HOAInfo] = Field(None, description="HOA information")

    # Listing information
    status: str = Field(..., description="Current listing status")
    price: Decimal = Field(..., gt=0, description="Current listing price")
    listingType: Optional[str] = Field(None, description="Type of listing")
    listedDate: Optional[datetime] = Field(None, description="Date property was listed")
    removedDate: Optional[datetime] = Field(None, description="Date property was removed from market")
    daysOnMarket: Optional[int] = Field(None, ge=0, description="Days on market")

    # MLS information
    mlsName: Optional[str] = Field(None, description="MLS system name")

    # Agent and office information
    listingAgent: Optional[ContactInfo] = Field(None, description="Listing agent information")
    listingOffice: Optional[ContactInfo] = Field(None, description="Listing office information")

    # Property history
    history: Dict[str, PropertyEvent] = Field(default_factory=dict, description="Property event history")

    # Additional timestamps
    createdDate: Optional[datetime] = Field(None, description="Date when the listing was created")
    lastSeenDate: Optional[datetime] = Field(None, description="Date when the listing was last seen")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional property metadata")

    @field_validator("state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Validate state format."""
        if len(v) > 10:
            raise ValueError("State code too long")
        return v.upper()

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: Optional[float]) -> Optional[float]:
        """Validate latitude range."""
        if v is not None and not (-90 <= v <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: Optional[float]) -> Optional[float]:
        """Validate longitude range."""
        if v is not None and not (-180 <= v <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        return v

    @property
    def price_per_sqft(self) -> Optional[Decimal]:
        """Calculate price per square foot."""
        if self.squareFootage and self.squareFootage > 0:
            return self.price / Decimal(self.squareFootage)
        return None

    @property
    def is_active(self) -> bool:
        """Check if property is actively listed."""
        active_statuses = ["active", "pending", "under contract", "contingent"]
        return self.status.lower() in active_statuses

    @property
    def is_sold(self) -> bool:
        """Check if property is sold."""
        sold_statuses = ["sold", "closed"]
        return self.status.lower() in sold_statuses

    @property
    def is_rental(self) -> bool:
        """Check if property is a rental listing."""
        if not self.history:
            return False

        for event in self.history.values():
            if event.event == "Rental Listing":
                return True

        return False

    def add_history_event(self, event_date: datetime, event_type: str, **kwargs) -> None:
        """Add an event to property history."""
        date_key = event_date.strftime("%Y-%m-%d")
        self.history[date_key] = PropertyEvent(event=event_type, **kwargs)

    def get_latest_event(self, event_type: Optional[str] = None) -> Optional[PropertyEvent]:
        """Get the most recent event, optionally filtered by type."""
        filtered_events = []

        for date_key, event in self.history.items():
            if event_type is None or event.event == event_type:
                filtered_events.append((date_key, event))

        if not filtered_events:
            return None

        # Sort by date key (ISO format ensures chronological order)
        filtered_events.sort(key=lambda x: x[0], reverse=True)
        return filtered_events[0][1]

    def calculate_days_on_market(self) -> Optional[int]:
        """Calculate current days on market."""
        if not self.listed_date:
            return self.days_on_market

        if self.removed_date:
            return (self.removed_date - self.listed_date).days

        return (datetime.now(timezone.utc) - self.listed_date).days

    def get_price_history(self) -> list[tuple[datetime, Decimal]]:
        """Get chronological price history."""
        price_events = []

        # Add listing price
        if self.listed_date:
            price_events.append((self.listed_date, self.price))

        # Add price change events from history
        for date_key, event in self.history.items():
            if event.price is not None:
                try:
                    event_date = datetime.strptime(date_key, "%Y-%m-%d")
                    price_events.append((event_date, event.price))
                except ValueError:
                    # Skip invalid date keys
                    continue

        # Sort by date
        price_events.sort(key=lambda x: x[0])
        return price_events

    def get_summary(self) -> str:
        """Get a summary description of the property."""
        summary_parts = []

        # Basic info
        if self.bedrooms and self.bathrooms:
            summary_parts.append(f"{self.bedrooms}BR/{self.bathrooms}BA")

        if self.squareFootage:
            summary_parts.append(f"{self.squareFootage:,} sq ft")

        summary_parts.append(f"${self.price:,.0f}")

        if self.propertyType:
            summary_parts.append(self.propertyType)

        basic_info = " ".join(summary_parts)

        return f"{basic_info} at {self.formattedAddress}"


class PropertySearchCriteria(BaseModel):
    """Criteria for property searches."""

    # Location filters
    city: Optional[str] = Field(None, description="City filter")
    state: Optional[str] = Field(None, description="State filter")
    zipCode: Optional[str] = Field(None, description="ZIP code filter")
    county: Optional[str] = Field(None, description="County filter")

    # Property type filters
    propertyTypes: Optional[list[str]] = Field(None, description="Property type filters")

    # Size filters
    min_bedrooms: Optional[int] = Field(None, ge=0, description="Minimum bedrooms")
    max_bedrooms: Optional[int] = Field(None, ge=0, description="Maximum bedrooms")
    min_bathrooms: Optional[Decimal] = Field(None, ge=0, description="Minimum bathrooms")
    max_bathrooms: Optional[Decimal] = Field(None, ge=0, description="Maximum bathrooms")
    min_squareFootage: Optional[int] = Field(None, ge=0, description="Minimum square footage")
    max_squareFootage: Optional[int] = Field(None, ge=0, description="Maximum square footage")

    # Price filters
    min_price: Optional[Decimal] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[Decimal] = Field(None, ge=0, description="Maximum price")

    # Age filters
    min_year_built: Optional[int] = Field(None, ge=1800, description="Minimum year built")
    max_year_built: Optional[int] = Field(None, le=2030, description="Maximum year built")

    # Status filters
    statuses: Optional[list[str]] = Field(None, description="Status filters")

    # Listing type filters
    listing_types: Optional[list[str]] = Field(None, description="Listing type filters (sale, rental)")

    # HOA filters
    has_hoa: Optional[bool] = Field(None, description="Filter for properties with HOA")
    max_hoa_fee: Optional[int] = Field(None, ge=0, description="Maximum HOA fee")

    # Market timing filters
    max_days_on_market: Optional[int] = Field(None, ge=0, description="Maximum days on market")

    # Geographic filters
    latitude: Optional[float] = Field(None, description="Center latitude for radius search")
    longitude: Optional[float] = Field(None, description="Center longitude for radius search")
    radius_miles: Optional[float] = Field(None, ge=0, description="Search radius in miles")

    # Pagination
    limit: int = Field(10, ge=1, le=100, description="Number of results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")

    @field_validator("max_bedrooms")
    @classmethod
    def validate_bedroom_range(cls, v: Optional[int], info) -> Optional[int]:
        """Validate bedroom range."""
        min_bedrooms = info.data.get("min_bedrooms")
        if v is not None and min_bedrooms is not None and v < min_bedrooms:
            raise ValueError("max_bedrooms must be >= min_bedrooms")
        return v

    @field_validator("max_price")
    @classmethod
    def validate_price_range(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Validate price range."""
        min_price = info.data.get("min_price")
        if v is not None and min_price is not None and v < min_price:
            raise ValueError("max_price must be >= min_price")
        return v

    def to_sql_filters(self) -> tuple[str, Dict[str, Any]]:
        """Convert criteria to SQL WHERE clause and parameters."""
        conditions = []
        params = {}
        param_counter = 1

        # Location filters
        if self.city:
            conditions.append(f"city = ${param_counter}")
            params[f"${param_counter}"] = self.city
            param_counter += 1

        if self.state:
            conditions.append(f"state = ${param_counter}")
            params[f"${param_counter}"] = self.state
            param_counter += 1

        if self.zipCode:
            conditions.append(f"zipCode = ${param_counter}")
            params[f"${param_counter}"] = self.zipCode
            param_counter += 1

        # Price filters
        if self.min_price:
            conditions.append(f"price >= ${param_counter}")
            params[f"${param_counter}"] = self.min_price
            param_counter += 1

        if self.max_price:
            conditions.append(f"price <= ${param_counter}")
            params[f"${param_counter}"] = self.max_price
            param_counter += 1

        # Size filters
        if self.min_bedrooms:
            conditions.append(f"bedrooms >= ${param_counter}")
            params[f"${param_counter}"] = self.min_bedrooms
            param_counter += 1

        if self.max_bedrooms:
            conditions.append(f"bedrooms <= ${param_counter}")
            params[f"${param_counter}"] = self.max_bedrooms
            param_counter += 1

        if self.min_squareFootage:
            conditions.append(f"squareFootage >= ${param_counter}")
            params[f"${param_counter}"] = self.min_squareFootage
            param_counter += 1

        if self.max_squareFootage:
            conditions.append(f"squareFootage <= ${param_counter}")
            params[f"${param_counter}"] = self.max_squareFootage
            param_counter += 1

        # Property type filters
        if self.propertyTypes:
            conditions.append(f"propertyType = ANY(${param_counter})")
            params[f"${param_counter}"] = self.propertyTypes
            param_counter += 1

        # Status filters
        if self.statuses:
            conditions.append(f"status = ANY(${param_counter})")
            params[f"${param_counter}"] = self.statuses
            param_counter += 1

        # HOA filters
        if self.has_hoa is not None:
            if self.has_hoa:
                conditions.append("hoa IS NOT NULL")
            else:
                conditions.append("hoa IS NULL")

        if self.max_hoa_fee is not None:
            conditions.append(f"(hoa->>'fee')::int <= ${param_counter}")
            params[f"${param_counter}"] = self.max_hoa_fee
            param_counter += 1

        # Listing type filters
        if self.listing_types:
            listing_conditions = []
            for listing_type in self.listing_types:
                if listing_type.lower() == "rental":
                    listing_conditions.append(
                        "EXISTS (SELECT 1 FROM jsonb_each(history) WHERE value->>'event' = 'Rental Listing')"
                    )
                elif listing_type.lower() == "sale":
                    listing_conditions.append(
                        "EXISTS (SELECT 1 FROM jsonb_each(history) WHERE value->>'event' = 'Sale Listing')"
                    )

            if listing_conditions:
                conditions.append(f"({' OR '.join(listing_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params


class PropertyListingResponse(PropertyListing):
    """Response model for property listings."""

    url: Optional[str] = Field(None, description="URL to the property listing")

    @computed_field
    @property
    def price_per_sqft(self) -> Optional[float]:
        """Calculate price per square foot."""
        if self.squareFootage and self.squareFootage > 0:
            return float(self.price / Decimal(self.squareFootage))
        return None

    @computed_field
    @property
    def summary(self) -> str:
        """Get a summary description of the property."""
        return self.get_summary()


class PropertyListingRequest(BaseModel):
    """Request model for creating or updating a property listing.

    This model is used for creating or updating property listings, including
    partial updates (PATCH). All fields are optional.
    """

    # Override all fields from PropertyListing and its mixins to be optional.

    # --- Identifiers ---
    property_id: Optional[str] = Field(None, description="Unique property identifier")
    mls_number: Optional[str] = Field(None, description="MLS listing number")

    # --- Address ---
    formatted_address: Optional[str] = Field(None, description="Full formatted address")
    address_line1: Optional[str] = Field(None, description="Street address line 1")
    address_line2: Optional[str] = Field(None, description="Street address line 2")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State/Province")
    zipcode: Optional[str] = Field(None, description="ZIP/Postal code")
    county: Optional[str] = Field(None, description="County")

    # --- Geo ---
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")

    # --- Characteristics ---
    property_type: Optional[str] = Field(None, description="Type of property")
    bedrooms: Optional[int] = Field(None, ge=0, description="Number of bedrooms")
    bathrooms: Optional[Decimal] = Field(None, ge=0, description="Number of bathrooms")
    square_feet: Optional[int] = Field(None, ge=0, description="Interior square footage")
    lot_size: Optional[int] = Field(None, ge=0, description="Lot size in square feet")
    year_built: Optional[int] = Field(None, ge=1800, le=2030, description="Year property was built")

    # --- HOA ---
    hoa: Optional[HOAInfo] = Field(None, description="HOA information")

    # --- Listing Details ---
    status: Optional[str] = Field(None, description="Current listing status")
    price: Optional[Decimal] = Field(None, gt=0, description="Current listing price")
    listing_type: Optional[str] = Field(None, description="Type of listing")
    listed_date: Optional[datetime] = Field(None, description="Date property was listed")
    removed_date: Optional[datetime] = Field(None, description="Date property was removed from market")
    days_on_market: Optional[int] = Field(None, ge=0, description="Days on market")

    # --- MLS ---
    mls_name: Optional[str] = Field(None, description="MLS system name")

    # --- Contacts ---
    listing_agent: Optional[ContactInfo] = Field(None, description="Listing agent information")
    listing_office: Optional[ContactInfo] = Field(None, description="Listing office information")

    # --- History & Metadata ---
    history: Optional[Dict[str, PropertyEvent]] = Field(None, description="Property event history")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional property metadata")

    # --- Timestamps ---
    created_date: Optional[datetime] = Field(None, description="Date when the listing was created")
    last_seen_date: Optional[datetime] = Field(None, description="Date when the listing was last seen")
    created_at: Optional[datetime] = Field(None, description="Timestamp of creation")
    updated_at: Optional[datetime] = Field(None, description="Timestamp of last update")
    validated_at: Optional[datetime] = Field(None, description="Timestamp of last validation")

    # --- Source ---
    source: Optional[str] = Field(None, description="Data source identifier")
    source_url: Optional[str] = Field(None, description="URL to the source data")
    source_last_updated: Optional[datetime] = Field(None, description="Timestamp of source data update")

    # --- Validation ---
    validation_status: Optional[str] = Field(None, description="Validation status of the record")
    validation_errors: Optional[List[str]] = Field(None, description="List of validation errors")


class PropertySearchResponse(BaseModel):
    """Response model for property searches."""

    results: List["PropertyListingResponse"]
    total: int
    limit: int
    offset: int
    filters_applied: Dict[str, Any]
