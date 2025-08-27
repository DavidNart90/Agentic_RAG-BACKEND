"""Market data models for TrackRealties AI Platform."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from .base import CustomBaseModel as BaseModel
from .base import SourceMixin, TimestampMixin, ValidationMixin


class MarketMetrics(BaseModel):
    """Market metrics for a specific time period."""

    # Price metrics
    median_sale_price: Optional[Decimal] = Field(None, description="Median sale price")
    median_sale_price_yoy: Optional[float] = Field(None, description="Year-over-year change in median sale price")
    median_new_listing_price: Optional[Decimal] = Field(None, description="Median new listing price")
    median_new_listing_price_yoy: Optional[float] = Field(
        None, description="Year-over-year change in median new listing price"
    )
    median_sale_ppsf: Optional[Decimal] = Field(None, description="Median sale price per square foot")
    median_sale_ppsf_yoy: Optional[float] = Field(
        None, description="Year-over-year change in median sale price per square foot"
    )

    # Inventory metrics
    active_listings: Optional[float] = Field(None, description="Number of active listings")
    active_listings_yoy: Optional[float] = Field(None, description="Year-over-year change in active listings")
    new_listings: Optional[float] = Field(None, description="Number of new listings")
    new_listings_yoy: Optional[float] = Field(None, description="Year-over-year change in new listings")

    # Market activity metrics
    homes_sold: Optional[float] = Field(None, description="Number of homes sold")
    homes_sold_yoy: Optional[float] = Field(None, description="Year-over-year change in homes sold")
    pending_sales: Optional[float] = Field(None, description="Number of pending sales")
    pending_sales_yoy: Optional[float] = Field(None, description="Year-over-year change in pending sales")

    # Market timing metrics
    days_on_market: Optional[float] = Field(None, description="Median days on market")
    days_on_market_yoy: Optional[float] = Field(None, description="Year-over-year change in days on market")
    days_to_close: Optional[float] = Field(None, description="Median days to close")
    days_to_close_yoy: Optional[float] = Field(None, description="Year-over-year change in days to close")

    # Supply metrics
    months_of_supply: Optional[float] = Field(None, description="Months of supply")
    months_supply: Optional[float] = Field(None, description="Months supply (alternative field name)")
    months_of_supply_yoy: Optional[float] = Field(None, description="Year-over-year change in months of supply")
    age_of_inventory: Optional[float] = Field(None, description="Age of inventory in days")
    age_of_inventory_yoy: Optional[float] = Field(None, description="Year-over-year change in age of inventory")

    # Price dynamics
    percent_listings_with_price_drops: Optional[float] = Field(
        None, description="Percentage of listings with price drops"
    )
    percent_listings_with_price_drops_yoy: Optional[float] = Field(
        None, description="Year-over-year change in percentage of listings with price drops"
    )
    sale_to_list_ratio: Optional[float] = Field(None, description="Average sale to list price ratio")
    sale_to_list_ratio_yoy: Optional[float] = Field(None, description="Year-over-year change in sale to list ratio")

    # Market conditions
    off_market_in_two_weeks: Optional[int] = Field(None, description="Properties going off market within two weeks")
    off_market_in_two_weeks_yoy: Optional[float] = Field(
        None, description="Year-over-year change in properties going off market within two weeks"
    )

    @field_validator("median_sale_price", "median_new_listing_price", "median_sale_ppsf")
    @classmethod
    def validate_positive_prices(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate that prices are positive."""
        if v is not None and v <= 0:
            raise ValueError("Price values must be positive")
        return v

    @field_validator("active_listings", "new_listings", "homes_sold", "pending_sales")
    @classmethod
    def validate_non_negative_counts(cls, v: Optional[int]) -> Optional[int]:
        """Validate that counts are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Count values must be non-negative")
        return v


class MarketDataPoint(BaseModel, TimestampMixin, SourceMixin, ValidationMixin):
    """Market data for a specific region and time period."""

    # Region identification
    region_id: str = Field(..., description="Unique region identifier")
    region_name: str = Field(..., description="Human-readable region name")
    region_type: Literal["metro", "county", "city", "zip", "neighborhood"] = Field(..., description="Type of region")

    # Location information (added for alignment with real data)
    location: Optional[str] = Field(None, description="Location string (e.g., 'Cape May County, NJ')")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State code")
    county: Optional[str] = Field(None, description="County name")

    # Geographic coordinates (optional)
    latitude: Optional[float] = Field(None, description="Region latitude")
    longitude: Optional[float] = Field(None, description="Region longitude")

    # Time period
    period_start: datetime = Field(..., description="Start of the data period")
    period_end: datetime = Field(..., description="End of the data period")
    duration: str = Field(..., description="Duration description (e.g., '4 weeks', '1 month')")
    date: Optional[datetime] = Field(None, description="Date of the market data point")
    last_updated: Optional[datetime] = Field(None, description="Last updated timestamp")

    # Market metrics (nested structure)
    metrics: MarketMetrics = Field(..., description="Market metrics for this period")

    # Direct metric fields (flat structure for alignment with real data)
    median_price: Optional[Decimal] = Field(None, description="Median price (flat field)")
    inventory_count: Optional[float] = Field(None, description="Inventory count (flat field)")
    sales_volume: Optional[float] = Field(None, description="Sales volume (flat field)")
    new_listings: Optional[float] = Field(None, description="New listings (flat field)")
    days_on_market: Optional[float] = Field(None, description="Days on market (flat field)")
    months_supply: Optional[float] = Field(None, description="Months of supply (flat field)")
    price_per_sqft: Optional[Decimal] = Field(None, description="Price per square foot (flat field)")

    # Data quality indicators
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality score")
    sample_size: Optional[int] = Field(None, description="Sample size for calculations")

    @field_validator("period_end")
    @classmethod
    def validate_period_order(cls, v: datetime, info) -> datetime:
        """Validate that period_end is after period_start."""
        period_start = info.data.get("period_start")
        if period_start and v <= period_start:
            raise ValueError("period_end must be after period_start")
        return v

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
    def period_duration_days(self) -> int:
        """Calculate the duration of the period in days."""
        return (self.period_end - self.period_start).days

    @property
    def is_current_period(self) -> bool:
        """Check if this data point represents the current period."""
        now = datetime.now(timezone.utc)
        return self.period_start <= now <= self.period_end

    def get_metric_value(self, metric_name: str) -> Optional[Any]:
        """Get a specific metric value by name."""
        # First check direct fields
        if hasattr(self, metric_name):
            return getattr(self, metric_name)

        # Then check nested metrics
        return getattr(self.metrics, metric_name, None)

    def calculate_market_health_score(self) -> Optional[float]:
        """Calculate a composite market health score (0-1)."""
        # Use direct fields if available, otherwise use nested metrics
        score_components = []

        # Price stability (lower YoY change is better for buyers)
        median_price_yoy = getattr(self.metrics, "median_sale_price_yoy", None)
        if median_price_yoy is not None:
            price_stability = max(0, 1 - abs(median_price_yoy) / 0.2)
            score_components.append(price_stability)

        # Market activity (more sales is generally positive)
        sales_volume_yoy = getattr(self.metrics, "homes_sold_yoy", None)
        if sales_volume_yoy is not None:
            activity_score = min(1, max(0, (sales_volume_yoy + 0.1) / 0.3))
            score_components.append(activity_score)

        # Supply balance (3-6 months is ideal)
        months_supply = self.months_supply or getattr(self.metrics, "months_of_supply", None)
        if months_supply is not None:
            ideal_supply = 4.5
            supply_score = max(0, 1 - abs(months_supply - ideal_supply) / ideal_supply)
            score_components.append(supply_score)

        # Market speed (fewer days on market is better)
        dom = self.days_on_market or getattr(self.metrics, "days_on_market", None)
        if dom is not None:
            speed_score = max(0, 1 - (dom - 30) / 90)
            score_components.append(speed_score)

        return sum(score_components) / len(score_components) if score_components else None

    def sync_flat_and_nested_metrics(self) -> None:
        """Synchronize flat fields with nested metrics structure."""
        # Update nested metrics from flat fields if they exist
        if self.median_price is not None and self.metrics.median_sale_price is None:
            self.metrics.median_sale_price = self.median_price

        if self.inventory_count is not None and self.metrics.active_listings is None:
            self.metrics.active_listings = self.inventory_count

        if self.sales_volume is not None and self.metrics.homes_sold is None:
            self.metrics.homes_sold = self.sales_volume

        if self.new_listings is not None and self.metrics.new_listings is None:
            self.metrics.new_listings = self.new_listings

        if self.days_on_market is not None and self.metrics.days_on_market is None:
            self.metrics.days_on_market = self.days_on_market

        if self.months_supply is not None and self.metrics.months_of_supply is None:
            self.metrics.months_of_supply = self.months_supply

        if self.price_per_sqft is not None and self.metrics.median_sale_ppsf is None:
            self.metrics.median_sale_ppsf = self.price_per_sqft

        # Update flat fields from nested metrics if they exist
        if self.median_price is None and self.metrics.median_sale_price is not None:
            self.median_price = self.metrics.median_sale_price

        if self.inventory_count is None and self.metrics.active_listings is not None:
            self.inventory_count = self.metrics.active_listings

        if self.sales_volume is None and self.metrics.homes_sold is not None:
            self.sales_volume = self.metrics.homes_sold

        if self.new_listings is None and self.metrics.new_listings is not None:
            self.new_listings = self.metrics.new_listings

        if self.days_on_market is None and self.metrics.days_on_market is not None:
            self.days_on_market = self.metrics.days_on_market

        if self.months_supply is None and self.metrics.months_of_supply is not None:
            self.months_supply = self.metrics.months_of_supply

        if self.price_per_sqft is None and self.metrics.median_sale_ppsf is not None:
            self.price_per_sqft = self.metrics.median_sale_ppsf


class MarketInsights(BaseModel):
    """Derived insights from market data analysis."""

    region_id: str = Field(..., description="Region identifier")
    analysis_date: datetime = Field(
        default_factory=datetime.now(timezone.utc), description="When analysis was performed"
    )

    # Location information
    location: Optional[str] = Field(None, description="Location string")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State code")
    county: Optional[str] = Field(None, description="County name")

    # Market condition assessment
    market_condition: Literal["hot", "warm", "balanced", "cool", "cold"] = Field(
        ..., description="Overall market condition"
    )
    market_trend: Literal["rising", "stable", "declining"] = Field(..., description="Market trend direction")

    # Key insights
    key_insights: List[str] = Field(default_factory=list, description="Key market insights")
    opportunities: List[str] = Field(default_factory=list, description="Investment opportunities")
    risks: List[str] = Field(default_factory=list, description="Market risks")

    # Predictions
    price_forecast_3m: Optional[float] = Field(None, description="3-month price change forecast")
    price_forecast_6m: Optional[float] = Field(None, description="6-month price change forecast")
    price_forecast_12m: Optional[float] = Field(None, description="12-month price change forecast")

    # Confidence scores
    forecast_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Forecast confidence score")
    data_completeness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data completeness score")

    # Supporting data
    comparable_regions: List[str] = Field(default_factory=list, description="Similar regions for comparison")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used in analysis")

    def add_insight(self, insight: str, category: Literal["insight", "opportunity", "risk"] = "insight") -> None:
        """Add an insight to the appropriate category."""
        if category == "insight":
            self.key_insights.append(insight)
        elif category == "opportunity":
            self.opportunities.append(insight)
        elif category == "risk":
            self.risks.append(insight)

    def get_summary(self) -> str:
        """Get a summary of the market insights."""
        location_str = self.location or f"{self.city}, {self.state}" if self.city and self.state else "this region"
        return f"Market in {location_str} is {self.market_condition} with {self.market_trend} trend. {len(self.key_insights)} insights, {len(self.opportunities)} opportunities, {len(self.risks)} risks identified."


class MarketSearchCriteria(BaseModel):
    """Criteria for market data searches."""

    # Location filters
    location: Optional[str] = Field(None, description="Location search term")
    city: Optional[str] = Field(None, description="City filter")
    state: Optional[str] = Field(None, description="State filter")
    county: Optional[str] = Field(None, description="County filter")
    region_type: Optional[Literal["metro", "county", "city", "zip", "neighborhood"]] = Field(
        None, description="Region type filter"
    )

    # Date filters
    date_from: Optional[datetime] = Field(None, description="Start date for market data")
    date_to: Optional[datetime] = Field(None, description="End date for market data")

    # Metric filters
    min_median_price: Optional[Decimal] = Field(None, ge=0, description="Minimum median price")
    max_median_price: Optional[Decimal] = Field(None, ge=0, description="Maximum median price")
    min_inventory: Optional[int] = Field(None, ge=0, description="Minimum inventory count")
    max_inventory: Optional[int] = Field(None, ge=0, description="Maximum inventory count")
    min_days_on_market: Optional[float] = Field(None, ge=0, description="Minimum days on market")
    max_days_on_market: Optional[float] = Field(None, ge=0, description="Maximum days on market")
    min_months_supply: Optional[float] = Field(None, ge=0, description="Minimum months supply")
    max_months_supply: Optional[float] = Field(None, ge=0, description="Maximum months supply")

    # Trend filters
    price_trend: Optional[Literal["rising", "stable", "declining"]] = Field(None, description="Price trend filter")
    inventory_trend: Optional[Literal["rising", "stable", "declining"]] = Field(
        None, description="Inventory trend filter"
    )

    # Geographic filters
    latitude: Optional[float] = Field(None, description="Center latitude for radius search")
    longitude: Optional[float] = Field(None, description="Center longitude for radius search")
    radius_miles: Optional[float] = Field(None, ge=0, description="Search radius in miles")

    # Pagination
    limit: int = Field(10, ge=1, le=100, description="Number of results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")

    @field_validator("max_median_price")
    @classmethod
    def validate_price_range(cls, v: Optional[Decimal], info) -> Optional[Decimal]:
        """Validate price range."""
        min_price = info.data.get("min_median_price")
        if v is not None and min_price is not None and v < min_price:
            raise ValueError("max_median_price must be >= min_median_price")
        return v

    @field_validator("max_days_on_market")
    @classmethod
    def validate_dom_range(cls, v: Optional[float], info) -> Optional[float]:
        """Validate days on market range."""
        min_dom = info.data.get("min_days_on_market")
        if v is not None and min_dom is not None and v < min_dom:
            raise ValueError("max_days_on_market must be >= min_days_on_market")
        return v

    def to_sql_filters(self) -> tuple[str, Dict[str, Any]]:
        """Convert criteria to SQL WHERE clause and parameters."""
        conditions = []
        params = {}
        param_counter = 1

        # Location filters
        if self.location:
            conditions.append(f"location ILIKE ${param_counter}")
            params[f"${param_counter}"] = f"%{self.location}%"
            param_counter += 1

        if self.city:
            conditions.append(f"city = ${param_counter}")
            params[f"${param_counter}"] = self.city
            param_counter += 1

        if self.state:
            conditions.append(f"state = ${param_counter}")
            params[f"${param_counter}"] = self.state
            param_counter += 1

        if self.county:
            conditions.append(f"county = ${param_counter}")
            params[f"${param_counter}"] = self.county
            param_counter += 1

        if self.region_type:
            conditions.append(f"region_type = ${param_counter}")
            params[f"${param_counter}"] = self.region_type
            param_counter += 1

        # Date filters
        if self.date_from:
            conditions.append(f"date >= ${param_counter}")
            params[f"${param_counter}"] = self.date_from
            param_counter += 1

        if self.date_to:
            conditions.append(f"date <= ${param_counter}")
            params[f"${param_counter}"] = self.date_to
            param_counter += 1

        # Metric filters - use both flat fields and nested metrics
        if self.min_median_price:
            conditions.append(
                f"(median_price >= ${param_counter} OR (metrics->>'median_sale_price')::decimal >= ${param_counter})"
            )
            params[f"${param_counter}"] = self.min_median_price
            param_counter += 1

        if self.max_median_price:
            conditions.append(
                f"(median_price <= ${param_counter} OR (metrics->>'median_sale_price')::decimal <= ${param_counter})"
            )
            params[f"${param_counter}"] = self.max_median_price
            param_counter += 1

        if self.min_inventory:
            conditions.append(
                f"(inventory_count >= ${param_counter} OR (metrics->>'active_listings')::int >= ${param_counter})"
            )
            params[f"${param_counter}"] = self.min_inventory
            param_counter += 1

        if self.max_inventory:
            conditions.append(
                f"(inventory_count <= ${param_counter} OR (metrics->>'active_listings')::int <= ${param_counter})"
            )
            params[f"${param_counter}"] = self.max_inventory
            param_counter += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params


class MarketDataResponse(MarketDataPoint):
    """Response model for market data, including derived fields."""

    market_health_score: Optional[float] = Field(None, description="Composite market health score")


class MarketSearchResponse(BaseModel):
    """Response model for market data searches."""

    results: List[MarketDataResponse]
    total: int
    limit: int
    offset: int
    filters_applied: Dict[str, Any]
