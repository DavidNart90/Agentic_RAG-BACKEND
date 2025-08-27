"""Search models for TrackRealties AI Platform."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import Field, field_validator

from .base import CustomBaseModel as BaseModel


class SearchRequest(BaseModel):
    """Request for search operations."""

    # Query information
    query: str = Field(..., description="Search query text")
    search_type: Literal["vector", "graph", "hybrid", "external"] = Field(
        default="hybrid", description="Type of search to perform"
    )

    # Search parameters
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve from the database")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")

    # Filters
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")

    # Context
    user_role: Optional[str] = Field(None, description="User role for context")
    session_id: Optional[UUID] = Field(None, description="Session identifier")

    # Performance settings
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Search timeout")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result."""

    # Result identification
    result_id: str = Field(..., description="Unique result identifier")
    result_type: Literal["market_data", "property_listing", "document", "graph_fact", "agent"] = Field(
        ..., description="Type of result"
    )

    # Content
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content or summary")

    # Relevance scoring
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to query")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Vector similarity score")

    # Source information
    source: str = Field(..., description="Data source")
    source_url: Optional[str] = Field(None, description="Source URL if available")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")

    # Timestamps
    created_at: Optional[datetime] = Field(None, description="When result was created")
    last_updated: Optional[datetime] = Field(None, description="When result was last updated")

    @property
    def is_highly_relevant(self) -> bool:
        """Check if result is highly relevant."""
        return self.relevance_score >= 0.8

    @property
    def is_recent(self, days: int = 30) -> bool:
        """Check if result is recent."""
        if not self.last_updated:
            return False

        age_days = (datetime.now(timezone.utc)() - self.last_updated).days
        return age_days <= days


class MarketDataSearchResult(SearchResult):
    """Search result for market data."""

    result_type: Literal["market_data"] = "market_data"

    # Market-specific fields
    region_id: str = Field(..., description="Region identifier")
    region_name: str = Field(..., description="Region name")
    region_type: str = Field(..., description="Region type")
    period_start: datetime = Field(..., description="Data period start")
    period_end: datetime = Field(..., description="Data period end")

    # Key metrics (subset for search results)
    median_price: Optional[float] = Field(None, description="Median sale price")
    price_change_yoy: Optional[float] = Field(None, description="Year-over-year price change")
    active_listings: Optional[int] = Field(None, description="Active listings count")
    days_on_market: Optional[float] = Field(None, description="Average days on market")


class PropertySearchResult(SearchResult):
    """Search result for property listings."""

    result_type: Literal["property_listing"] = "property_listing"

    # Property-specific fields
    property_id: str = Field(..., description="Property identifier")
    address: str = Field(..., description="Property address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")

    # Key property details
    property_type: str = Field(..., description="Property type")
    price: float = Field(..., description="Listing price")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    square_footage: Optional[int] = Field(None, description="Square footage")

    # Listing details
    status: str = Field(..., description="Listing status")
    days_on_market: Optional[int] = Field(None, description="Days on market")


class GraphSearchResult(SearchResult):
    """Search result from knowledge graph."""

    result_type: Literal["graph_fact"] = "graph_fact"

    # Graph-specific fields
    fact_id: str = Field(..., description="Fact identifier")
    entities: List[str] = Field(default_factory=list, description="Related entities")
    relationships: List[str] = Field(default_factory=list, description="Relationship types")

    # Temporal information
    valid_from: Optional[datetime] = Field(None, description="When fact became valid")
    valid_until: Optional[datetime] = Field(None, description="When fact expires")

    @property
    def is_current(self) -> bool:
        """Check if graph fact is currently valid."""
        now = datetime.now(timezone.utc)

        if self.valid_from and now < self.valid_from:
            return False

        if self.valid_until and now > self.valid_until:
            return False

        return True


class SearchResponse(BaseModel):
    """Response from search operations."""

    # Request context
    query: str = Field(..., description="Original search query")
    search_type: str = Field(..., description="Type of search performed")

    # Results
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results found")

    # Performance metrics
    search_time_ms: int = Field(..., ge=0, description="Search execution time")

    # Search metadata
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Filters that were applied")
    sources_searched: List[str] = Field(default_factory=list, description="Data sources that were searched")

    # Quality indicators
    average_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average relevance score")
    has_high_quality_results: bool = Field(default=False, description="Whether results include high-quality matches")

    # Pagination
    offset: int = Field(default=0, ge=0, description="Result offset")
    limit: int = Field(default=10, ge=1, description="Result limit")
    has_more: bool = Field(default=False, description="Whether more results are available")

    @field_validator("results")
    @classmethod
    def validate_results(cls, v: List[SearchResult]) -> List[SearchResult]:
        """Sort results by relevance score."""
        return sorted(v, key=lambda x: x.relevance_score, reverse=True)

    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the top-ranked result."""
        return self.results[0] if self.results else None

    @property
    def high_relevance_results(self) -> List[SearchResult]:
        """Get only high-relevance results."""
        return [r for r in self.results if r.is_highly_relevant]

    def get_results_by_type(self, result_type: str) -> List[SearchResult]:
        """Get results of a specific type."""
        return [r for r in self.results if r.result_type == result_type]

    def calculate_average_relevance(self) -> float:
        """Calculate average relevance score."""
        if not self.results:
            return 0.0

        total_relevance = sum(r.relevance_score for r in self.results)
        return total_relevance / len(self.results)

    def get_summary(self) -> str:
        """Get a summary of search results."""
        if not self.results:
            return f"No results found for '{self.query}'"

        summary_parts = [f"Found {self.total_results} results", f"in {self.search_time_ms}ms"]

        if self.average_relevance:
            summary_parts.append(f"avg relevance: {self.average_relevance:.2f}")

        high_quality_count = len(self.high_relevance_results)
        if high_quality_count > 0:
            summary_parts.append(f"{high_quality_count} high-quality")

        return " | ".join(summary_parts)


class SearchFilter(BaseModel):
    """Filter for search operations."""

    field: str = Field(..., description="Field to filter on")
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "starts_with", "ends_with"] = (
        Field(..., description="Filter operator")
    )
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(..., description="Filter value")

    def to_sql_condition(self, table_alias: str = "") -> tuple[str, Any]:
        """Convert filter to SQL condition."""
        field_name = f"{table_alias}.{self.field}" if table_alias else self.field

        if self.operator == "eq":
            return f"{field_name} = %s", self.value
        elif self.operator == "ne":
            return f"{field_name} != %s", self.value
        elif self.operator == "gt":
            return f"{field_name} > %s", self.value
        elif self.operator == "gte":
            return f"{field_name} >= %s", self.value
        elif self.operator == "lt":
            return f"{field_name} < %s", self.value
        elif self.operator == "lte":
            return f"{field_name} <= %s", self.value
        elif self.operator == "in":
            placeholders = ",".join(["%s"] * len(self.value))
            return f"{field_name} IN ({placeholders})", self.value
        elif self.operator == "not_in":
            placeholders = ",".join(["%s"] * len(self.value))
            return f"{field_name} NOT IN ({placeholders})", self.value
        elif self.operator == "contains":
            return f"{field_name} ILIKE %s", f"%{self.value}%"
        elif self.operator == "starts_with":
            return f"{field_name} ILIKE %s", f"{self.value}%"
        elif self.operator == "ends_with":
            return f"{field_name} ILIKE %s", f"%{self.value}"
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")


class SearchAggregation(BaseModel):
    """Aggregation for search results."""

    field: str = Field(..., description="Field to aggregate")
    aggregation_type: Literal["count", "sum", "avg", "min", "max", "distinct_count"] = Field(
        ..., description="Type of aggregation"
    )
    result: Union[int, float, str] = Field(..., description="Aggregation result")

    @property
    def formatted_result(self) -> str:
        """Get formatted aggregation result."""
        if self.aggregation_type == "count":
            return f"{self.result:,}"
        elif self.aggregation_type in ["sum", "avg"]:
            return f"{self.result:,.2f}"
        else:
            return str(self.result)


class SearchQuery(BaseModel):
    """Search query model."""

    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None


class SearchFilters(BaseModel):
    """Search filters model."""

    filters: List[SearchFilter] = []


class QueryRequest(BaseModel):
    """Request for intelligent querying."""

    query: str = Field(..., description="The user's natural language query.")
