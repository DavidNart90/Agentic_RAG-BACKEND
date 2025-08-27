"""Enumeration classes for TrackRealties AI Platform."""

from enum import Enum
from typing import List


class PropertyType(str, Enum):
    """Property type enumeration."""

    SINGLE_FAMILY = "Single Family"
    CONDO = "Condo"
    TOWNHOUSE = "Townhouse"
    MULTI_FAMILY = "Multi-Family"
    DUPLEX = "Duplex"
    TRIPLEX = "Triplex"
    FOURPLEX = "Fourplex"
    APARTMENT = "Apartment"
    COMMERCIAL = "Commercial"
    OFFICE = "Office"
    RETAIL = "Retail"
    INDUSTRIAL = "Industrial"
    WAREHOUSE = "Warehouse"
    LAND = "Land"
    LOT = "Lot"
    MOBILE_HOME = "Mobile Home"
    MANUFACTURED = "Manufactured"
    FARM = "Farm"
    RANCH = "Ranch"
    OTHER = "Other"

    @classmethod
    def residential_types(cls) -> List[str]:
        """Get residential property types."""
        return [
            cls.SINGLE_FAMILY,
            cls.CONDO,
            cls.TOWNHOUSE,
            cls.MULTI_FAMILY,
            cls.DUPLEX,
            cls.TRIPLEX,
            cls.FOURPLEX,
            cls.APARTMENT,
            cls.MOBILE_HOME,
            cls.MANUFACTURED,
        ]

    @classmethod
    def commercial_types(cls) -> List[str]:
        """Get commercial property types."""
        return [
            cls.COMMERCIAL,
            cls.OFFICE,
            cls.RETAIL,
            cls.INDUSTRIAL,
            cls.WAREHOUSE,
        ]

    @classmethod
    def land_types(cls) -> List[str]:
        """Get land property types."""
        return [
            cls.LAND,
            cls.LOT,
            cls.FARM,
            cls.RANCH,
        ]

    def is_residential(self) -> bool:
        """Check if property type is residential."""
        return self.value in self.residential_types()

    def is_commercial(self) -> bool:
        """Check if property type is commercial."""
        return self.value in self.commercial_types()

    def is_land(self) -> bool:
        """Check if property type is land."""
        return self.value in self.land_types()


class ListingStatus(str, Enum):
    """Listing status enumeration."""

    ACTIVE = "Active"
    PENDING = "Pending"
    UNDER_CONTRACT = "Under Contract"
    CONTINGENT = "Contingent"
    SOLD = "Sold"
    CLOSED = "Closed"
    WITHDRAWN = "Withdrawn"
    EXPIRED = "Expired"
    CANCELLED = "Cancelled"
    COMING_SOON = "Coming Soon"
    TEMPORARILY_OFF_MARKET = "Temporarily Off Market"
    PRICE_REDUCED = "Price Reduced"
    BACK_ON_MARKET = "Back on Market"

    @classmethod
    def active_statuses(cls) -> List[str]:
        """Get statuses that indicate property is actively for sale."""
        return [
            cls.ACTIVE,
            cls.PRICE_REDUCED,
            cls.BACK_ON_MARKET,
            cls.COMING_SOON,
        ]

    @classmethod
    def pending_statuses(cls) -> List[str]:
        """Get statuses that indicate property is under contract."""
        return [
            cls.PENDING,
            cls.UNDER_CONTRACT,
            cls.CONTINGENT,
        ]

    @classmethod
    def sold_statuses(cls) -> List[str]:
        """Get statuses that indicate property is sold."""
        return [
            cls.SOLD,
            cls.CLOSED,
        ]

    @classmethod
    def inactive_statuses(cls) -> List[str]:
        """Get statuses that indicate property is not available."""
        return [
            cls.WITHDRAWN,
            cls.EXPIRED,
            cls.CANCELLED,
            cls.TEMPORARILY_OFF_MARKET,
        ]

    def is_active(self) -> bool:
        """Check if status indicates property is actively for sale."""
        return self.value in self.active_statuses()

    def is_pending(self) -> bool:
        """Check if status indicates property is under contract."""
        return self.value in self.pending_statuses()

    def is_sold(self) -> bool:
        """Check if status indicates property is sold."""
        return self.value in self.sold_statuses()

    def is_available(self) -> bool:
        """Check if property is available for purchase."""
        return self.is_active() or self.is_pending()


class UserRole(str, Enum):
    """User role enumeration."""

    INVESTOR = "investor"
    DEVELOPER = "developer"
    BUYER = "buyer"
    AGENT = "agent"
    GENERAL = "general"

    @classmethod
    def professional_roles(cls) -> List[str]:
        """Get professional user roles."""
        return [
            cls.INVESTOR,
            cls.DEVELOPER,
            cls.AGENT,
        ]

    @classmethod
    def consumer_roles(cls) -> List[str]:
        """Get consumer user roles."""
        return [
            cls.BUYER,
            cls.GENERAL,
        ]

    def is_professional(self) -> bool:
        """Check if role is a professional role."""
        return self.value in self.professional_roles()

    def is_consumer(self) -> bool:
        """Check if role is a consumer role."""
        return self.value in self.consumer_roles()

    def get_role_description(self) -> str:
        """Get description of the role."""
        descriptions = {
            self.INVESTOR: "Real estate investor focused on ROI and cash flow analysis",
            self.DEVELOPER: "Real estate developer focused on development opportunities and feasibility",
            self.BUYER: "Home buyer looking for residential properties",
            self.AGENT: "Real estate agent helping clients buy and sell properties",
            self.GENERAL: "General user with varied real estate interests",
        }
        return descriptions.get(self, "Unknown role")


class RegionType(str, Enum):
    """Region type enumeration for market data."""

    METRO = "metro"
    COUNTY = "county"
    CITY = "city"
    ZIP = "zip"
    NEIGHBORHOOD = "neighborhood"
    STATE = "state"
    MSA = "msa"  # Metropolitan Statistical Area

    @classmethod
    def hierarchical_order(cls) -> List[str]:
        """Get region types in hierarchical order (largest to smallest)."""
        return [
            cls.STATE,
            cls.MSA,
            cls.METRO,
            cls.COUNTY,
            cls.CITY,
            cls.ZIP,
            cls.NEIGHBORHOOD,
        ]

    def get_hierarchy_level(self) -> int:
        """Get hierarchy level (0 = largest, higher = smaller)."""
        hierarchy = self.hierarchical_order()
        try:
            return hierarchy.index(self.value)
        except ValueError:
            return len(hierarchy)  # Unknown types go to the end


class ListingType(str, Enum):
    """Listing type enumeration."""

    FOR_SALE = "For Sale"
    FOR_RENT = "For Rent"
    SOLD = "Sold"
    RENTED = "Rented"
    AUCTION = "Auction"
    FORECLOSURE = "Foreclosure"
    SHORT_SALE = "Short Sale"
    NEW_CONSTRUCTION = "New Construction"
    PRE_CONSTRUCTION = "Pre-Construction"
    COMING_SOON = "Coming Soon"

    def is_sale_listing(self) -> bool:
        """Check if listing is for sale."""
        sale_types = [
            self.FOR_SALE,
            self.AUCTION,
            self.FORECLOSURE,
            self.SHORT_SALE,
            self.NEW_CONSTRUCTION,
            self.PRE_CONSTRUCTION,
            self.COMING_SOON,
        ]
        return self.value in sale_types

    def is_rental_listing(self) -> bool:
        """Check if listing is for rent."""
        return self.value == self.FOR_RENT

    def is_historical(self) -> bool:
        """Check if listing represents historical data."""
        return self.value in [self.SOLD, self.RENTED]


class SearchType(str, Enum):
    """Search type enumeration."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    EXTERNAL = "external"
    KEYWORD = "keyword"

    def requires_embeddings(self) -> bool:
        """Check if search type requires vector embeddings."""
        return self.value in [self.VECTOR, self.HYBRID]

    def requires_graph(self) -> bool:
        """Check if search type requires graph database."""
        return self.value in [self.GRAPH, self.HYBRID]

    def requires_external_api(self) -> bool:
        """Check if search type requires external API calls."""
        return self.value == self.EXTERNAL


class ValidationSeverity(str, Enum):
    """Validation issue severity enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def severity_order(cls) -> List[str]:
        """Get severity levels in order (lowest to highest)."""
        return [cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]

    def get_severity_level(self) -> int:
        """Get numeric severity level (0 = lowest, higher = more severe)."""
        severity_order = self.severity_order()
        try:
            return severity_order.index(self.value)
        except ValueError:
            return 0

    def is_actionable(self) -> bool:
        """Check if severity level requires action."""
        return self.value in [self.HIGH, self.CRITICAL]


class ValidationIssueType(str, Enum):
    """Validation issue type enumeration."""

    PRICE = "price"
    ROI = "roi"
    GEOGRAPHIC = "geographic"
    METRIC = "metric"
    FACTUAL = "factual"
    CONSISTENCY = "consistency"
    CALCULATION = "calculation"
    VALIDATION_ERROR = "validation_error"

    def get_description(self) -> str:
        """Get description of the validation issue type."""
        descriptions = {
            self.PRICE: "Price-related validation issues",
            self.ROI: "ROI and financial projection issues",
            self.GEOGRAPHIC: "Geographic and location consistency issues",
            self.METRIC: "Market metrics validation issues",
            self.FACTUAL: "Factual accuracy issues",
            self.CONSISTENCY: "Internal consistency issues",
            self.CALCULATION: "Mathematical calculation errors",
            self.VALIDATION_ERROR: "Validation system errors",
        }
        return descriptions.get(self, "Unknown validation issue type")


class MarketCondition(str, Enum):
    """Market condition enumeration."""

    HOT = "hot"
    WARM = "warm"
    BALANCED = "balanced"
    COOL = "cool"
    COLD = "cold"

    def get_description(self) -> str:
        """Get description of market condition."""
        descriptions = {
            self.HOT: "Seller's market with high demand and low inventory",
            self.WARM: "Favorable seller's market with good demand",
            self.BALANCED: "Balanced market with equal buyer and seller advantages",
            self.COOL: "Buyer's market with lower demand and higher inventory",
            self.COLD: "Strong buyer's market with very low demand",
        }
        return descriptions.get(self, "Unknown market condition")

    def is_sellers_market(self) -> bool:
        """Check if condition favors sellers."""
        return self.value in [self.HOT, self.WARM]

    def is_buyers_market(self) -> bool:
        """Check if condition favors buyers."""
        return self.value in [self.COOL, self.COLD]


class MarketTrend(str, Enum):
    """Market trend enumeration."""

    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

    def get_description(self) -> str:
        """Get description of market trend."""
        descriptions = {
            self.RISING: "Market values are increasing",
            self.STABLE: "Market values are relatively stable",
            self.DECLINING: "Market values are decreasing",
            self.VOLATILE: "Market values are fluctuating significantly",
        }
        return descriptions.get(self, "Unknown market trend")

    def is_positive(self) -> bool:
        """Check if trend is positive for property values."""
        return self.value == self.RISING

    def is_negative(self) -> bool:
        """Check if trend is negative for property values."""
        return self.value == self.DECLINING


class MessageRole(str, Enum):
    """Enumeration for message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
