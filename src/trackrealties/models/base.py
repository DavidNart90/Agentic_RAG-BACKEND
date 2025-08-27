"""Base models for TrackRealties AI Platform."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, EmailStr, Field
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CustomBaseModel(PydanticBaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class TimestampMixin:
    """Mixin for models that need timestamp fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class UUIDMixin:
    """Mixin for models that need UUID primary keys."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")


class MetadataMixin:
    """Mixin for models that need metadata fields."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata key-value pair."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default)


class SourceMixin:
    """Mixin for models that track data sources."""

    source: str = Field(..., description="Data source identifier")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")


class ValidationMixin:
    """Mixin for models that need validation tracking."""

    is_validated: bool = Field(default=False, description="Whether data has been validated")
    validation_errors: List[str] = Field(default_factory=list, description="Validation error messages")
    validation_timestamp: Optional[datetime] = Field(None, description="When validation was performed")

    def mark_validated(self, errors: Optional[List[str]] = None) -> None:
        """Mark as validated with optional errors."""
        self.is_validated = True
        self.validation_errors = errors or []
        self.validation_timestamp = datetime.now(timezone.utc)

    def add_validation_error(self, error: str) -> None:
        """Add a validation error."""
        self.validation_errors.append(error)
        self.is_validated = False
