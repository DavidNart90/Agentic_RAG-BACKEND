"""Base validation classes for TrackRealties AI Platform."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.agent import ValidationIssue, ValidationResult


class ResponseValidator(ABC):
    """Base class for all validators."""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version

    @abstractmethod
    async def validate(self, response: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate a response and return validation result."""
        pass

    def create_issue(
        self,
        issue_type: str,
        severity: str,
        description: str,
        field: Optional[str] = None,
        value: Optional[str] = None,
        suggested_correction: Optional[str] = None,
        confidence: float = 1.0,
    ) -> ValidationIssue:
        """Create a validation issue."""
        return ValidationIssue(
            issue_type=issue_type,
            severity=severity,
            field=field,
            value=value,
            description=description,
            suggested_correction=suggested_correction,
            confidence=confidence,
        )

    def create_result(
        self, is_valid: bool, confidence_score: float, issues: List[ValidationIssue], correction_needed: bool = False
    ) -> ValidationResult:
        """Create a validation result."""
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            validation_type=self.name,
            validator_version=self.version,
            correction_needed=correction_needed,
        )


class ValidationContext(BaseModel):
    """Context information for validation."""

    # Query context
    original_query: str = Field(..., description="Original user query")
    user_role: str = Field(..., description="User role")

    # Response context
    response_text: str = Field(..., description="Response to validate")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")

    # Market context
    location_context: Optional[str] = Field(None, description="Location context from query")
    property_context: Optional[Dict[str, Any]] = Field(None, description="Property context")
    market_context: Optional[Dict[str, Any]] = Field(None, description="Market data context")

    # Validation settings
    strict_mode: bool = Field(default=False, description="Whether to use strict validation")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a context value by key."""
        # Check property context first
        if self.property_context and key in self.property_context:
            return self.property_context[key]

        # Check market context
        if self.market_context and key in self.market_context:
            return self.market_context[key]

        return default


class ValidationPipeline:
    """Pipeline for running multiple validators."""

    def __init__(self, validators: List[ResponseValidator]):
        self.validators = validators

    async def validate(self, response: str, context: Dict[str, Any]) -> ValidationResult:
        """Run all validators and combine results."""
        all_issues = []
        min_confidence = 1.0
        any_correction_needed = False

        for validator in self.validators:
            try:
                result = await validator.validate(response, context)
                all_issues.extend(result.issues)
                min_confidence = min(min_confidence, result.confidence_score)

                if result.correction_needed:
                    any_correction_needed = True

            except Exception as e:
                # Log error and create an issue for validator failure
                issue = ValidationIssue(
                    issue_type="validation_error",
                    severity="medium",
                    description=f"Validator {validator.name} failed: {str(e)}",
                    confidence=0.5,
                )
                all_issues.append(issue)

        # Determine overall validity
        critical_issues = [issue for issue in all_issues if issue.severity == "critical"]
        is_valid = len(critical_issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=min_confidence,
            issues=all_issues,
            validation_type="pipeline",
            validator_version="1.0",
            correction_needed=any_correction_needed,
        )

    def add_validator(self, validator: ResponseValidator) -> None:
        """Add a validator to the pipeline."""
        self.validators.append(validator)

    def remove_validator(self, validator_name: str) -> bool:
        """Remove a validator by name."""
        original_count = len(self.validators)
        self.validators = [v for v in self.validators if v.name != validator_name]
        return len(self.validators) < original_count


class ValidationConfig(BaseModel):
    """Configuration for validation system."""

    # General settings
    enabled: bool = Field(default=True, description="Whether validation is enabled")
    strict_mode: bool = Field(default=False, description="Whether to use strict validation")

    # Confidence thresholds
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    correction_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Threshold for triggering corrections")

    # Validator-specific settings
    enable_price_validation: bool = Field(default=True, description="Enable price validation")
    enable_roi_validation: bool = Field(default=True, description="Enable ROI validation")
    enable_geographic_validation: bool = Field(default=True, description="Enable geographic validation")
    enable_market_validation: bool = Field(default=True, description="Enable market metrics validation")
    enable_factual_validation: bool = Field(default=True, description="Enable factual validation")

    # Price validation settings
    price_tolerance_percent: float = Field(default=0.2, ge=0.0, le=1.0, description="Price tolerance as percentage")
    max_reasonable_price_per_sqft: float = Field(default=1000.0, description="Maximum reasonable price per sq ft")

    # ROI validation settings
    max_reasonable_roi: float = Field(default=0.5, description="Maximum reasonable ROI (50%)")
    min_reasonable_roi: float = Field(default=-0.2, description="Minimum reasonable ROI (-20%)")

    # Geographic validation settings
    enable_coordinate_validation: bool = Field(default=True, description="Enable coordinate validation")
    coordinate_precision_tolerance: float = Field(default=0.01, description="Coordinate precision tolerance")

    # Performance settings
    validation_timeout_seconds: int = Field(default=10, ge=1, le=60, description="Validation timeout")
    max_concurrent_validations: int = Field(default=5, ge=1, le=20, description="Maximum concurrent validations")

    # Correction settings
    max_correction_attempts: int = Field(default=3, ge=0, le=10, description="Maximum correction attempts")
    enable_auto_correction: bool = Field(default=True, description="Enable automatic correction")

    def is_validator_enabled(self, validator_type: str) -> bool:
        """Check if a specific validator type is enabled."""
        validator_flags = {
            "price": self.enable_price_validation,
            "roi": self.enable_roi_validation,
            "geographic": self.enable_geographic_validation,
            "market": self.enable_market_validation,
            "factual": self.enable_factual_validation,
        }
        return validator_flags.get(validator_type, True)


class ValidationMetrics(BaseModel):
    """Metrics for validation performance tracking."""

    # Validation counts
    total_validations: int = Field(default=0, description="Total validations performed")
    successful_validations: int = Field(default=0, description="Successful validations")
    failed_validations: int = Field(default=0, description="Failed validations")

    # Issue counts by type
    price_issues: int = Field(default=0, description="Price validation issues")
    roi_issues: int = Field(default=0, description="ROI validation issues")
    geographic_issues: int = Field(default=0, description="Geographic validation issues")
    market_issues: int = Field(default=0, description="Market validation issues")
    factual_issues: int = Field(default=0, description="Factual validation issues")

    # Issue counts by severity
    critical_issues: int = Field(default=0, description="Critical issues found")
    high_issues: int = Field(default=0, description="High severity issues")
    medium_issues: int = Field(default=0, description="Medium severity issues")
    low_issues: int = Field(default=0, description="Low severity issues")

    # Performance metrics
    average_validation_time_ms: float = Field(default=0.0, description="Average validation time")
    total_validation_time_ms: int = Field(default=0, description="Total validation time")

    # Correction metrics
    corrections_attempted: int = Field(default=0, description="Correction attempts")
    corrections_successful: int = Field(default=0, description="Successful corrections")

    # Timestamps
    last_reset: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last metrics reset")

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_validations == 0:
            return 0.0
        return self.successful_validations / self.total_validations

    @property
    def critical_issue_rate(self) -> float:
        """Calculate critical issue rate."""
        if self.total_validations == 0:
            return 0.0
        return self.critical_issues / self.total_validations

    @property
    def correction_success_rate(self) -> float:
        """Calculate correction success rate."""
        if self.corrections_attempted == 0:
            return 0.0
        return self.corrections_successful / self.corrections_attempted

    def record_validation(self, success: bool, validation_time_ms: int, issues: List[ValidationIssue]) -> None:
        """Record a validation result."""
        self.total_validations += 1

        if success:
            self.successful_validations += 1
        else:
            self.failed_validations += 1

        # Update timing
        self.total_validation_time_ms += validation_time_ms
        self.average_validation_time_ms = self.total_validation_time_ms / self.total_validations

        # Count issues by type and severity
        for issue in issues:
            # Count by type
            if issue.issue_type == "price":
                self.price_issues += 1
            elif issue.issue_type == "roi":
                self.roi_issues += 1
            elif issue.issue_type == "geographic":
                self.geographic_issues += 1
            elif issue.issue_type == "metric":
                self.market_issues += 1
            elif issue.issue_type == "factual":
                self.factual_issues += 1

            # Count by severity
            if issue.severity == "critical":
                self.critical_issues += 1
            elif issue.severity == "high":
                self.high_issues += 1
            elif issue.severity == "medium":
                self.medium_issues += 1
            elif issue.severity == "low":
                self.low_issues += 1

    def record_correction(self, success: bool) -> None:
        """Record a correction attempt."""
        self.corrections_attempted += 1
        if success:
            self.corrections_successful += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.__init__()
        self.last_reset = datetime.now(timezone.utc)

    def get_summary(self) -> str:
        """Get a summary of validation metrics."""
        return (
            f"Validations: {self.total_validations} "
            f"(Success: {self.success_rate:.1%}) | "
            f"Critical Issues: {self.critical_issues} "
            f"({self.critical_issue_rate:.1%}) | "
            f"Avg Time: {self.average_validation_time_ms:.0f}ms"
        )
