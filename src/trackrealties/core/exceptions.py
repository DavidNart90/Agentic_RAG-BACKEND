"""Custom exceptions for TrackRealties AI Platform."""

from typing import Any, Dict, Optional


class TrackRealtiesError(Exception):
    """Base exception for TrackRealties AI Platform."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(TrackRealtiesError):
    """Raised when there's a configuration error."""

    pass


class DatabaseError(TrackRealtiesError):
    """Raised when there's a database-related error."""

    pass


class VectorSearchError(TrackRealtiesError):
    """Raised when there's a vector search error."""

    pass


class GraphSearchError(TrackRealtiesError):
    """Raised when there's a graph search error."""

    pass


class ExternalAPIError(TrackRealtiesError):
    """Raised when there's an external API error."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code
        self.response_data = response_data or {}


class ValidationError(TrackRealtiesError):
    """Raised when there's a validation error."""

    def __init__(self, message: str, validation_type: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.field = field


class HallucinationDetectedError(ValidationError):
    """Raised when hallucination is detected in AI response."""

    def __init__(self, message: str, confidence_score: float, detected_issues: list, **kwargs):
        super().__init__(message, validation_type="hallucination", **kwargs)
        self.confidence_score = confidence_score
        self.detected_issues = detected_issues


class AgentError(TrackRealtiesError):
    """Raised when there's an agent-related error."""

    def __init__(self, message: str, agent_type: str, query: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.agent_type = agent_type
        self.query = query


class RateLimitError(TrackRealtiesError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(TrackRealtiesError):
    """Raised when there's an authentication error."""

    pass


class AuthorizationError(TrackRealtiesError):
    """Raised when there's an authorization error."""

    pass


class DataProcessingError(TrackRealtiesError):
    """Raised when there's a data processing error."""

    def __init__(self, message: str, data_type: str, processing_stage: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.processing_stage = processing_stage


class ModelError(TrackRealtiesError):
    """Raised when there's an LLM model error."""

    def __init__(self, message: str, model_name: str, provider: str, **kwargs):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.provider = provider


class CacheError(TrackRealtiesError):
    """Raised when there's a caching error."""

    pass


class SessionError(TrackRealtiesError):
    """Raised when there's a session-related error."""

    def __init__(self, message: str, session_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.session_id = session_id


class PropertyDataError(DataProcessingError):
    """Raised when there's a property data error."""

    def __init__(self, message: str, property_id: Optional[str] = None, **kwargs):
        super().__init__(message, data_type="property", **kwargs)
        self.property_id = property_id


class MarketDataError(DataProcessingError):
    """Raised when there's a market data error."""

    def __init__(self, message: str, region_id: Optional[str] = None, **kwargs):
        super().__init__(message, data_type="market", **kwargs)
        self.region_id = region_id


class FinancialCalculationError(TrackRealtiesError):
    """Raised when there's a financial calculation error."""

    def __init__(self, message: str, calculation_type: str, input_data: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.calculation_type = calculation_type
        self.input_data = input_data or {}


class NotFoundError(TrackRealtiesError):
    """Raised when a resource is not found."""

    pass
