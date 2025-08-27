"""Agent response and validation models for TrackRealties AI Platform."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import Field, field_validator

from .base import CustomBaseModel as BaseModel
from .base import TimestampMixin


class ValidationIssue(BaseModel):
    """Represents a validation issue found in an AI response."""

    issue_type: Literal["price", "roi", "geographic", "metric", "factual", "consistency"] = Field(
        ..., description="Type of validation issue"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Severity level")
    field: Optional[str] = Field(None, description="Specific field with the issue")
    value: Optional[str] = Field(None, description="Problematic value")
    description: str = Field(..., description="Description of the issue")
    suggested_correction: Optional[str] = Field(None, description="Suggested correction")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this issue detection")

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical issue."""
        return self.severity == "critical"

    @property
    def is_high_confidence(self) -> bool:
        """Check if this issue has high confidence."""
        return self.confidence >= 0.8


class QualityMetrics(BaseModel):
    """Quality metrics for AI responses."""

    # Accuracy metrics
    factual_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Factual accuracy score")
    source_grounding: Optional[float] = Field(None, ge=0.0, le=1.0, description="How well grounded in sources")
    consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Internal consistency score")

    # Completeness metrics
    completeness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Response completeness")
    relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance to query")

    # Role-specific metrics
    role_appropriateness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Appropriateness for user role")
    actionability: Optional[float] = Field(None, ge=0.0, le=1.0, description="How actionable the response is")

    # Technical metrics
    response_time_ms: Optional[int] = Field(None, ge=0, description="Response generation time")
    token_count: Optional[int] = Field(None, ge=0, description="Total tokens used")
    source_count: Optional[int] = Field(None, ge=0, description="Number of sources used")

    def calculate_overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = []

        # Core quality metrics
        if self.factual_accuracy is not None:
            scores.append(self.factual_accuracy * 0.3)  # 30% weight

        if self.source_grounding is not None:
            scores.append(self.source_grounding * 0.2)  # 20% weight

        if self.completeness is not None:
            scores.append(self.completeness * 0.2)  # 20% weight

        if self.relevance is not None:
            scores.append(self.relevance * 0.15)  # 15% weight

        if self.role_appropriateness is not None:
            scores.append(self.role_appropriateness * 0.15)  # 15% weight

        return sum(scores) if scores else 0.0

    @property
    def is_high_quality(self) -> bool:
        """Check if response meets high quality threshold."""
        return self.calculate_overall_score() >= 0.8


class ValidationResult(BaseModel, TimestampMixin):
    """Result of response validation."""

    # Overall validation status
    is_valid: bool = Field(..., description="Whether response passed validation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in response")

    # Detected issues
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of validation issues")

    # Quality assessment
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality metrics")

    # Validation metadata
    validation_type: str = Field(..., description="Type of validation performed")
    validator_version: str = Field(default="1.0", description="Version of validator used")

    # Correction information
    correction_needed: bool = Field(default=False, description="Whether response needs correction")
    correction_attempts: int = Field(default=0, ge=0, description="Number of correction attempts made")
    max_correction_attempts: int = Field(default=3, ge=1, description="Maximum correction attempts allowed")

    @field_validator("issues")
    @classmethod
    def validate_issues(cls, v: List[ValidationIssue]) -> List[ValidationIssue]:
        """Validate and sort issues by severity."""
        # Sort by severity (critical first) and confidence (high first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(v, key=lambda x: (severity_order[x.severity], -x.confidence))

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.is_critical for issue in self.issues)

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues."""
        return [issue for issue in self.issues if issue.is_critical]

    @property
    def high_confidence_issues(self) -> List[ValidationIssue]:
        """Get only high confidence issues."""
        return [issue for issue in self.issues if issue.is_high_confidence]

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)

        # Update correction needed flag
        if issue.severity in ["critical", "high"]:
            self.correction_needed = True

    def get_issues_by_type(self, issue_type: str) -> List[ValidationIssue]:
        """Get issues of a specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    def get_correction_summary(self) -> str:
        """Get a summary of corrections needed."""
        if not self.correction_needed:
            return "No corrections needed."

        critical_count = len(self.critical_issues)
        total_count = len(self.issues)

        summary = f"Found {total_count} validation issues"
        if critical_count > 0:
            summary += f" ({critical_count} critical)"

        return summary + ". Correction recommended."


class ToolCall(BaseModel):
    """Information about a tool call made by an agent."""

    tool_name: str = Field(..., description="Name of the tool called")
    tool_call_id: Optional[str] = Field(None, description="Unique identifier for this tool call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    result: Optional[Any] = Field(None, description="Result returned by the tool")
    execution_time_ms: Optional[int] = Field(None, ge=0, description="Tool execution time in milliseconds")
    success: bool = Field(default=True, description="Whether tool call was successful")
    error_message: Optional[str] = Field(None, description="Error message if tool call failed")

    @property
    def is_search_tool(self) -> bool:
        """Check if this is a search-related tool."""
        search_tools = ["vector_search", "graph_search", "hybrid_search", "property_search", "market_search"]
        return self.tool_name in search_tools

    @property
    def is_analysis_tool(self) -> bool:
        """Check if this is an analysis-related tool."""
        analysis_tools = ["roi_calculator", "market_analyzer", "risk_assessor", "financial_calculator"]
        return self.tool_name in analysis_tools


class SourceDocument(BaseModel):
    """Information about a source document used in response generation."""

    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Document source")
    content_snippet: Optional[str] = Field(None, description="Relevant content snippet")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance to query")
    document_type: Optional[str] = Field(None, description="Type of document")
    url: Optional[str] = Field(None, description="Source URL if available")

    @property
    def is_highly_relevant(self) -> bool:
        """Check if document is highly relevant."""
        return self.relevance_score is not None and self.relevance_score >= 0.8


class AgentResponse(BaseModel, TimestampMixin):
    """Complete response from an AI agent."""

    # Response identification
    response_id: Optional[UUID] = Field(None, description="Unique response identifier")
    session_id: Optional[UUID] = Field(None, description="Session identifier")

    # User context
    user_role: str = Field(..., description="Role of the user (investor, developer, buyer, agent)")
    query: str = Field(..., description="Original user query")

    # Response content
    response: str = Field(..., description="Generated response text")
    response_type: Literal["answer", "recommendation", "analysis", "error"] = Field(
        default="answer", description="Type of response"
    )

    # Tool usage
    tools_used: List[ToolCall] = Field(default_factory=list, description="Tools called during response generation")

    # Source information
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used")

    # Validation and quality
    validation_result: Optional[ValidationResult] = Field(None, description="Validation results")

    # Performance metrics
    processing_time_ms: Optional[int] = Field(None, ge=0, description="Total processing time")
    model_name: Optional[str] = Field(None, description="LLM model used")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")

    @property
    def is_validated(self) -> bool:
        """Check if response has been validated."""
        return self.validation_result is not None

    @property
    def is_high_quality(self) -> bool:
        """Check if response meets high quality standards."""
        if not self.validation_result:
            return False

        return (
            self.validation_result.is_valid and self.validation_result.confidence_score >= 0.8 and not self.validation_result.has_critical_issues
        )

    @property
    def confidence_score(self) -> float:
        """Get confidence score from validation result."""
        if self.validation_result:
            return self.validation_result.confidence_score
        return 0.0

    @property
    def search_tools_used(self) -> List[ToolCall]:
        """Get only search-related tool calls."""
        return [tool for tool in self.tools_used if tool.is_search_tool]

    @property
    def analysis_tools_used(self) -> List[ToolCall]:
        """Get only analysis-related tool calls."""
        return [tool for tool in self.tools_used if tool.is_analysis_tool]

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to the response."""
        self.tools_used.append(tool_call)

    def add_source(self, source: SourceDocument) -> None:
        """Add a source document to the response."""
        self.sources.append(source)

    def get_summary(self) -> str:
        """Get a summary of the response."""
        summary_parts = [
            f"Response for {self.user_role}",
            f"{len(self.tools_used)} tools used",
            f"{len(self.sources)} sources",
        ]

        if self.validation_result:
            summary_parts.append(f"confidence: {self.validation_result.confidence_score:.2f}")

        return " | ".join(summary_parts)


class AgentContext(BaseModel):
    """Context information for agent execution."""

    # Session information
    session_id: UUID = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_role: str = Field(..., description="User role")

    # Conversation context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous messages")
    current_query: str = Field(..., description="Current user query")

    # User preferences
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

    # Search and analysis context
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Current search results")
    analysis_cache: Dict[str, Any] = Field(default_factory=dict, description="Cached analysis results")

    # Performance tracking
    start_time: datetime = Field(default_factory=datetime.now(timezone.utc), description="Context creation time")

    def add_to_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc)().isoformat(),
                "metadata": metadata or {},
            }
        )

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        self.preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.preferences.get(key, default)

    @property
    def processing_time_ms(self) -> int:
        """Get current processing time in milliseconds."""
        return int((datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000)
