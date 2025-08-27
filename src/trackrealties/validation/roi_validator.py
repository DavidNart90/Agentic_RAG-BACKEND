"""ROI and financial projection validation for real estate responses."""

import re
from typing import Any

from ..models.agent import ValidationIssue, ValidationResult
from .base import ResponseValidator


class ROIValidator(ResponseValidator):
    """Validates ROI projections and financial calculations."""

    def __init__(self):
        super().__init__("roi_validator", "1.0")

        # Reasonable ROI ranges for different investment types
        self.roi_ranges = {
            "rental_property": {"min": -0.05, "max": 0.25},  # -5% to 25%
            "fix_and_flip": {"min": 0.05, "max": 0.50},  # 5% to 50%
            "commercial": {"min": 0.03, "max": 0.20},  # 3% to 20%
            "reit": {"min": 0.02, "max": 0.15},  # 2% to 15%
            "land": {"min": -0.10, "max": 0.30},  # -10% to 30%
        }

        # Cap rate ranges by property type
        self.cap_rate_ranges = {
            "single_family": {"min": 0.04, "max": 0.12},
            "multi_family": {"min": 0.05, "max": 0.10},
            "commercial": {"min": 0.06, "max": 0.12},
            "retail": {"min": 0.06, "max": 0.10},
            "office": {"min": 0.05, "max": 0.09},
        }

    async def validate(self, response: str, context: dict[str, Any]) -> ValidationResult:
        """Validate ROI and financial projections in the response."""
        issues = []

        # Extract ROI mentions
        roi_mentions = self._extract_roi_mentions(response)

        # Extract cap rate mentions
        cap_rate_mentions = self._extract_cap_rate_mentions(response)

        # Extract cash-on-cash return mentions
        cash_on_cash_mentions = self._extract_cash_on_cash_mentions(response)

        # Validate each type of financial metric
        for roi_info in roi_mentions:
            roi_issues = await self._validate_roi(roi_info, context)
            issues.extend(roi_issues)

        for cap_rate_info in cap_rate_mentions:
            cap_rate_issues = await self._validate_cap_rate(cap_rate_info, context)
            issues.extend(cap_rate_issues)

        for cash_on_cash_info in cash_on_cash_mentions:
            cash_issues = await self._validate_cash_on_cash(cash_on_cash_info, context)
            issues.extend(cash_issues)

        # Check for calculation consistency
        consistency_issues = self._check_financial_consistency(
            roi_mentions, cap_rate_mentions, cash_on_cash_mentions, response
        )
        issues.extend(consistency_issues)

        # Determine overall validity
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        is_valid = len(critical_issues) == 0

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(issues)

        return self.create_result(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            correction_needed=len(critical_issues) > 0,
        )

    def _extract_roi_mentions(self, text: str) -> list[dict[str, Any]]:
        """Extract ROI mentions from text."""
        roi_mentions = []

        patterns = [
            r"ROI\s+of\s+([0-9.]+)%",
            r"return\s+on\s+investment\s+of\s+([0-9.]+)%",
            r"([0-9.]+)%\s+ROI",
            r"([0-9.]+)%\s+return\s+on\s+investment",
            r"ROI:\s*([0-9.]+)%",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    roi_percent = float(match.group(1))
                    roi_decimal = roi_percent / 100.0

                    roi_mentions.append(
                        {
                            "roi": roi_decimal,
                            "roi_percent": roi_percent,
                            "text": match.group(0),
                            "position": match.span(),
                        }
                    )
                except ValueError:
                    continue

        return roi_mentions

    def _extract_cap_rate_mentions(self, text: str) -> list[dict[str, Any]]:
        """Extract cap rate mentions from text."""
        cap_rate_mentions = []

        patterns = [
            r"cap\s+rate\s+of\s+([0-9.]+)%",
            r"capitalization\s+rate\s+of\s+([0-9.]+)%",
            r"([0-9.]+)%\s+cap\s+rate",
            r"cap\s+rate:\s*([0-9.]+)%",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    cap_rate_percent = float(match.group(1))
                    cap_rate_decimal = cap_rate_percent / 100.0

                    cap_rate_mentions.append(
                        {
                            "cap_rate": cap_rate_decimal,
                            "cap_rate_percent": cap_rate_percent,
                            "text": match.group(0),
                            "position": match.span(),
                        }
                    )
                except ValueError:
                    continue

        return cap_rate_mentions

    def _extract_cash_on_cash_mentions(self, text: str) -> list[dict[str, Any]]:
        """Extract cash-on-cash return mentions from text."""
        cash_mentions = []

        patterns = [
            r"cash[- ]on[- ]cash\s+(?:return\s+)?of\s+([0-9.]+)%",
            r"([0-9.]+)%\s+cash[- ]on[- ]cash",
            r"cash[- ]on[- ]cash:\s*([0-9.]+)%",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    cash_percent = float(match.group(1))
                    cash_decimal = cash_percent / 100.0

                    cash_mentions.append(
                        {
                            "cash_on_cash": cash_decimal,
                            "cash_on_cash_percent": cash_percent,
                            "text": match.group(0),
                            "position": match.span(),
                        }
                    )
                except ValueError:
                    continue

        return cash_mentions

    async def _validate_roi(self, roi_info: dict[str, Any], context: dict[str, Any]) -> list[ValidationIssue]:
        """Validate a specific ROI mention."""
        issues = []
        roi = roi_info["roi"]
        roi_percent = roi_info["roi_percent"]

        # Determine investment type from context
        investment_type = self._determine_investment_type(context)
        roi_range = self.roi_ranges.get(investment_type, self.roi_ranges["rental_property"])

        # Check if ROI is within reasonable range
        if roi < roi_range["min"]:
            severity = "high" if roi < roi_range["min"] - 0.1 else "medium"
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity=severity,
                    description=f"ROI of {roi_percent:.1f}% is unusually low for {investment_type}",
                    value=f"{roi_percent:.1f}%",
                    suggested_correction=f"Typical range: {roi_range['min'] * 100:.1f}% to {roi_range['max'] * 100:.1f}%",
                    confidence=0.8,
                )
            )
        elif roi > roi_range["max"]:
            severity = "high" if roi > roi_range["max"] + 0.2 else "medium"
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity=severity,
                    description=f"ROI of {roi_percent:.1f}% is unusually high for {investment_type}",
                    value=f"{roi_percent:.1f}%",
                    suggested_correction=f"Typical range: {roi_range['min'] * 100:.1f}% to {roi_range['max'] * 100:.1f}%",
                    confidence=0.8,
                )
            )

        # Check for unrealistic projections
        if roi > 1.0:  # 100% ROI
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity="critical",
                    description=f"ROI of {roi_percent:.1f}% is extremely high and likely unrealistic",
                    value=f"{roi_percent:.1f}%",
                    suggested_correction="Verify calculation methodology and assumptions",
                    confidence=0.95,
                )
            )

        return issues

    async def _validate_cap_rate(self, cap_rate_info: dict[str, Any], context: dict[str, Any]) -> list[ValidationIssue]:
        """Validate a specific cap rate mention."""
        issues = []
        cap_rate = cap_rate_info["cap_rate"]
        cap_rate_percent = cap_rate_info["cap_rate_percent"]

        # Determine property type from context
        property_type = context.get("property_type", "single_family").lower()
        property_type = self._normalize_property_type_for_cap_rate(property_type)

        cap_rate_range = self.cap_rate_ranges.get(property_type, self.cap_rate_ranges["single_family"])

        # Check if cap rate is within reasonable range
        if cap_rate < cap_rate_range["min"]:
            severity = "medium" if cap_rate > 0.02 else "high"
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity=severity,
                    description=f"Cap rate of {cap_rate_percent:.1f}% is low for {property_type}",
                    value=f"{cap_rate_percent:.1f}%",
                    suggested_correction=f"Typical range: {
                        cap_rate_range['min'] * 100:.1f}% to {cap_rate_range['max'] * 100:.1f}%",
                    confidence=0.7,
                )
            )
        elif cap_rate > cap_rate_range["max"]:
            severity = "medium" if cap_rate < 0.20 else "high"
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity=severity,
                    description=f"Cap rate of {cap_rate_percent:.1f}% is high for {property_type}",
                    value=f"{cap_rate_percent:.1f}%",
                    suggested_correction=f"Typical range: {
                        cap_rate_range['min'] * 100:.1f}% to {cap_rate_range['max'] * 100:.1f}%",
                    confidence=0.7,
                )
            )

        # Check for unrealistic cap rates
        if cap_rate <= 0:
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity="critical",
                    description=f"Cap rate of {cap_rate_percent:.1f}% is zero or negative, which is unrealistic",
                    value=f"{cap_rate_percent:.1f}%",
                    suggested_correction="Cap rate should be positive for income-producing properties",
                    confidence=0.95,
                )
            )
        elif cap_rate > 0.25:  # 25%
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity="high",
                    description=f"Cap rate of {cap_rate_percent:.1f}% is extremely high and may indicate high risk",
                    value=f"{cap_rate_percent:.1f}%",
                    suggested_correction="Verify NOI calculation and property condition",
                    confidence=0.8,
                )
            )

        return issues

    async def _validate_cash_on_cash(self, cash_info: dict[str, Any], context: dict[str, Any]) -> list[ValidationIssue]:
        """Validate cash-on-cash return mention."""
        issues = []
        cash_on_cash = cash_info["cash_on_cash"]
        cash_percent = cash_info["cash_on_cash_percent"]

        # Cash-on-cash can be negative but should be reasonable
        if cash_on_cash < -0.20:  # -20%
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity="medium",
                    description=f"Cash-on-cash return of {cash_percent:.1f}% is very negative",
                    value=f"{cash_percent:.1f}%",
                    suggested_correction="Consider if negative cash flow is sustainable",
                    confidence=0.7,
                )
            )
        elif cash_on_cash > 0.30:  # 30%
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity="medium",
                    description=f"Cash-on-cash return of {cash_percent:.1f}% is unusually high",
                    value=f"{cash_percent:.1f}%",
                    suggested_correction="Verify cash investment and cash flow calculations",
                    confidence=0.7,
                )
            )

        # Extremely high returns are suspicious
        if cash_on_cash > 0.50:  # 50%
            issues.append(
                self.create_issue(
                    issue_type="roi",
                    severity="high",
                    description=f"Cash-on-cash return of {cash_percent:.1f}% is extremely high",
                    value=f"{cash_percent:.1f}%",
                    suggested_correction="Double-check calculation methodology",
                    confidence=0.8,
                )
            )

        return issues

    def _check_financial_consistency(
        self,
        roi_mentions: list[dict[str, Any]],
        cap_rate_mentions: list[dict[str, Any]],
        cash_on_cash_mentions: list[dict[str, Any]],
        response: str,
    ) -> list[ValidationIssue]:
        """Check for consistency between different financial metrics."""
        issues = []

        # If both cap rate and cash-on-cash are mentioned, they should be related
        if cap_rate_mentions and cash_on_cash_mentions:
            cap_rate = cap_rate_mentions[0]["cap_rate"]
            cash_on_cash = cash_on_cash_mentions[0]["cash_on_cash"]

            # Cash-on-cash is typically lower than cap rate for leveraged properties
            # but can be higher for all-cash purchases
            if cash_on_cash > cap_rate * 2:  # More than double
                issues.append(
                    self.create_issue(
                        issue_type="roi",
                        severity="medium",
                        description=f"Cash-on-cash return ({cash_on_cash * 100:.1f}%) is much higher than cap rate ({cap_rate * 100:.1f}%)",
                        suggested_correction="Verify if property is purchased with cash or check calculations",
                        confidence=0.6,
                    )
                )

        # Check for time period consistency in ROI mentions
        if len(roi_mentions) > 1:
            roi_values = [roi["roi"] for roi in roi_mentions]
            if max(roi_values) / min(roi_values) > 3:  # 3x difference
                issues.append(
                    self.create_issue(
                        issue_type="roi",
                        severity="medium",
                        description="Multiple ROI figures mentioned with significant differences",
                        suggested_correction="Clarify time periods and calculation methods for different ROI figures",
                        confidence=0.7,
                    )
                )

        return issues

    def _determine_investment_type(self, context: dict[str, Any]) -> str:
        """Determine investment type from context."""
        query = context.get("original_query", "").lower()
        user_role = context.get("user_role", "").lower()

        if "flip" in query or "renovation" in query:
            return "fix_and_flip"
        elif "commercial" in query or user_role == "developer":
            return "commercial"
        elif "reit" in query or "trust" in query:
            return "reit"
        elif "land" in query or "lot" in query:
            return "land"
        else:
            return "rental_property"

    def _normalize_property_type_for_cap_rate(self, property_type: str) -> str:
        """Normalize property type for cap rate validation."""
        if "single" in property_type or "sfr" in property_type:
            return "single_family"
        elif "multi" in property_type or "apartment" in property_type:
            return "multi_family"
        elif "office" in property_type:
            return "office"
        elif "retail" in property_type or "shopping" in property_type:
            return "retail"
        elif "commercial" in property_type or "industrial" in property_type:
            return "commercial"
        else:
            return "single_family"

    def _calculate_confidence_score(self, issues: list[ValidationIssue]) -> float:
        """Calculate confidence score based on issues found."""
        if not issues:
            return 1.0

        # Weight issues by severity
        severity_weights = {"critical": 0.5, "high": 0.3, "medium": 0.2, "low": 0.1}
        total_weight = sum(severity_weights[issue.severity] for issue in issues)

        # Confidence decreases with more severe issues
        confidence = max(0.1, 1.0 - (total_weight / 3.0))
        return confidence
