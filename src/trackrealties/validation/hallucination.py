import re
from typing import Any, Dict, List

from ..models.agent import ValidationIssue, ValidationResult
from .base import ResponseValidator


class RealEstateHallucinationDetector(ResponseValidator):
    """Simple hallucination detector for real estate responses."""

    def __init__(self) -> None:
        super().__init__("real_estate_hallucination_detector", "1.0")

    async def validate(self, response: str, context: Dict[str, Any]) -> ValidationResult:
        issues: List[ValidationIssue] = []

        # Detect unrealistic percentages (greater than 100%)
        for match in re.findall(r"([0-9]+(?:\.[0-9]+)?)%", response):
            try:
                pct = float(match)
                if pct > 100:
                    issues.append(
                        self.create_issue(
                            issue_type="factual",
                            severity="high",
                            description=f"Unrealistic percentage {pct}%",
                            value=f"{pct}%",
                            confidence=0.8,
                        )
                    )
            except ValueError:
                continue

        # Detect extremely large price figures (>$100M)
        for match in re.findall(r"\$([0-9][0-9,]*)", response):
            try:
                price = float(match.replace(",", ""))
                if price > 100_000_000:
                    issues.append(
                        self.create_issue(
                            issue_type="factual",
                            severity="high",
                            description=f"Price ${price:,.0f} seems unrealistic",
                            value=f"${match}",
                            confidence=0.8,
                        )
                    )
            except ValueError:
                continue

        is_valid = len(issues) == 0
        confidence = max(0.1, 1.0 - 0.1 * len(issues))

        return self.create_result(
            is_valid=is_valid,
            confidence_score=confidence,
            issues=issues,
            correction_needed=not is_valid,
        )
