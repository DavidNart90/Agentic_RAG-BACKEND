"""Price validation for real estate responses."""

import re
from typing import Any

from ..models.agent import ValidationIssue, ValidationResult
from .base import ResponseValidator


class PriceValidator(ResponseValidator):
    """Validates price information for realism and consistency."""

    def __init__(self):
        super().__init__("price_range_validator", "1.0")

        # Price ranges by property type (per sq ft)
        self.price_ranges = {
            "single_family": {"min": 50, "max": 800},
            "condo": {"min": 100, "max": 1200},
            "townhouse": {"min": 80, "max": 600},
            "multi_family": {"min": 60, "max": 400},
            "commercial": {"min": 100, "max": 2000},
            "land": {"min": 1, "max": 100},
        }

        # Regional price multipliers (rough estimates)
        self.regional_multipliers = {
            "san_francisco": 3.0,
            "new_york": 2.5,
            "los_angeles": 2.2,
            "seattle": 1.8,
            "boston": 1.7,
            "washington_dc": 1.6,
            "miami": 1.4,
            "chicago": 1.2,
            "denver": 1.1,
            "atlanta": 1.0,
            "phoenix": 0.9,
            "dallas": 0.8,
            "houston": 0.7,
            "kansas_city": 0.6,
            "cleveland": 0.5,
        }

    async def validate(self, response: str, context: dict[str, Any]) -> ValidationResult:
        """Validate price information in the response."""
        issues = []

        # Extract price mentions from response
        price_mentions = self._extract_price_mentions(response)

        if not price_mentions:
            # No price mentions found - not necessarily an issue
            return self.create_result(is_valid=True, confidence_score=1.0, issues=[])

        # Get context information
        location = context.get("location_context", "").lower()
        property_type = context.get("property_type", "single_family").lower()
        square_footage = context.get("square_footage")

        # Validate each price mention
        for price_info in price_mentions:
            price_issues = await self._validate_price(price_info, location, property_type, square_footage, context)
            issues.extend(price_issues)

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

    def _extract_price_mentions(self, text: str) -> list[dict[str, Any]]:
        """Extract price mentions from text."""
        price_mentions = []

        # Patterns for different price formats
        patterns = [
            r"\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:per\s+(?:sq\s*ft|square\s+foot))?",
            r"\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|M)",
            r"\$([0-9,]+(?:\.[0-9]{2})?)\s*(?:thousand|K)",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*dollars?\s*(?:per\s+(?:sq\s*ft|square\s+foot))?",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                price_str = match.group(1).replace(",", "")

                try:
                    price = float(price_str)

                    # Handle millions and thousands
                    if "million" in match.group(0).lower() or "M" in match.group(0):
                        price *= 1_000_000
                    elif "thousand" in match.group(0).lower() or "K" in match.group(0):
                        price *= 1_000

                    price_mentions.append(
                        {
                            "price": price,
                            "text": match.group(0),
                            "position": match.span(),
                            "is_per_sqft": "per sq" in match.group(0).lower() or "square foot" in match.group(0).lower(),
                        }
                    )
                except ValueError:
                    continue

        return price_mentions

    async def _validate_price(
        self,
        price_info: dict[str, Any],
        location: str,
        property_type: str,
        square_footage: int | None,
        context: dict[str, Any],
    ) -> list[ValidationIssue]:
        """Validate a specific price mention."""
        issues = []
        price = price_info["price"]
        is_per_sqft = price_info["is_per_sqft"]

        # Get regional multiplier
        regional_multiplier = self._get_regional_multiplier(location)

        # Get property type ranges
        prop_type_key = self._normalize_property_type(property_type)
        price_range = self.price_ranges.get(prop_type_key, self.price_ranges["single_family"])

        if is_per_sqft:
            # Validate price per square foot
            adjusted_min = price_range["min"] * regional_multiplier
            adjusted_max = price_range["max"] * regional_multiplier

            if price < adjusted_min * 0.5:  # 50% below minimum
                issues.append(
                    self.create_issue(
                        issue_type="price",
                        severity="high",
                        description=f"Price per sq ft ${price:.0f} is unusually low for {property_type} in {location}",
                        value=str(price),
                        suggested_correction=f"Typical range: ${adjusted_min:.0f}-${adjusted_max:.0f} per sq ft",
                        confidence=0.8,
                    )
                )
            elif price > adjusted_max * 2:  # 200% above maximum
                issues.append(
                    self.create_issue(
                        issue_type="price",
                        severity="high",
                        description=f"Price per sq ft ${price:.0f} is unusually high for {property_type} in {location}",
                        value=str(price),
                        suggested_correction=f"Typical range: ${adjusted_min:.0f}-${adjusted_max:.0f} per sq ft",
                        confidence=0.8,
                    )
                )
        else:
            # Validate total price
            if square_footage:
                implied_price_per_sqft = price / square_footage
                adjusted_min = price_range["min"] * regional_multiplier
                adjusted_max = price_range["max"] * regional_multiplier

                if implied_price_per_sqft < adjusted_min * 0.3:
                    issues.append(
                        self.create_issue(
                            issue_type="price",
                            severity="medium",
                            description=f"Total price ${price:,.0f} implies ${implied_price_per_sqft:.0f}/sq ft, which is low for {property_type}",
                            value=str(price),
                            suggested_correction=f"Expected range: ${adjusted_min * square_footage:,.0f}-${adjusted_max * square_footage:,.0f}",
                            confidence=0.7,
                        )
                    )
                elif implied_price_per_sqft > adjusted_max * 3:
                    issues.append(
                        self.create_issue(
                            issue_type="price",
                            severity="medium",
                            description=f"Total price ${price:,.0f} implies ${implied_price_per_sqft:.0f}/sq ft, which is high for {property_type}",
                            value=str(price),
                            suggested_correction=f"Expected range: ${adjusted_min * square_footage:,.0f}-${adjusted_max * square_footage:,.0f}",
                            confidence=0.7,
                        )
                    )
            else:
                # General price reasonableness check without square footage
                if price < 10_000:
                    issues.append(
                        self.create_issue(
                            issue_type="price",
                            severity="high",
                            description=f"Property price ${price:,.0f} is extremely low",
                            value=str(price),
                            suggested_correction="Verify price is not missing zeros or in wrong units",
                            confidence=0.9,
                        )
                    )
                elif price > 50_000_000:
                    issues.append(
                        self.create_issue(
                            issue_type="price",
                            severity="medium",
                            description=f"Property price ${price:,.0f} is extremely high",
                            value=str(price),
                            suggested_correction="Verify price is not inflated or in wrong units",
                            confidence=0.7,
                        )
                    )

        return issues

    def _get_regional_multiplier(self, location: str) -> float:
        """Get regional price multiplier based on location."""
        location_lower = location.lower()

        # Check for major cities
        for city, multiplier in self.regional_multipliers.items():
            if city.replace("_", " ") in location_lower:
                return multiplier

        # Check for high-cost states
        high_cost_states = ["california", "new york", "massachusetts", "washington", "hawaii"]
        for state in high_cost_states:
            if state in location_lower:
                return 1.5

        # Check for low-cost states
        low_cost_states = ["mississippi", "arkansas", "oklahoma", "kansas", "alabama"]
        for state in low_cost_states:
            if state in location_lower:
                return 0.7

        # Default multiplier
        return 1.0

    def _normalize_property_type(self, property_type: str) -> str:
        """Normalize property type to match our ranges."""
        prop_type_lower = property_type.lower()

        if "single" in prop_type_lower or "sfr" in prop_type_lower:
            return "single_family"
        elif "condo" in prop_type_lower or "condominium" in prop_type_lower:
            return "condo"
        elif "town" in prop_type_lower:
            return "townhouse"
        elif "multi" in prop_type_lower or "duplex" in prop_type_lower or "triplex" in prop_type_lower:
            return "multi_family"
        elif "commercial" in prop_type_lower or "office" in prop_type_lower or "retail" in prop_type_lower:
            return "commercial"
        elif "land" in prop_type_lower or "lot" in prop_type_lower:
            return "land"
        else:
            return "single_family"  # Default

    def _calculate_confidence_score(self, issues: list[ValidationIssue]) -> float:
        """Calculate confidence score based on issues found."""
        if not issues:
            return 1.0

        # Weight issues by severity
        severity_weights = {"critical": 0.4, "high": 0.3, "medium": 0.2, "low": 0.1}
        total_weight = sum(severity_weights[issue.severity] for issue in issues)

        # Confidence decreases with more severe issues
        confidence = max(0.1, 1.0 - (total_weight / 2.0))
        return confidence


class PriceConsistencyValidator(ResponseValidator):
    """Validates price consistency within a response."""

    def __init__(self):
        super().__init__("price_consistency_validator", "1.0")

    async def validate(self, response: str, context: dict[str, Any]) -> ValidationResult:
        """Validate price consistency in the response."""
        issues = []

        # Extract all price mentions
        price_mentions = self._extract_all_prices(response)

        if len(price_mentions) < 2:
            # Need at least 2 prices to check consistency
            return self.create_result(is_valid=True, confidence_score=1.0, issues=[])

        # Check for inconsistencies
        consistency_issues = self._check_price_consistency(price_mentions)
        issues.extend(consistency_issues)

        # Check for calculation errors
        calculation_issues = self._check_price_calculations(response, price_mentions)
        issues.extend(calculation_issues)

        # Determine validity
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        is_valid = len(critical_issues) == 0

        confidence_score = 1.0 - (len(issues) * 0.2)  # Decrease confidence with more issues
        confidence_score = max(0.1, confidence_score)

        return self.create_result(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            correction_needed=len(critical_issues) > 0,
        )

    def _extract_all_prices(self, text: str) -> list[dict[str, Any]]:
        """Extract all price mentions from text."""
        # This would be similar to the price extraction in PriceRangeValidator
        # but more comprehensive to catch all price formats
        price_mentions = []

        patterns = [
            r"\$([0-9,]+(?:\.[0-9]{2})?)",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*dollars?",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*(?:million|M)",
            r"([0-9,]+(?:\.[0-9]{2})?)\s*(?:thousand|K)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price_str = match.group(1).replace(",", "")
                    price = float(price_str)

                    # Handle units
                    full_match = match.group(0).lower()
                    if "million" in full_match or "m" in full_match:
                        price *= 1_000_000
                    elif "thousand" in full_match or "k" in full_match:
                        price *= 1_000

                    price_mentions.append({"price": price, "text": match.group(0), "position": match.span()})
                except ValueError:
                    continue

        return price_mentions

    def _check_price_consistency(self, price_mentions: list[dict[str, Any]]) -> list[ValidationIssue]:
        """Check for price consistency issues."""
        issues = []

        # Group similar prices (within 10% of each other)
        price_groups = []
        for price_info in price_mentions:
            price = price_info["price"]

            # Find existing group or create new one
            found_group = False
            for group in price_groups:
                group_avg = sum(p["price"] for p in group) / len(group)
                if abs(price - group_avg) / group_avg <= 0.1:  # Within 10%
                    group.append(price_info)
                    found_group = True
                    break

            if not found_group:
                price_groups.append([price_info])

        # Check for conflicting price ranges
        if len(price_groups) > 1:
            prices = [group[0]["price"] for group in price_groups]
            min_price = min(prices)
            max_price = max(prices)

            # If prices differ by more than 50%, flag as inconsistent
            if max_price / min_price > 1.5:
                issues.append(
                    self.create_issue(
                        issue_type="price",
                        severity="medium",
                        description=f"Inconsistent price mentions: ${min_price:,.0f} to ${max_price:,.0f}",
                        suggested_correction="Verify all price mentions refer to the same property or clarify context",
                        confidence=0.7,
                    )
                )

        return issues

    def _check_price_calculations(self, response: str, price_mentions: list[dict[str, Any]]) -> list[ValidationIssue]:
        """Check for price calculation errors."""
        issues = []

        # Look for price per square foot calculations
        sqft_pattern = r"([0-9,]+)\s*(?:sq\s*ft|square\s+feet?)"
        sqft_matches = re.findall(sqft_pattern, response, re.IGNORECASE)

        if sqft_matches and len(price_mentions) >= 2:
            try:
                sqft = float(sqft_matches[0].replace(",", ""))

                # Find total price and price per sqft
                total_prices = [p for p in price_mentions if p["price"] > 1000]  # Likely total prices
                per_sqft_prices = [p for p in price_mentions if p["price"] < 1000]  # Likely per sqft prices

                if total_prices and per_sqft_prices:
                    total_price = total_prices[0]["price"]
                    per_sqft_price = per_sqft_prices[0]["price"]

                    calculated_total = per_sqft_price * sqft

                    # Check if calculation is consistent (within 5%)
                    if abs(calculated_total - total_price) / total_price > 0.05:
                        issues.append(
                            self.create_issue(
                                issue_type="price",
                                severity="high",
                                description=f"Price calculation error: ${per_sqft_price}/sq ft × {sqft:,.0f} sq ft = ${calculated_total:,.0f}, but total price stated as ${total_price:,.0f}",
                                suggested_correction=f"Correct calculation should be ${calculated_total:,.0f}",
                                confidence=0.9,
                            )
                        )
            except (ValueError, IndexError):
                pass

        return issues
