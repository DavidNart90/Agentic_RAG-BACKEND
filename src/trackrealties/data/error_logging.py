"""
Error Logging and Validation Tracking Module

This module provides utilities for logging errors and tracking validation issues
during data ingestion and transformation.
"""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an error with context information.

    Args:
        error: Exception that occurred
        context: Additional context information
    """
    # Format the context information for logging
    context_str = ""
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())

    # Log the error with context information
    logger.error(f"Error: {str(error)} [{context_str}]", exc_info=True)

    # Create error log directory if it doesn't exist
    log_dir = "logs/errors"
    os.makedirs(log_dir, exist_ok=True)

    try:
        # Generate unique error ID
        error_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Create error info
        error_info = {
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }

        # Create log file path
        log_file = f"{log_dir}/{error_id}.json"

        # Write error info to file
        with open(log_file, "w") as f:
            json.dump(error_info, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to write error log file: {e}")


class ErrorLogger:
    """
    Logger for detailed error information during data processing.
    """

    def __init__(self, log_dir: str = "logs/errors"):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    async def log_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """
        Log detailed error information.

        Args:
            error: Exception that occurred
            context: Additional context information

        Returns:
            Path to error log file
        """
        try:
            # Generate unique error ID
            error_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Create error info
            error_info = {
                "error_id": error_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context,
            }

            # Create log file path
            log_file = f"{self.log_dir}/{error_id}.json"

            # Write error info to file
            with open(log_file, "w") as f:
                json.dump(error_info, f, indent=2)

            self.logger.info(f"Error logged to {log_file}")

            return log_file

        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")
            return ""

    async def log_validation_error(
        self, record_id: str, errors: List[str], record_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log validation error information.

        Args:
            record_id: ID of the record that failed validation
            errors: List of validation errors
            record_data: Optional record data (sensitive fields will be redacted)

        Returns:
            Path to validation error log file
        """
        try:
            # Generate unique error ID
            error_id = f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Redact sensitive fields if record data is provided
            redacted_data = None
            if record_data:
                redacted_data = self._redact_sensitive_fields(record_data)

            # Create error info
            error_info = {
                "error_id": error_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "record_id": record_id,
                "errors": errors,
                "record_data": redacted_data,
            }

            # Create log file path
            log_file = f"{self.log_dir}/{error_id}.json"

            # Write error info to file
            with open(log_file, "w") as f:
                json.dump(error_info, f, indent=2)

            self.logger.info(f"Validation error logged to {log_file}")

            return log_file

        except Exception as e:
            self.logger.error(f"Failed to log validation error: {e}")
            return ""

    def _redact_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive fields from record data.

        Args:
            data: Record data

        Returns:
            Redacted record data
        """
        # Create a copy of the data
        redacted = data.copy()

        # List of sensitive fields to redact
        sensitive_fields = [
            "email",
            "phone",
            "password",
            "ssn",
            "social_security",
            "credit_card",
            "bank_account",
            "routing_number",
        ]

        # Redact sensitive fields
        for field in sensitive_fields:
            if field in redacted:
                redacted[field] = "[REDACTED]"

        # Check nested dictionaries
        for key, value in redacted.items():
            if isinstance(value, dict):
                redacted[key] = self._redact_sensitive_fields(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self._redact_sensitive_fields(item) if isinstance(item, dict) else item for item in value
                ]

        return redacted


class ValidationTracker:
    """
    Tracker for validation issues during data processing.
    """

    def __init__(self, log_dir: str = "logs/validation"):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    async def track_validation_issues(self, job_id: str, validation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track and analyze validation issues.

        Args:
            job_id: ID of the ingestion job
            validation_stats: Validation statistics

        Returns:
            Validation analysis report
        """
        try:
            # Calculate validation metrics
            total_records = validation_stats.get("total_records", 0)
            if total_records == 0:
                return {"error": "No records processed"}

            valid_records = validation_stats.get("valid_records", 0)
            invalid_records = validation_stats.get("invalid_records", 0)

            validation_rate = valid_records / total_records if total_records > 0 else 0
            error_rate = invalid_records / total_records if total_records > 0 else 0

            # Analyze common errors
            errors = validation_stats.get("errors", [])
            error_categories = self._categorize_errors(errors)

            # Create report
            report = {
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_records": total_records,
                "valid_records": valid_records,
                "invalid_records": invalid_records,
                "validation_rate": validation_rate,
                "error_rate": error_rate,
                "error_categories": error_categories,
                "error_samples": errors[:10],  # Include up to 10 sample errors
            }

            # Log report
            await self._log_validation_report(job_id, report)

            return report

        except Exception as e:
            self.logger.error(f"Failed to track validation issues: {e}")
            return {"error": str(e)}

    def _categorize_errors(self, errors: List[str]) -> Dict[str, int]:
        """
        Categorize errors by type.

        Args:
            errors: List of error messages

        Returns:
            Dictionary of error categories and counts
        """
        categories = {}

        for error in errors:
            # Extract error category
            if "missing required field" in error.lower():
                category = "Missing Required Field"
            elif "invalid format" in error.lower():
                category = "Invalid Format"
            elif "validation error" in error.lower():
                category = "Validation Error"
            elif "type error" in error.lower():
                category = "Type Error"
            else:
                category = "Other Error"

            categories[category] = categories.get(category, 0) + 1

        return categories

    async def _log_validation_report(self, job_id: str, report: Dict[str, Any]) -> str:
        """
        Log validation report to file.

        Args:
            job_id: ID of the ingestion job
            report: Validation report

        Returns:
            Path to validation report file
        """
        try:
            # Create log file path
            log_file = f"{self.log_dir}/{job_id}_validation.json"

            # Write report to file
            with open(log_file, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Validation report logged to {log_file}")

            return log_file

        except Exception as e:
            self.logger.error(f"Failed to log validation report: {e}")
            return ""


class DataQualityMonitor:
    """
    Monitor for data quality issues during data processing.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            "validation_failure_rate": 0.05,  # 5% max failure rate
            "missing_required_fields": 0.02,  # 2% max missing required fields
            "duplicate_records": 0.01,  # 1% max duplicate records
        }

    async def monitor_data_quality(self, job_id: str, validation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor data quality for an ingestion job.

        Args:
            job_id: ID of the ingestion job
            validation_stats: Validation statistics

        Returns:
            Data quality report
        """
        try:
            # Calculate quality metrics
            total_records = validation_stats.get("total_records", 0)
            if total_records == 0:
                return {"error": "No records processed"}

            valid_records = validation_stats.get("valid_records", 0)
            invalid_records = validation_stats.get("invalid_records", 0)

            validation_failure_rate = invalid_records / total_records if total_records > 0 else 0
            processing_success_rate = valid_records / total_records if total_records > 0 else 0

            # Check for quality issues
            issues = []
            recommendations = []

            # Check validation failure rate
            if validation_failure_rate > self.quality_thresholds["validation_failure_rate"]:
                issues.append(f"High validation failure rate: {validation_failure_rate:.2%}")
                recommendations.append("Review data source quality and validation rules")

            # Calculate overall quality score
            quality_score = processing_success_rate * 100
            if issues:
                quality_score -= len(issues) * 10

            quality_score = max(0, min(100, quality_score))

            # Create report
            report = {
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "quality_score": quality_score,
                "validation_failure_rate": validation_failure_rate,
                "processing_success_rate": processing_success_rate,
                "issues": issues,
                "recommendations": recommendations,
            }

            # Generate alert if needed
            if quality_score < 80:
                await self._generate_quality_alert(job_id, report)

            return report

        except Exception as e:
            self.logger.error(f"Failed to monitor data quality: {e}")
            return {"error": str(e)}

    async def _generate_quality_alert(self, job_id: str, quality_report: Dict[str, Any]) -> None:
        """
        Generate quality alert if thresholds are exceeded.

        Args:
            job_id: ID of the ingestion job
            quality_report: Data quality report
        """
        try:
            alert = {
                "alert_type": "data_quality",
                "severity": "warning" if quality_report["quality_score"] > 60 else "critical",
                "job_id": job_id,
                "quality_score": quality_report["quality_score"],
                "issues": quality_report["issues"],
                "recommendations": quality_report["recommendations"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Log alert
            self.logger.warning(f"Data quality alert: {json.dumps(alert)}")

            # TODO: Send alert to notification system

        except Exception as e:
            self.logger.error(f"Failed to generate quality alert: {e}")
