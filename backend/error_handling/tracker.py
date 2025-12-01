"""
Error Tracking and Reporting System

Tracks errors, aggregates metrics, and provides reporting capabilities.
"""

import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock
import asyncio

from .exceptions import AutoRLBaseException, ErrorCategory, ErrorSeverity


logger = logging.getLogger(__name__)


class ErrorTracker:
    """
    Centralized error tracking system with metrics and reporting.
    
    Features:
    - Error aggregation and categorization
    - Time-series error tracking
    - Error rate calculation
    - Alert threshold monitoring
    - Persistent error logging
    """
    
    def __init__(
        self,
        log_dir: str = "./logs/errors",
        max_memory_errors: int = 1000,
        alert_threshold: int = 10,
        alert_window_seconds: int = 60
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_errors = max_memory_errors
        self.alert_threshold = alert_threshold
        self.alert_window_seconds = alert_window_seconds
        
        # In-memory error storage
        self._errors = deque(maxlen=max_memory_errors)
        self._error_counts = defaultdict(int)
        self._error_by_category = defaultdict(list)
        self._error_by_severity = defaultdict(list)
        self._recent_errors = deque(maxlen=100)
        
        # Thread-safe access
        self._lock = Lock()
        
        # Alert tracking
        self._alert_state = defaultdict(lambda: {"count": 0, "last_alert": 0})
        
        logger.info(f"ErrorTracker initialized. Log directory: {self.log_dir}")
    
    def track_error(
        self,
        error: AutoRLBaseException,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track an error with full context.
        
        Args:
            error: The exception to track
            additional_context: Additional context information
            
        Returns:
            Error ID for tracking
        """
        with self._lock:
            # Generate error ID if not present
            if not error.context.error_id:
                error.context.error_id = self._generate_error_id(error)
            
            # Add additional context
            if additional_context:
                error.context.additional_data.update(additional_context)
            
            # Store error
            error_data = self._serialize_error(error)
            self._errors.append(error_data)
            self._recent_errors.append(error_data)
            
            # Update counters
            self._error_counts[error.error_code] += 1
            self._error_by_category[error.category.value].append(error_data)
            self._error_by_severity[error.severity.value].append(error_data)
            
            # Check for alerts
            self._check_alert_threshold(error)
            
            # Persist to disk
            self._persist_error(error_data)
            
            logger.debug(f"Tracked error: {error.error_code} ({error.context.error_id})")
            
            return error.context.error_id
    
    def _generate_error_id(self, error: AutoRLBaseException) -> str:
        """Generate unique error ID"""
        timestamp = int(time.time() * 1000)
        return f"{error.error_code}_{timestamp}"
    
    def _serialize_error(self, error: AutoRLBaseException) -> Dict[str, Any]:
        """Serialize error to dictionary"""
        return {
            "error_id": error.context.error_id,
            "error_code": error.error_code,
            "message": error.message,
            "category": error.category.value,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "retry_after": error.retry_after,
            "timestamp": error.context.timestamp,
            "context": error.context.to_dict(),
            "recovery_suggestions": error.recovery_suggestions,
            "original_error": str(error.original_exception) if error.original_exception else None
        }
    
    def _persist_error(self, error_data: Dict[str, Any]):
        """Persist error to disk"""
        try:
            # Create date-based log file
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"errors_{date_str}.jsonl"
            
            with open(log_file, 'a') as f:
                json.dump(error_data, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to persist error to disk: {e}")
    
    def _check_alert_threshold(self, error: AutoRLBaseException):
        """Check if error rate exceeds alert threshold"""
        current_time = time.time()
        error_code = error.error_code
        
        alert_data = self._alert_state[error_code]
        
        # Reset count if outside window
        if current_time - alert_data["last_alert"] > self.alert_window_seconds:
            alert_data["count"] = 0
        
        alert_data["count"] += 1
        
        # Trigger alert if threshold exceeded
        if alert_data["count"] >= self.alert_threshold:
            if current_time - alert_data["last_alert"] > self.alert_window_seconds:
                self._trigger_alert(error, alert_data["count"])
                alert_data["last_alert"] = current_time
    
    def _trigger_alert(self, error: AutoRLBaseException, count: int):
        """Trigger alert for high error rate"""
        logger.warning(
            f"ALERT: High error rate detected! "
            f"{error.error_code} occurred {count} times in {self.alert_window_seconds}s"
        )
        
        # Save alert to separate file
        alert_file = self.log_dir / "alerts.jsonl"
        alert_data = {
            "timestamp": time.time(),
            "error_code": error.error_code,
            "count": count,
            "window_seconds": self.alert_window_seconds,
            "severity": error.severity.value,
            "category": error.category.value
        }
        
        try:
            with open(alert_file, 'a') as f:
                json.dump(alert_data, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors"""
        with self._lock:
            return {
                "total_errors": len(self._errors),
                "error_counts": dict(self._error_counts),
                "by_category": {
                    cat: len(errors) for cat, errors in self._error_by_category.items()
                },
                "by_severity": {
                    sev: len(errors) for sev, errors in self._error_by_severity.items()
                },
                "recent_errors": list(self._recent_errors)[-10:],
                "timestamp": time.time()
            }
    
    def get_error_rate(self, time_window_seconds: int = 60) -> Dict[str, float]:
        """Calculate error rate for each error code"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - time_window_seconds
            
            recent_errors = defaultdict(int)
            for error_data in self._errors:
                if error_data["timestamp"] >= cutoff_time:
                    recent_errors[error_data["error_code"]] += 1
            
            # Calculate rate per minute
            rates = {
                code: (count / time_window_seconds) * 60
                for code, count in recent_errors.items()
            }
            
            return rates
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[Dict[str, Any]]:
        """Get errors by category"""
        with self._lock:
            return list(self._error_by_category.get(category.value, []))
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[Dict[str, Any]]:
        """Get errors by severity"""
        with self._lock:
            return list(self._error_by_severity.get(severity.value, []))
    
    def get_critical_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent critical errors"""
        critical_errors = self.get_errors_by_severity(ErrorSeverity.CRITICAL)
        return sorted(
            critical_errors,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_error_trends(self, hours: int = 24) -> Dict[str, List[int]]:
        """Get error trends over time"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (hours * 3600)
            
            # Create hourly buckets
            buckets = defaultdict(lambda: defaultdict(int))
            
            for error_data in self._errors:
                if error_data["timestamp"] >= cutoff_time:
                    hour_bucket = int((error_data["timestamp"] - cutoff_time) / 3600)
                    error_code = error_data["error_code"]
                    buckets[error_code][hour_bucket] += 1
            
            # Fill in missing buckets with zeros
            trends = {}
            for error_code, hour_counts in buckets.items():
                trends[error_code] = [
                    hour_counts.get(i, 0) for i in range(hours)
                ]
            
            return trends
    
    def clear_old_errors(self, days: int = 7):
        """Clear errors older than specified days"""
        with self._lock:
            cutoff_time = time.time() - (days * 86400)
            
            # Filter in-memory errors
            self._errors = deque(
                [e for e in self._errors if e["timestamp"] >= cutoff_time],
                maxlen=self.max_memory_errors
            )
            
            logger.info(f"Cleared errors older than {days} days")
    
    def export_errors(
        self,
        output_file: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> str:
        """Export errors to JSON file"""
        with self._lock:
            errors_to_export = list(self._errors)
            
            # Filter by time range
            if start_time or end_time:
                errors_to_export = [
                    e for e in errors_to_export
                    if (not start_time or e["timestamp"] >= start_time) and
                       (not end_time or e["timestamp"] <= end_time)
                ]
            
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(self.log_dir / f"error_export_{timestamp}.json")
            
            # Export to file
            with open(output_file, 'w') as f:
                json.dump({
                    "export_time": time.time(),
                    "error_count": len(errors_to_export),
                    "start_time": start_time,
                    "end_time": end_time,
                    "errors": errors_to_export
                }, f, indent=2)
            
            logger.info(f"Exported {len(errors_to_export)} errors to {output_file}")
            
            return output_file


# Global error tracker instance
_global_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ErrorTracker()
    return _global_tracker


def track_error(
    error: AutoRLBaseException,
    additional_context: Optional[Dict[str, Any]] = None
) -> str:
    """Track error using global tracker"""
    return get_error_tracker().track_error(error, additional_context)

