"""
Comprehensive Exception Hierarchy for AutoRL Application

This module defines a structured exception hierarchy with detailed error codes,
context information, and recovery strategies.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import traceback
import time


class ErrorCategory(Enum):
    """High-level error categories for classification"""
    INFRASTRUCTURE = "infrastructure"  # System, network, resource issues
    VALIDATION = "validation"  # Input validation failures
    BUSINESS_LOGIC = "business_logic"  # Application logic errors
    EXTERNAL_SERVICE = "external_service"  # Third-party API failures
    SECURITY = "security"  # Authentication, authorization, security issues
    DATA = "data"  # Data integrity, database issues
    TIMEOUT = "timeout"  # Operation timeout
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    UNKNOWN = "unknown"  # Uncategorized errors


class ErrorSeverity(Enum):
    """Error severity levels for prioritization"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Rich context information for errors"""
    timestamp: float = field(default_factory=time.time)
    error_id: str = ""
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    task_id: Optional[str] = None
    request_id: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization"""
        return {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "task_id": self.task_id,
            "request_id": self.request_id,
            "module": self.module,
            "function": self.function,
            "additional_data": self.additional_data,
            "stack_trace": self.stack_trace
        }


class AutoRLBaseException(Exception):
    """Base exception for all AutoRL custom exceptions"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "AUTORL_ERROR",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        self.recovery_suggestions = recovery_suggestions or []
        
        # Capture stack trace
        if not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "context": self.context.to_dict(),
            "recovery_suggestions": self.recovery_suggestions,
            "original_error": str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# Infrastructure Exceptions
class InfrastructureException(AutoRLBaseException):
    """Base for infrastructure-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INFRASTRUCTURE,
            **kwargs
        )


class DeviceConnectionException(InfrastructureException):
    """Device connection failures"""
    def __init__(self, message: str, device_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_code="DEVICE_CONNECTION_ERROR",
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=[
                "Check device connectivity",
                "Restart Appium server",
                "Verify USB/network connection",
                "Check device authorization"
            ],
            **kwargs
        )
        if device_id:
            self.context.device_id = device_id


class AppiumServerException(InfrastructureException):
    """Appium server not available or not responding"""
    def __init__(self, message: str = "Appium server is not available", **kwargs):
        super().__init__(
            message,
            error_code="APPIUM_SERVER_ERROR",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            recovery_suggestions=[
                "Start Appium server",
                "Check Appium server URL configuration",
                "Verify Appium server logs"
            ],
            **kwargs
        )


class ResourceExhaustedException(InfrastructureException):
    """System resources exhausted"""
    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="RESOURCE_EXHAUSTED",
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=[
                f"Free up {resource_type} resources",
                "Scale up system capacity",
                "Reduce concurrent operations"
            ],
            **kwargs
        )
        self.context.additional_data["resource_type"] = resource_type


# Validation Exceptions
class ValidationException(AutoRLBaseException):
    """Base for validation-related errors"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            recoverable=False,
            **kwargs
        )
        if field:
            self.context.additional_data["field"] = field


class InvalidInputException(ValidationException):
    """Invalid input parameters"""
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Any = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="INVALID_INPUT",
            field=field,
            **kwargs
        )
        if expected_type:
            self.context.additional_data["expected_type"] = expected_type
        if actual_value is not None:
            self.context.additional_data["actual_value"] = str(actual_value)[:100]


class ConfigurationException(ValidationException):
    """Configuration validation errors"""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        if config_key:
            self.context.additional_data["config_key"] = config_key


# Business Logic Exceptions
class BusinessLogicException(AutoRLBaseException):
    """Base for business logic errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )


class TaskExecutionException(BusinessLogicException):
    """Task execution failures"""
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_code="TASK_EXECUTION_ERROR",
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        if task_id:
            self.context.task_id = task_id


class PlanningException(BusinessLogicException):
    """Action planning failures"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="PLANNING_ERROR",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Retry with updated UI state",
                "Simplify task description",
                "Check LLM service availability"
            ],
            **kwargs
        )


class PerceptionException(BusinessLogicException):
    """UI perception and analysis failures"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="PERCEPTION_ERROR",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Recapture screenshot",
                "Check screen visibility",
                "Verify OCR service availability"
            ],
            **kwargs
        )


# External Service Exceptions
class ExternalServiceException(AutoRLBaseException):
    """Base for external service errors"""
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs
        )
        if service_name:
            self.context.additional_data["service_name"] = service_name


class LLMServiceException(ExternalServiceException):
    """LLM service failures"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="LLM",
            error_code="LLM_SERVICE_ERROR",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Check API key validity",
                "Verify API endpoint availability",
                "Check rate limits",
                "Retry with exponential backoff"
            ],
            **kwargs
        )


class DatabaseException(ExternalServiceException):
    """Database operation failures"""
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            service_name="Database",
            error_code="DATABASE_ERROR",
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        if operation:
            self.context.additional_data["operation"] = operation


class VectorDBException(ExternalServiceException):
    """Vector database (Qdrant) failures"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            service_name="Qdrant",
            error_code="VECTOR_DB_ERROR",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Check Qdrant server availability",
                "Verify collection exists",
                "Check network connectivity"
            ],
            **kwargs
        )


# Timeout Exceptions
class TimeoutException(AutoRLBaseException):
    """Operation timeout"""
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Increase timeout value",
                "Optimize operation performance",
                "Check for blocking operations"
            ],
            **kwargs
        )
        if operation:
            self.context.additional_data["operation"] = operation
        if timeout_seconds:
            self.context.additional_data["timeout_seconds"] = timeout_seconds


class ElementTimeoutException(TimeoutException):
    """UI element not found within timeout"""
    def __init__(
        self,
        message: str,
        element_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            operation="element_wait",
            timeout_seconds=timeout_seconds,
            error_code="ELEMENT_TIMEOUT",
            recovery_suggestions=[
                "Verify element exists on screen",
                "Check element locator accuracy",
                "Wait for page load completion",
                "Increase timeout duration"
            ],
            **kwargs
        )
        if element_id:
            self.context.additional_data["element_id"] = element_id


# Rate Limit Exceptions
class RateLimitException(AutoRLBaseException):
    """Rate limit exceeded"""
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        retry_after: int = 60,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="RATE_LIMIT_EXCEEDED",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            retry_after=retry_after,
            recovery_suggestions=[
                f"Wait {retry_after} seconds before retrying",
                "Reduce request frequency",
                "Implement request queuing"
            ],
            **kwargs
        )
        if service:
            self.context.additional_data["service"] = service


# Security Exceptions
class SecurityException(AutoRLBaseException):
    """Base for security-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )


class AuthenticationException(SecurityException):
    """Authentication failures"""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            error_code="AUTHENTICATION_ERROR",
            recovery_suggestions=[
                "Check credentials",
                "Verify API key",
                "Renew authentication token"
            ],
            **kwargs
        )


class AuthorizationException(SecurityException):
    """Authorization failures"""
    def __init__(self, message: str = "Access denied", resource: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_code="AUTHORIZATION_ERROR",
            recovery_suggestions=[
                "Verify user permissions",
                "Request appropriate access level"
            ],
            **kwargs
        )
        if resource:
            self.context.additional_data["resource"] = resource


class PIIException(SecurityException):
    """PII data handling errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="PII_ERROR",
            recovery_suggestions=[
                "Enable PII masking",
                "Review data handling policies"
            ],
            **kwargs
        )


# Data Exceptions
class DataException(AutoRLBaseException):
    """Base for data-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA,
            **kwargs
        )


class DataIntegrityException(DataException):
    """Data integrity violations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DATA_INTEGRITY_ERROR",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class DataNotFoundException(DataException):
    """Required data not found"""
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="DATA_NOT_FOUND",
            severity=ErrorSeverity.WARNING,
            **kwargs
        )
        if data_type:
            self.context.additional_data["data_type"] = data_type
        if identifier:
            self.context.additional_data["identifier"] = identifier


# UI Interaction Exceptions
class UIException(AutoRLBaseException):
    """Base for UI interaction errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )


class ElementNotFoundException(UIException):
    """UI element not found"""
    def __init__(
        self,
        message: str,
        element_id: Optional[str] = None,
        locator_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="ELEMENT_NOT_FOUND",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Verify element exists in UI",
                "Check element locator",
                "Wait for UI to load",
                "Re-analyze screen state"
            ],
            **kwargs
        )
        if element_id:
            self.context.additional_data["element_id"] = element_id
        if locator_type:
            self.context.additional_data["locator_type"] = locator_type


class ElementInteractionException(UIException):
    """Failed to interact with UI element"""
    def __init__(
        self,
        message: str,
        element_id: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="ELEMENT_INTERACTION_ERROR",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=[
                "Verify element is visible and enabled",
                "Wait for element to be interactive",
                "Try alternative interaction method",
                "Check for UI overlays"
            ],
            **kwargs
        )
        if element_id:
            self.context.additional_data["element_id"] = element_id
        if action:
            self.context.additional_data["action"] = action


class StaleElementException(UIException):
    """Element reference is stale"""
    def __init__(self, message: str, element_id: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            error_code="STALE_ELEMENT",
            severity=ErrorSeverity.WARNING,
            recovery_suggestions=[
                "Re-locate element",
                "Refresh UI state",
                "Wait for page stabilization"
            ],
            **kwargs
        )
        if element_id:
            self.context.additional_data["element_id"] = element_id


# Recovery Exception
class RecoveryFailedException(AutoRLBaseException):
    """Recovery strategy failed"""
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        attempted_strategies: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="RECOVERY_FAILED",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )
        if attempted_strategies:
            self.context.additional_data["attempted_strategies"] = attempted_strategies
        self.original_exception = original_error


# Exception Factory
class ExceptionFactory:
    """Factory for creating appropriate exceptions from generic exceptions"""
    
    @staticmethod
    def from_exception(
        exc: Exception,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> AutoRLBaseException:
        """Convert a generic exception to an AutoRL exception"""
        
        exc_type = type(exc).__name__
        exc_message = str(exc)
        
        # Map common exception types
        if "timeout" in exc_type.lower() or "timeout" in exc_message.lower():
            return TimeoutException(
                exc_message,
                context=context,
                original_exception=exc,
                **kwargs
            )
        
        elif "element" in exc_message.lower() and "not found" in exc_message.lower():
            return ElementNotFoundException(
                exc_message,
                context=context,
                original_exception=exc,
                **kwargs
            )
        
        elif "connection" in exc_message.lower() or "network" in exc_message.lower():
            return DeviceConnectionException(
                exc_message,
                context=context,
                original_exception=exc,
                **kwargs
            )
        
        elif "auth" in exc_message.lower():
            return AuthenticationException(
                exc_message,
                context=context,
                original_exception=exc,
                **kwargs
            )
        
        elif "rate" in exc_message.lower() and "limit" in exc_message.lower():
            return RateLimitException(
                exc_message,
                context=context,
                original_exception=exc,
                **kwargs
            )
        
        # Default to base exception
        return AutoRLBaseException(
            exc_message,
            error_code="UNKNOWN_ERROR",
            context=context,
            original_exception=exc,
            **kwargs
        )

