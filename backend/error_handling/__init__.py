"""
AutoRL Error Handling Module

Comprehensive error handling, tracking, and recovery system.
"""

from .exceptions import (
    AutoRLBaseException,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    InfrastructureException,
    DeviceConnectionException,
    AppiumServerException,
    ResourceExhaustedException,
    ValidationException,
    InvalidInputException,
    ConfigurationException,
    BusinessLogicException,
    TaskExecutionException,
    PlanningException,
    PerceptionException,
    ExternalServiceException,
    LLMServiceException,
    DatabaseException,
    VectorDBException,
    TimeoutException,
    ElementTimeoutException,
    RateLimitException,
    SecurityException,
    AuthenticationException,
    AuthorizationException,
    PIIException,
    DataException,
    DataIntegrityException,
    DataNotFoundException,
    UIException,
    ElementNotFoundException,
    ElementInteractionException,
    StaleElementException,
    RecoveryFailedException,
    ExceptionFactory
)

from .tracker import (
    ErrorTracker,
    get_error_tracker,
    track_error
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenException,
    CircuitBreakerManager,
    get_circuit_breaker_manager,
    circuit_breaker
)

from .validators import (
    Validator,
    TypeValidator,
    StringValidator,
    NumberValidator,
    ListValidator,
    DictValidator,
    PathValidator,
    EmailValidator,
    URLValidator,
    validate_required,
    validate_string,
    validate_number,
    validate_list,
    validate_dict,
    validate_path,
    validate_email,
    validate_url
)

from .decorators import (
    with_error_handling,
    with_retry,
    with_timeout,
    log_execution,
    fallback_on_error,
    safe_execute,
    validate_args
)

from .recovery import (
    RecoveryStrategy,
    FailedOperation,
    DeadLetterQueue,
    RecoveryManager,
    get_dead_letter_queue,
    get_recovery_manager
)

from .logging import (
    ErrorLogFormatter,
    ErrorFilter,
    ErrorLogger,
    get_error_logger,
    setup_error_logging
)

__all__ = [
    # Exceptions
    "AutoRLBaseException",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorContext",
    "ExceptionFactory",
    "InfrastructureException",
    "DeviceConnectionException",
    "AppiumServerException",
    "ResourceExhaustedException",
    "ValidationException",
    "InvalidInputException",
    "ConfigurationException",
    "BusinessLogicException",
    "TaskExecutionException",
    "PlanningException",
    "PerceptionException",
    "ExternalServiceException",
    "LLMServiceException",
    "DatabaseException",
    "VectorDBException",
    "TimeoutException",
    "ElementTimeoutException",
    "RateLimitException",
    "SecurityException",
    "AuthenticationException",
    "AuthorizationException",
    "PIIException",
    "DataException",
    "DataIntegrityException",
    "DataNotFoundException",
    "UIException",
    "ElementNotFoundException",
    "ElementInteractionException",
    "StaleElementException",
    "RecoveryFailedException",
    
    # Tracking
    "ErrorTracker",
    "get_error_tracker",
    "track_error",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitBreakerOpenException",
    "CircuitBreakerManager",
    "get_circuit_breaker_manager",
    "circuit_breaker",
    
    # Validators
    "Validator",
    "TypeValidator",
    "StringValidator",
    "NumberValidator",
    "ListValidator",
    "DictValidator",
    "PathValidator",
    "EmailValidator",
    "URLValidator",
    "validate_required",
    "validate_string",
    "validate_number",
    "validate_list",
    "validate_dict",
    "validate_path",
    "validate_email",
    "validate_url",
    
    # Decorators
    "with_error_handling",
    "with_retry",
    "with_timeout",
    "log_execution",
    "fallback_on_error",
    "safe_execute",
    "validate_args",
    
    # Recovery
    "RecoveryStrategy",
    "FailedOperation",
    "DeadLetterQueue",
    "RecoveryManager",
    "get_dead_letter_queue",
    "get_recovery_manager",
    
    # Logging
    "ErrorLogFormatter",
    "ErrorFilter",
    "ErrorLogger",
    "get_error_logger",
    "setup_error_logging"
]

