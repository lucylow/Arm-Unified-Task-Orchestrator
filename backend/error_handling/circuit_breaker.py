"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by detecting failing services and temporarily blocking requests.
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Optional, Any, Dict
from dataclasses import dataclass, field
from collections import deque
from threading import Lock

from .exceptions import (
    AutoRLBaseException,
    ExternalServiceException,
    InfrastructureException
)


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Successes needed in half-open to close
    timeout: float = 60.0  # Seconds before trying again (half-open)
    window_size: int = 100  # Size of rolling window for tracking
    excluded_exceptions: tuple = field(default_factory=tuple)  # Exceptions that don't count as failures


class CircuitBreakerOpenException(AutoRLBaseException):
    """Raised when circuit breaker is open"""
    def __init__(self, service_name: str, retry_after: float, **kwargs):
        super().__init__(
            f"Circuit breaker is OPEN for service '{service_name}'. "
            f"Service temporarily unavailable. Retry after {retry_after:.1f}s",
            error_code="CIRCUIT_BREAKER_OPEN",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.WARNING,
            retry_after=int(retry_after),
            **kwargs
        )


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    State transitions:
    CLOSED -> OPEN: After failure_threshold failures
    OPEN -> HALF_OPEN: After timeout period
    HALF_OPEN -> CLOSED: After success_threshold successes
    HALF_OPEN -> OPEN: On any failure
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        
        # Rolling window of recent calls
        self._call_history = deque(maxlen=self.config.window_size)
        
        self._lock = Lock()
        
        logger.info(f"CircuitBreaker '{name}' initialized with config: {self.config}")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._opened_at and (time.time() - self._opened_at >= self.config.timeout):
                    self._transition_to_half_open()
            
            return self._state
    
    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        logger.info(f"CircuitBreaker '{self.name}' transitioned to HALF_OPEN")
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        logger.warning(
            f"CircuitBreaker '{self.name}' OPENED after {self._failure_count} failures. "
            f"Will retry in {self.config.timeout}s"
        )
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None
        logger.info(f"CircuitBreaker '{self.name}' CLOSED - service recovered")
    
    def _record_success(self):
        """Record successful call"""
        with self._lock:
            self._call_history.append({"success": True, "time": time.time()})
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)
    
    def _record_failure(self, exception: Exception):
        """Record failed call"""
        with self._lock:
            # Check if this exception should be excluded
            if isinstance(exception, self.config.excluded_exceptions):
                return
            
            self._call_history.append({
                "success": False,
                "time": time.time(),
                "exception": type(exception).__name__
            })
            
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens circuit
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection (synchronous).
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Original exception from func
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            time_until_retry = self.config.timeout - (time.time() - self._opened_at)
            raise CircuitBreakerOpenException(
                service_name=self.name,
                retry_after=max(0, time_until_retry)
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Original exception from func
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            time_until_retry = self.config.timeout - (time.time() - self._opened_at)
            raise CircuitBreakerOpenException(
                service_name=self.name,
                retry_after=max(0, time_until_retry)
            )
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            recent_calls = list(self._call_history)
            total_calls = len(recent_calls)
            successful_calls = sum(1 for call in recent_calls if call["success"])
            failed_calls = total_calls - successful_calls
            
            success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
            
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "success_rate": success_rate,
                "last_failure_time": self._last_failure_time,
                "opened_at": self._opened_at,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                }
            }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self._transition_to_closed()
            self._call_history.clear()
            logger.info(f"CircuitBreaker '{self.name}' manually reset")


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()
        logger.info("CircuitBreakerManager initialized")
    
    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
                logger.info(f"Created new circuit breaker for service '{name}'")
            return self._breakers[name]
    
    def remove_breaker(self, name: str):
        """Remove circuit breaker"""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker for service '{name}'")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global circuit breaker manager
_global_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = CircuitBreakerManager()
    return _global_manager


def circuit_breaker(
    service_name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to add circuit breaker protection to a function.
    
    Usage:
        @circuit_breaker("my_service")
        async def call_external_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        breaker = get_circuit_breaker_manager().get_breaker(service_name, config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Import ErrorCategory and ErrorSeverity for CircuitBreakerOpenException
from .exceptions import ErrorCategory, ErrorSeverity

