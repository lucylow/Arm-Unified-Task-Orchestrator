"""
Error Handling Decorators and Utilities

Provides decorators for automatic error handling, retry logic, and timeouts.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Optional, Type, Tuple, Any, Union
import traceback

from .exceptions import (
    AutoRLBaseException,
    TimeoutException,
    RecoveryFailedException,
    ExceptionFactory,
    ErrorContext
)
from .tracker import track_error


logger = logging.getLogger(__name__)


def with_error_handling(
    error_category: Optional[str] = None,
    reraise: bool = True,
    default_return: Any = None,
    log_errors: bool = True,
    track_errors: bool = True
):
    """
    Decorator to add comprehensive error handling to a function.
    
    Args:
        error_category: Category for error tracking
        reraise: Whether to reraise exceptions after handling
        default_return: Default value to return on error (if not reraising)
        log_errors: Whether to log errors
        track_errors: Whether to track errors in error tracker
        
    Usage:
        @with_error_handling(error_category="device_management")
        async def connect_device(device_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AutoRLBaseException as e:
                # Already an AutoRL exception
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if track_errors:
                    track_error(e)
                
                if reraise:
                    raise
                return default_return
                
            except Exception as e:
                # Convert to AutoRL exception
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__,
                    additional_data={"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
                )
                
                autorl_exc = ExceptionFactory.from_exception(e, context=context)
                
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {e}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                
                if track_errors:
                    track_error(autorl_exc)
                
                if reraise:
                    raise autorl_exc from e
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AutoRLBaseException as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                
                if track_errors:
                    track_error(e)
                
                if reraise:
                    raise
                return default_return
                
            except Exception as e:
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__,
                    additional_data={"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
                )
                
                autorl_exc = ExceptionFactory.from_exception(e, context=context)
                
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {e}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                
                if track_errors:
                    track_error(autorl_exc)
                
                if reraise:
                    raise autorl_exc from e
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator to add retry logic with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay (exponential backoff)
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to retry on
        on_retry: Callback function called on each retry
        
    Usage:
        @with_retry(max_attempts=3, delay=1.0, backoff=2.0)
        async def unstable_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            if asyncio.iscoroutinefunction(on_retry):
                                await on_retry(attempt, e, current_delay)
                            else:
                                on_retry(attempt, e, current_delay)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    # Wait before retry
                    await asyncio.sleep(current_delay)
                    
                    # Exponential backoff
                    current_delay = min(current_delay * backoff, max_delay)
            
            # Should not reach here
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    if on_retry:
                        try:
                            on_retry(attempt, e, current_delay)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def with_timeout(
    seconds: float,
    error_message: Optional[str] = None
):
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
        error_message: Custom error message
        
    Usage:
        @with_timeout(30)
        async def long_running_task():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError as e:
                msg = error_message or f"{func.__name__} timed out after {seconds}s"
                raise TimeoutException(
                    msg,
                    operation=func.__name__,
                    timeout_seconds=int(seconds)
                ) from e
        
        return wrapper
    
    return decorator


def log_execution(
    level: int = logging.INFO,
    include_args: bool = True,
    include_result: bool = False,
    include_duration: bool = True
):
    """
    Decorator to log function execution.
    
    Args:
        level: Logging level
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        include_duration: Whether to log execution duration
        
    Usage:
        @log_execution(include_duration=True)
        async def important_operation(param1, param2):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log entry
            log_msg = f"Executing {func.__name__}"
            if include_args:
                log_msg += f" with args={args[:3]}, kwargs={list(kwargs.keys())}"  # Limit args logging
            logger.log(level, log_msg)
            
            try:
                result = await func(*args, **kwargs)
                
                # Log exit
                duration = time.time() - start_time
                log_msg = f"Completed {func.__name__}"
                if include_duration:
                    log_msg += f" in {duration:.3f}s"
                if include_result:
                    log_msg += f" with result={str(result)[:100]}"
                logger.log(level, log_msg)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {func.__name__} after {duration:.3f}s: {e}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            log_msg = f"Executing {func.__name__}"
            if include_args:
                log_msg += f" with args={args[:3]}, kwargs={list(kwargs.keys())}"
            logger.log(level, log_msg)
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                log_msg = f"Completed {func.__name__}"
                if include_duration:
                    log_msg += f" in {duration:.3f}s"
                if include_result:
                    log_msg += f" with result={str(result)[:100]}"
                logger.log(level, log_msg)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {func.__name__} after {duration:.3f}s: {e}"
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def fallback_on_error(fallback_func: Callable):
    """
    Decorator to call fallback function on error.
    
    Args:
        fallback_func: Function to call when main function fails
        
    Usage:
        def fallback_implementation(*args, **kwargs):
            return "fallback result"
        
        @fallback_on_error(fallback_implementation)
        async def primary_implementation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{func.__name__} failed: {e}. Using fallback implementation."
                )
                
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                return fallback_func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{func.__name__} failed: {e}. Using fallback implementation."
                )
                return fallback_func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def safe_execute(
    default_return: Any = None,
    log_errors: bool = True,
    suppress_exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator to execute function safely, suppressing exceptions.
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
        suppress_exceptions: Tuple of exceptions to suppress
        
    Usage:
        @safe_execute(default_return=[])
        def get_items():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except suppress_exceptions as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except suppress_exceptions as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_args(**validators):
    """
    Decorator to validate function arguments.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validator functions
        
    Usage:
        from error_handling.validators import validate_string, validate_number
        
        @validate_args(
            device_id=lambda x: validate_string(x, "device_id", not_empty=True),
            timeout=lambda x: validate_number(x, "timeout", positive=True)
        )
        async def connect_device(device_id: str, timeout: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except Exception as e:
                        logger.error(f"Validation failed for {param_name}: {e}")
                        raise
            
            return await func(*bound_args.args, **bound_args.kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except Exception as e:
                        logger.error(f"Validation failed for {param_name}: {e}")
                        raise
            
            return func(*bound_args.args, **bound_args.kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

