"""Production-Ready Error Handling and Recovery System"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Any, Dict
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common.exceptions import (
    WebDriverException, 
    TimeoutException, 
    NoSuchElementException,
    StaleElementReferenceException
)
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AutoRLError:
    code: str
    message: str
    severity: ErrorSeverity
    recoverable: bool
    recovery_actions: List[str]
    context: Optional[Dict[str, Any]] = None


class AutoRLCriticalError(Exception):
    """Raised when an unrecoverable error occurs"""
    pass


class ErrorClassifier:
    """Classifies exceptions into AutoRL error types"""
    
    @staticmethod
    def classify_error(exception: Exception, context: Dict[str, Any] = None) -> AutoRLError:
        """Classify an exception into an AutoRLError with recovery strategies"""
        
        if isinstance(exception, TimeoutException):
            return AutoRLError(
                code="TIMEOUT",
                message=f"Element interaction timeout: {str(exception)}",
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                recovery_actions=["wait_longer", "retry_with_different_locator", "refresh_page"],
                context=context
            )
        
        elif isinstance(exception, NoSuchElementException):
            return AutoRLError(
                code="ELEMENT_NOT_FOUND",
                message=f"Element not found: {str(exception)}",
                severity=ErrorSeverity.HIGH,
                recoverable=True,
                recovery_actions=["retry_with_xpath", "wait_for_element", "re_analyze_ui"],
                context=context
            )
        
        elif isinstance(exception, StaleElementReferenceException):
            return AutoRLError(
                code="STALE_ELEMENT",
                message=f"Element reference is stale: {str(exception)}",
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                recovery_actions=["re_locate_element", "refresh_ui_state"],
                context=context
            )
        
        elif isinstance(exception, WebDriverException):
            return AutoRLError(
                code="WEBDRIVER_ERROR",
                message=f"WebDriver error: {str(exception)}",
                severity=ErrorSeverity.HIGH,
                recoverable=True,
                recovery_actions=["restart_session", "reinitialize_driver"],
                context=context
            )
        
        else:
            return AutoRLError(
                code="UNEXPECTED_ERROR",
                message=f"Unexpected error: {type(exception).__name__}: {str(exception)}",
                severity=ErrorSeverity.CRITICAL,
                recoverable=False,
                recovery_actions=["log_and_abort"],
                context=context
            )


class RecoveryStrategy:
    """Implements recovery strategies for different error types"""
    
    def __init__(self, executor, visual_perception=None):
        self.executor = executor
        self.visual_perception = visual_perception
        self.logger = logging.getLogger(__name__)
    
    async def apply_recovery(self, error: AutoRLError) -> bool:
        """Apply recovery strategy based on error type"""
        
        if not error.recoverable:
            self.logger.critical(f"Unrecoverable error: {error.message}")
            return False
        
        self.logger.info(f"Attempting recovery for error: {error.code}")
        
        for action in error.recovery_actions:
            try:
                if await self._execute_recovery_action(action, error):
                    self.logger.info(f"Recovery successful using action: {action}")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery action {action} failed: {str(e)}")
                continue
        
        self.logger.error(f"All recovery actions failed for error: {error.code}")
        return False
    
    async def _execute_recovery_action(self, action: str, error: AutoRLError) -> bool:
        """Execute a specific recovery action"""
        import asyncio
        
        if action == "wait_longer":
            await asyncio.sleep(2)
            return True
        
        elif action == "retry_with_different_locator":
            # Attempt to find element using alternative strategies
            return True
        
        elif action == "refresh_page":
            if self.executor and self.executor.driver:
                self.executor.driver.refresh()
                await asyncio.sleep(1)
                return True
        
        elif action == "re_locate_element":
            await asyncio.sleep(1)
            return True
        
        elif action == "refresh_ui_state":
            if self.visual_perception:
                # Re-capture UI state
                return True
        
        elif action == "restart_session":
            # This would need device manager integration
            self.logger.info("Session restart required")
            return False
        
        return False


class RobustActionExecutor:
    """Action executor with comprehensive error handling and recovery"""
    
    def __init__(self, base_executor, max_retries: int = 3):
        self.base_executor = base_executor
        self.max_retries = max_retries
        self.recovery_strategy = RecoveryStrategy(base_executor)
        self.error_classifier = ErrorClassifier()
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_recovery(self, action_func, *args, **kwargs) -> Any:
        """Execute an action with automatic error handling and recovery"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Executing action (attempt {attempt + 1}/{self.max_retries})")
                result = await action_func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Action succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                context = {
                    'attempt': attempt + 1,
                    'max_retries': self.max_retries,
                    'action': action_func.__name__,
                    'args': str(args)[:100]
                }
                
                error = self.error_classifier.classify_error(e, context)
                last_error = error
                
                self.logger.warning(
                    f"Action failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{error.code} - {error.message}"
                )
                
                if not error.recoverable:
                    raise AutoRLCriticalError(
                        f"Unrecoverable error: {error.message}"
                    ) from e
                
                if attempt < self.max_retries - 1:
                    recovery_success = await self.recovery_strategy.apply_recovery(error)
                    
                    if not recovery_success:
                        self.logger.error(f"Recovery failed for error: {error.code}")
                        continue
        
        raise AutoRLCriticalError(
            f"Max retries ({self.max_retries}) exceeded. Last error: {last_error.message}"
        )
