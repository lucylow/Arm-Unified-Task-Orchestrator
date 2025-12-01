"""
Example Integration of Error Handling System

Demonstrates how to integrate the comprehensive error handling system
into existing AutoRL modules.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from error_handling import (
    # Exceptions
    DeviceConnectionException,
    TaskExecutionException,
    ElementNotFoundException,
    TimeoutException,
    AppiumServerException,
    # Decorators
    with_error_handling,
    with_retry,
    with_timeout,
    log_execution,
    circuit_breaker,
    validate_args,
    # Validators
    validate_string,
    validate_number,
    validate_dict,
    # Circuit Breaker
    CircuitBreakerConfig,
    # Recovery
    get_recovery_manager,
    RecoveryStrategy,
    # Tracking
    track_error,
    # Logging
    setup_error_logging
)


logger = logging.getLogger(__name__)


class EnhancedDeviceManager:
    """
    Device manager with comprehensive error handling.
    
    Demonstrates integration of:
    - Custom exceptions
    - Validation
    - Retry logic
    - Circuit breaker
    - Error tracking
    - Recovery strategies
    """
    
    def __init__(self):
        self.devices: Dict[str, Any] = {}
        self.recovery_manager = get_recovery_manager()
        logger.info("EnhancedDeviceManager initialized")
    
    @validate_args(
        device_id=lambda x: validate_string(x, "device_id", not_empty=True, min_length=3)
    )
    @with_retry(max_attempts=3, delay=2.0, backoff=2.0)
    @with_timeout(30)
    @log_execution(include_duration=True)
    @with_error_handling()
    async def connect_device(self, device_id: str) -> Dict[str, Any]:
        """
        Connect to a device with full error handling.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device connection info
            
        Raises:
            DeviceConnectionException: If connection fails
        """
        logger.info(f"Connecting to device: {device_id}")
        
        try:
            # Check if Appium server is available
            appium_available = await self._check_appium_server()
            if not appium_available:
                raise AppiumServerException(
                    "Appium server is not responding"
                )
            
            # Simulate connection
            await asyncio.sleep(1)  # Simulated delay
            
            # Store device info
            device_info = {
                "device_id": device_id,
                "status": "connected",
                "platform": "android",
                "connected_at": asyncio.get_event_loop().time()
            }
            self.devices[device_id] = device_info
            
            logger.info(f"Successfully connected to device: {device_id}")
            return device_info
            
        except AppiumServerException:
            raise  # Re-raise as is
            
        except Exception as e:
            # Convert to DeviceConnectionException
            error = DeviceConnectionException(
                f"Failed to connect to device {device_id}",
                device_id=device_id,
                original_exception=e
            )
            
            # Track error
            track_error(error, additional_context={"operation": "device_connection"})
            
            # Attempt recovery
            recovered = await self.recovery_manager.recover(
                operation_name="device_connection",
                error=error,
                context={"device_id": device_id},
                strategy=RecoveryStrategy.RETRY
            )
            
            if not recovered:
                raise error
    
    @circuit_breaker("appium_server")
    @with_timeout(5)
    async def _check_appium_server(self) -> bool:
        """
        Check if Appium server is available.
        Uses circuit breaker to prevent cascading failures.
        """
        try:
            # Simulate health check
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    @validate_args(
        device_id=lambda x: validate_string(x, "device_id", not_empty=True),
        element_id=lambda x: validate_string(x, "element_id", not_empty=True),
        timeout=lambda x: validate_number(x, "timeout", positive=True, max_value=60)
    )
    @with_retry(max_attempts=3, delay=1.0)
    @with_timeout(30)
    @with_error_handling()
    async def find_element(
        self,
        device_id: str,
        element_id: str,
        timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Find UI element with error handling.
        
        Args:
            device_id: Device identifier
            element_id: Element identifier
            timeout: Search timeout in seconds
            
        Returns:
            Element information
            
        Raises:
            ElementNotFoundException: If element not found
        """
        logger.debug(f"Finding element {element_id} on device {device_id}")
        
        # Verify device is connected
        if device_id not in self.devices:
            raise DeviceConnectionException(
                f"Device {device_id} is not connected",
                device_id=device_id
            )
        
        try:
            # Simulate element search
            await asyncio.sleep(0.5)
            
            # Simulate element found
            element = {
                "element_id": element_id,
                "device_id": device_id,
                "found_at": asyncio.get_event_loop().time()
            }
            
            return element
            
        except Exception as e:
            error = ElementNotFoundException(
                f"Element {element_id} not found on device {device_id}",
                element_id=element_id
            )
            
            # Track error
            track_error(error, additional_context={
                "device_id": device_id,
                "timeout": timeout
            })
            
            raise error


class EnhancedTaskExecutor:
    """
    Task executor with comprehensive error handling.
    """
    
    def __init__(self, device_manager: EnhancedDeviceManager):
        self.device_manager = device_manager
        self.recovery_manager = get_recovery_manager()
    
    @validate_args(
        task_name=lambda x: validate_string(x, "task_name", not_empty=True),
        device_id=lambda x: validate_string(x, "device_id", not_empty=True),
        task_config=lambda x: validate_dict(
            x,
            "task_config",
            required_keys=["actions"],
            allow_extra_keys=True
        )
    )
    @with_timeout(60)
    @log_execution(include_duration=True, include_result=True)
    @with_error_handling()
    async def execute_task(
        self,
        task_name: str,
        device_id: str,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute task with comprehensive error handling.
        
        Args:
            task_name: Name of the task
            device_id: Target device
            task_config: Task configuration
            
        Returns:
            Task execution result
        """
        logger.info(f"Executing task '{task_name}' on device {device_id}")
        
        try:
            # Ensure device is connected
            await self.device_manager.connect_device(device_id)
            
            # Execute actions
            actions = task_config.get("actions", [])
            results = []
            
            for action in actions:
                result = await self._execute_action(
                    device_id=device_id,
                    action=action
                )
                results.append(result)
            
            return {
                "task_name": task_name,
                "device_id": device_id,
                "status": "completed",
                "results": results
            }
            
        except (DeviceConnectionException, ElementNotFoundException) as e:
            # Known exceptions - track and attempt recovery
            track_error(e, additional_context={
                "task_name": task_name,
                "device_id": device_id
            })
            
            # Try recovery
            recovered = await self.recovery_manager.recover(
                operation_name=f"task_{task_name}",
                error=e,
                context={
                    "task_name": task_name,
                    "device_id": device_id,
                    "task_config": task_config
                },
                strategy=RecoveryStrategy.RETRY
            )
            
            if not recovered:
                raise TaskExecutionException(
                    f"Task '{task_name}' failed on device {device_id}",
                    task_id=task_name,
                    original_exception=e
                )
        
        except Exception as e:
            # Unknown exception
            error = TaskExecutionException(
                f"Task '{task_name}' failed unexpectedly: {e}",
                task_id=task_name,
                original_exception=e
            )
            track_error(error)
            raise error
    
    @with_retry(max_attempts=2, delay=1.0)
    @with_error_handling()
    async def _execute_action(
        self,
        device_id: str,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single action with retry"""
        action_type = action.get("type")
        logger.debug(f"Executing action: {action_type}")
        
        # Simulate action execution
        await asyncio.sleep(0.2)
        
        return {
            "action": action_type,
            "status": "success",
            "device_id": device_id
        }


# Example usage
async def example_usage():
    """Demonstrate usage of enhanced modules"""
    
    # Setup error logging
    setup_error_logging(
        log_dir="./logs",
        console_level=logging.INFO,
        json_format=False
    )
    
    # Create instances
    device_manager = EnhancedDeviceManager()
    task_executor = EnhancedTaskExecutor(device_manager)
    
    try:
        # Connect to device
        device_info = await device_manager.connect_device("emulator-5554")
        print(f"Connected: {device_info}")
        
        # Find element
        element = await device_manager.find_element(
            device_id="emulator-5554",
            element_id="login_button",
            timeout=10
        )
        print(f"Element found: {element}")
        
        # Execute task
        result = await task_executor.execute_task(
            task_name="login_flow",
            device_id="emulator-5554",
            task_config={
                "actions": [
                    {"type": "tap", "element": "username_field"},
                    {"type": "type", "text": "testuser"},
                    {"type": "tap", "element": "login_button"}
                ]
            }
        )
        print(f"Task completed: {result}")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        # Error is already tracked and logged


if __name__ == "__main__":
    asyncio.run(example_usage())

