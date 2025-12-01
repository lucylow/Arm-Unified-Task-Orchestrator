"""Comprehensive Testing Suite for AutoRL"""
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime
import asyncio


class MockDevice:
    """Mock device for testing"""
    def __init__(self, platform: str = "android", device_id: str = "test_device"):
        self.platform = platform
        self.device_id = device_id
        self.session = Mock()
        self.is_real = False
        self._screen_state = None
    
    async def setup(self):
        """Setup mock device"""
        self.session.current_activity = "com.example.MainActivity"
    
    async def cleanup(self):
        """Cleanup mock device"""
        self.session = None
    
    def set_screen_state(self, ui_state):
        """Set mock screen state"""
        self._screen_state = ui_state


class TestUIState:
    """Mock UI state for testing"""
    def __init__(self, elements: list, app_package: str = "com.example.app"):
        self.elements = elements
        self.app_package = app_package


@pytest.fixture
async def mock_device():
    """Fixture for mock device"""
    device = MockDevice(platform="android")
    await device.setup()
    yield device
    await device.cleanup()


@pytest.fixture
def sample_ui_state():
    """Fixture for sample UI state"""
    return TestUIState(
        elements=[
            {"id": "login_button", "type": "button", "bounds": [100, 200, 200, 250], "text": "Login"},
            {"id": "username", "type": "input", "bounds": [50, 100, 250, 140], "text": ""},
            {"id": "password", "type": "input", "bounds": [50, 150, 250, 190], "text": ""}
        ],
        app_package="com.example.app"
    )


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test that errors are correctly classified"""
        from src.core.error_handling import ErrorClassifier
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        
        classifier = ErrorClassifier()
        
        # Test timeout error
        timeout_error = classifier.classify_error(
            TimeoutException("Element not found"),
            {"action": "tap", "element": "button"}
        )
        assert timeout_error.code == "TIMEOUT"
        assert timeout_error.recoverable == True
        
        # Test element not found error
        element_error = classifier.classify_error(
            NoSuchElementException("Element not found"),
            {"action": "tap", "element": "button"}
        )
        assert element_error.code == "ELEMENT_NOT_FOUND"
        assert element_error.recoverable == True
    
    @pytest.mark.asyncio
    async def test_recovery_manager(self, mock_device):
        """Test recovery manager functionality"""
        from src.runtime.recovery import RecoveryManager
        from src.tools.action_execution import ActionExecutor
        
        executor = Mock(spec=ActionExecutor)
        executor.driver = mock_device.session
        executor.tap = AsyncMock(return_value=True)
        executor.wait_for_displayed = AsyncMock(return_value=True)
        
        recovery_manager = RecoveryManager(executor, max_recovery_attempts=2)
        
        # Test successful recovery
        result = await recovery_manager.recover("test_stage")
        assert result == True


class TestDeviceHealthMonitoring:
    """Test device health monitoring"""
    
    @pytest.mark.asyncio
    async def test_device_health_tracking(self, mock_device):
        """Test device health is tracked correctly"""
        from src.runtime.device_health_monitor import DeviceHealthMonitor, DeviceStatus
        
        monitor = DeviceHealthMonitor(heartbeat_interval=0.1, unhealthy_threshold=2)
        
        await monitor.start_monitoring(mock_device)
        
        # Wait for a few heartbeats
        await asyncio.sleep(0.3)
        
        health = monitor.get_device_health(mock_device.device_id)
        assert health is not None
        assert health.device_id == mock_device.device_id
        
        await monitor.stop_monitoring(mock_device.device_id)
    
    @pytest.mark.asyncio
    async def test_unhealthy_device_detection(self, mock_device):
        """Test that unhealthy devices are detected"""
        from src.runtime.device_health_monitor import DeviceHealthMonitor, DeviceStatus
        
        # Make device unresponsive
        mock_device.session = None
        
        monitor = DeviceHealthMonitor(heartbeat_interval=0.1, unhealthy_threshold=2)
        await monitor.start_monitoring(mock_device)
        
        # Wait for health checks to fail
        await asyncio.sleep(0.5)
        
        health = monitor.get_device_health(mock_device.device_id)
        assert health.error_count >= 2
        
        await monitor.stop_monitoring(mock_device.device_id)


class TestSecurityAndPrivacy:
    """Test security and data privacy features"""
    
    def test_data_encryption(self):
        """Test sensitive data encryption"""
        from src.security.secure_data_handler import SecureDataHandler
        
        handler = SecureDataHandler()
        
        test_data = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "abc123xyz"
        }
        
        encrypted = handler.encrypt_sensitive_data(test_data)
        
        # Sensitive fields should be encrypted
        assert encrypted["password"] != "secret123"
        assert encrypted["api_key"] != "abc123xyz"
        assert encrypted["username"] == "test_user"  # Not in sensitive list
        
        # Decrypt and verify
        decrypted = handler.decrypt_sensitive_data(encrypted)
        assert decrypted["password"] == "secret123"
        assert decrypted["api_key"] == "abc123xyz"
    
    def test_log_sanitization(self):
        """Test log data sanitization"""
        from src.security.secure_data_handler import SecureDataHandler
        
        handler = SecureDataHandler()
        
        log_data = {
            "user_id": "12345",
            "password": "secret",
            "screenshot": "base64_data_here",
            "action": "login"
        }
        
        sanitized = handler.sanitize_logs(log_data)
        
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["screenshot"] == "[SCREENSHOT_DATA_REDACTED]"
        assert sanitized["action"] == "login"
        assert len(sanitized["user_id"]) == 16  # Hashed


class TestLoggingSystem:
    """Test advanced logging system"""
    
    def test_pii_masking(self):
        """Test that PII is masked in logs"""
        from src.core.advanced_logging import PIIMasker
        
        masker = PIIMasker()
        
        # Test SSN masking
        text = "My SSN is 123-45-6789"
        masked = masker.mask(text)
        assert "123-45-6789" not in masked
        assert "[REDACTED_SSN]" in masked
        
        # Test email masking
        text = "Contact me at test@example.com"
        masked = masker.mask(text)
        assert "test@example.com" not in masked
        assert "[REDACTED_EMAIL]" in masked
    
    def test_structured_logging(self):
        """Test structured log output"""
        from src.core.advanced_logging import get_logger
        import json
        
        logger = get_logger("TestLogger")
        
        # Log should not raise exceptions
        logger.info("Test message", user_id="123", action="test")
        logger.log_agent_action(
            agent_name="TestAgent",
            action="test_action",
            context={"key": "value"},
            result={"status": "success"}
        )


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_error_recovery(self, mock_device):
        """Test end-to-end error handling and recovery"""
        from src.core.error_handling import RobustActionExecutor
        from src.tools.action_execution import ActionExecutor
        
        base_executor = Mock(spec=ActionExecutor)
        base_executor.driver = mock_device.session
        
        robust_executor = RobustActionExecutor(base_executor, max_retries=3)
        
        # Mock action that fails twice then succeeds
        call_count = 0
        async def mock_action():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"success": True}
        
        result = await robust_executor.execute_with_recovery(mock_action)
        assert result["success"] == True
        assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
