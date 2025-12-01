import re
import logging
from typing import Dict, Any, Optional
import hashlib
import json

logger = logging.getLogger("autorl.security")

class PIIMaskingManager:
    """Comprehensive PII masking and security manager"""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[- ]?)?\d{10,}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'api_key': r'sk-[a-zA-Z0-9]{48}',  # OpenAI API key pattern
            'password': r'password["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'token': r'token["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'secret': r'secret["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'key': r'key["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            'address': r'\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
            'uuid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
        }
        
        self.logger = logging.getLogger("autorl.security.pii")
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask PII in text with asterisks"""
        if not isinstance(text, str):
            return text
            
        masked_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            def mask_match(match):
                matched_text = match.group(0)
                if len(matched_text) <= 4:
                    return "*" * len(matched_text)
                else:
                    # Show first 2 and last 2 characters, mask the rest
                    return matched_text[:2] + "*" * (len(matched_text) - 4) + matched_text[-2:]
            
            masked_text = re.sub(pattern, mask_match, masked_text, flags=re.IGNORECASE)
        
        return masked_text
    
    def mask_log_data(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask PII in log data structures"""
        if isinstance(log_data, dict):
            return {k: self.mask_log_data(v) for k, v in log_data.items()}
        elif isinstance(log_data, list):
            return [self.mask_log_data(item) for item in log_data]
        elif isinstance(log_data, str):
            return self.mask_sensitive_data(log_data)
        else:
            return log_data
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for storage/transmission"""
        if not isinstance(data, str):
            data = str(data)
        return hashlib.sha256(data.encode()).hexdigest()[:16]  # First 16 chars of hash
    
    def sanitize_for_logging(self, data: Any) -> Any:
        """Sanitize data for safe logging"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Mask keys that might contain sensitive information
                safe_key = self.mask_sensitive_data(key) if isinstance(key, str) else key
                sanitized[safe_key] = self.sanitize_for_logging(value)
            return sanitized
        elif isinstance(data, list):
            return [self.sanitize_for_logging(item) for item in data]
        elif isinstance(data, str):
            return self.mask_sensitive_data(data)
        else:
            return data

class SecureFormatter(logging.Formatter):
    """Secure logging formatter that masks PII"""
    
    def __init__(self):
        super().__init__(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.pii_manager = PIIMaskingManager()
    
    def format(self, record):
        # Mask PII in log messages
        if hasattr(record, 'msg'):
            record.msg = self.pii_manager.mask_sensitive_data(str(record.msg))
        
        # Mask PII in exception messages
        if record.exc_info:
            record.exc_text = self.pii_manager.mask_sensitive_data(
                self.formatException(record.exc_info)
            )
        
        return super().format(record)

class SecurityManager:
    """Comprehensive security manager for AutoRL"""
    
    def __init__(self):
        self.pii_manager = PIIMaskingManager()
        self.logger = logging.getLogger("autorl.security.manager")
        
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and security"""
        if not api_key:
            return False
        
        # Check if API key is properly masked
        if self.pii_manager.mask_sensitive_data(api_key) == api_key:
            self.logger.warning("API key appears to be unmasked in logs")
            return False
        
        # Basic format validation for OpenAI keys
        if api_key.startswith('sk-') and len(api_key) == 51:
            return True
        
        return False
    
    def sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data before processing"""
        return self.pii_manager.sanitize_for_logging(data)
    
    def validate_device_access(self, device_id: str, user_id: str) -> bool:
        """Validate if user has access to device"""
        # Implement device access validation logic
        # For now, basic validation
        if not device_id or not user_id:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = ['..', 'admin', 'root', 'system']
        if any(pattern in device_id.lower() for pattern in suspicious_patterns):
            self.logger.warning(f"Suspicious device ID pattern: {device_id}")
            return False
        
        return True
    
    def audit_action(self, action: str, user_id: str, device_id: str, success: bool):
        """Audit security-relevant actions"""
        audit_data = {
            "action": action,
            "user_id": self.pii_manager.hash_sensitive_data(user_id),
            "device_id": device_id,
            "success": success,
            "timestamp": self.pii_manager.mask_sensitive_data(str(time.time()))
        }
        
        self.logger.info(f"SECURITY_AUDIT: {json.dumps(audit_data)}")

# Global security manager instance
security_manager = SecurityManager()

def setup_secure_logging():
    """Configure secure logging for the application"""
    logger = logging.getLogger("autorl")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    handler = logging.StreamHandler()
    handler.setFormatter(SecureFormatter())
    
    logger.addHandler(handler)
    return logger




