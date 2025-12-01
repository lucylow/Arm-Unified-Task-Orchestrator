"""Security and Data Privacy Module"""
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Set
import re
import os
from pathlib import Path
import json

class SecureDataHandler:
    """Handle sensitive data with encryption and anonymization"""
    
    def __init__(self, encryption_key: Optional[bytes] = None, key_file: Optional[Path] = None):
        if encryption_key:
            self.encryption_key = encryption_key
        elif key_file and key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            if key_file:
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
        
        self.cipher = Fernet(self.encryption_key)
        self.sensitive_fields = {
            'password', 'passwd', 'pwd',
            'secret', 'token', 'api_key', 'apikey',
            'ssn', 'social_security',
            'credit_card', 'cc_number', 'card_number',
            'auth', 'authorization',
            'private_key', 'secret_key'
        }
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data dictionary"""
        encrypted_data = {}
        
        for key, value in data.items():
            if self._is_sensitive_field(key):
                if isinstance(value, str):
                    encrypted_data[key] = self.cipher.encrypt(value.encode()).decode()
                else:
                    encrypted_data[key] = self.cipher.encrypt(str(value).encode()).decode()
            elif isinstance(value, dict):
                encrypted_data[key] = self.encrypt_sensitive_data(value)
            elif isinstance(value, list):
                encrypted_data[key] = [
                    self.encrypt_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in data dictionary"""
        decrypted_data = {}
        
        for key, value in data.items():
            if self._is_sensitive_field(key) and isinstance(value, str):
                try:
                    decrypted_data[key] = self.cipher.decrypt(value.encode()).decode()
                except Exception:
                    decrypted_data[key] = value
            elif isinstance(value, dict):
                decrypted_data[key] = self.decrypt_sensitive_data(value)
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field name indicates sensitive data"""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.sensitive_fields)
    
    def hash_user_identifier(self, identifier: str, salt: Optional[str] = None) -> str:
        """Create anonymized hash for user identification"""
        if salt is None:
            salt = os.getenv('AUTORL_HASH_SALT', 'default_salt_change_in_production')
        
        combined = f"{identifier}_{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def anonymize_device_id(self, device_id: str) -> str:
        """Anonymize device ID for logging"""
        return f"device_{self.hash_user_identifier(device_id)}"
    
    def sanitize_logs(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive information from logs"""
        sanitized = {}
        
        for key, value in log_data.items():
            if self._is_sensitive_field(key):
                sanitized[key] = '[REDACTED]'
            elif key in {'screenshot', 'screen_capture', 'image_data'}:
                sanitized[key] = '[SCREENSHOT_DATA_REDACTED]'
            elif key in {'user_input', 'input_text'}:
                sanitized[key] = '[USER_INPUT_MASKED]'
            elif key == 'device_id':
                sanitized[key] = self.anonymize_device_id(str(value))
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_logs(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_logs(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def mask_pii_patterns(self, text: str) -> str:
        """Mask common PII patterns in text"""
        patterns = {
            'ssn': (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
            'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
            'credit_card': (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CC_REDACTED]'),
            'phone': (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]'),
        }
        
        masked_text = text
        for pattern_type, (pattern, replacement) in patterns.items():
            masked_text = re.sub(pattern, replacement, masked_text)
        
        return masked_text


class AccessControl:
    """Manage access control and permissions"""
    
    def __init__(self):
        self.permissions: Dict[str, Set[str]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate a secure API key for a user"""
        api_key = secrets.token_urlsafe(32)
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'permissions': set(permissions),
            'created_at': None,  # Would use datetime in production
            'last_used': None
        }
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated metadata"""
        return self.api_keys.get(api_key)
    
    def check_permission(self, api_key: str, required_permission: str) -> bool:
        """Check if API key has required permission"""
        key_data = self.verify_api_key(api_key)
        
        if not key_data:
            return False
        
        return required_permission in key_data['permissions']
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]


class SecureConfigManager:
    """Manage secure configuration with encryption"""
    
    def __init__(self, config_file: Path, encryption_key: Optional[bytes] = None):
        self.config_file = config_file
        self.security_handler = SecureDataHandler(encryption_key)
        self.config: Dict[str, Any] = {}
        
        if config_file.exists():
            self.load_config()
    
    def load_config(self):
        """Load and decrypt configuration"""
        with open(self.config_file, 'r') as f:
            encrypted_config = json.load(f)
        
        self.config = self.security_handler.decrypt_sensitive_data(encrypted_config)
    
    def save_config(self):
        """Encrypt and save configuration"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        encrypted_config = self.security_handler.encrypt_sensitive_data(self.config)
        
        with open(self.config_file, 'w') as f:
            json.dump(encrypted_config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(updates)


# Module-level security handler
_security_handler: Optional[SecureDataHandler] = None


def get_security_handler(
    encryption_key: Optional[bytes] = None,
    key_file: Optional[Path] = None
) -> SecureDataHandler:
    """Get or create global security handler"""
    global _security_handler
    
    if _security_handler is None:
        _security_handler = SecureDataHandler(encryption_key, key_file)
    
    return _security_handler
