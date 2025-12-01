"""
Input Validation Utilities

Provides comprehensive validation with detailed error messages.
"""

import re
from typing import Any, Optional, List, Dict, Type, Union, Callable
from pathlib import Path
import logging

from .exceptions import (
    InvalidInputException,
    ConfigurationException,
    ValidationException
)


logger = logging.getLogger(__name__)


class Validator:
    """Base validator class"""
    
    def __init__(self, field_name: Optional[str] = None):
        self.field_name = field_name or "value"
    
    def validate(self, value: Any) -> Any:
        """
        Validate value and return it if valid.
        
        Raises:
            ValidationException: If validation fails
        """
        raise NotImplementedError()
    
    def _raise_error(self, message: str, **kwargs):
        """Raise validation error with context"""
        raise InvalidInputException(
            message,
            field=self.field_name,
            **kwargs
        )


class TypeValidator(Validator):
    """Validate value type"""
    
    def __init__(self, expected_type: Type, field_name: Optional[str] = None):
        super().__init__(field_name)
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> Any:
        if not isinstance(value, self.expected_type):
            self._raise_error(
                f"{self.field_name} must be of type {self.expected_type.__name__}, "
                f"got {type(value).__name__}",
                expected_type=self.expected_type.__name__,
                actual_value=value
            )
        return value


class StringValidator(Validator):
    """Validate string with various constraints"""
    
    def __init__(
        self,
        field_name: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[str]] = None,
        not_empty: bool = False
    ):
        super().__init__(field_name)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_values = set(allowed_values) if allowed_values else None
        self.not_empty = not_empty
    
    def validate(self, value: Any) -> str:
        # Type check
        if not isinstance(value, str):
            self._raise_error(
                f"{self.field_name} must be a string, got {type(value).__name__}",
                expected_type="str",
                actual_value=value
            )
        
        # Empty check
        if self.not_empty and not value.strip():
            self._raise_error(f"{self.field_name} cannot be empty")
        
        # Length checks
        if self.min_length is not None and len(value) < self.min_length:
            self._raise_error(
                f"{self.field_name} must be at least {self.min_length} characters, "
                f"got {len(value)}"
            )
        
        if self.max_length is not None and len(value) > self.max_length:
            self._raise_error(
                f"{self.field_name} must be at most {self.max_length} characters, "
                f"got {len(value)}"
            )
        
        # Pattern check
        if self.pattern and not self.pattern.match(value):
            self._raise_error(
                f"{self.field_name} does not match required pattern"
            )
        
        # Allowed values check
        if self.allowed_values and value not in self.allowed_values:
            self._raise_error(
                f"{self.field_name} must be one of {sorted(self.allowed_values)}, "
                f"got '{value}'"
            )
        
        return value


class NumberValidator(Validator):
    """Validate numeric values"""
    
    def __init__(
        self,
        field_name: Optional[str] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        integer_only: bool = False,
        positive: bool = False
    ):
        super().__init__(field_name)
        self.min_value = min_value
        self.max_value = max_value
        self.integer_only = integer_only
        self.positive = positive
    
    def validate(self, value: Any) -> Union[int, float]:
        # Type check
        if not isinstance(value, (int, float)):
            self._raise_error(
                f"{self.field_name} must be a number, got {type(value).__name__}",
                expected_type="number",
                actual_value=value
            )
        
        # Integer check
        if self.integer_only and not isinstance(value, int):
            self._raise_error(
                f"{self.field_name} must be an integer, got float"
            )
        
        # Positive check
        if self.positive and value <= 0:
            self._raise_error(
                f"{self.field_name} must be positive, got {value}"
            )
        
        # Range checks
        if self.min_value is not None and value < self.min_value:
            self._raise_error(
                f"{self.field_name} must be >= {self.min_value}, got {value}"
            )
        
        if self.max_value is not None and value > self.max_value:
            self._raise_error(
                f"{self.field_name} must be <= {self.max_value}, got {value}"
            )
        
        return value


class ListValidator(Validator):
    """Validate list/array values"""
    
    def __init__(
        self,
        field_name: Optional[str] = None,
        item_validator: Optional[Validator] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        not_empty: bool = False,
        unique_items: bool = False
    ):
        super().__init__(field_name)
        self.item_validator = item_validator
        self.min_length = min_length
        self.max_length = max_length
        self.not_empty = not_empty
        self.unique_items = unique_items
    
    def validate(self, value: Any) -> List[Any]:
        # Type check
        if not isinstance(value, (list, tuple)):
            self._raise_error(
                f"{self.field_name} must be a list, got {type(value).__name__}",
                expected_type="list",
                actual_value=value
            )
        
        value_list = list(value)
        
        # Empty check
        if self.not_empty and len(value_list) == 0:
            self._raise_error(f"{self.field_name} cannot be empty")
        
        # Length checks
        if self.min_length is not None and len(value_list) < self.min_length:
            self._raise_error(
                f"{self.field_name} must have at least {self.min_length} items, "
                f"got {len(value_list)}"
            )
        
        if self.max_length is not None and len(value_list) > self.max_length:
            self._raise_error(
                f"{self.field_name} must have at most {self.max_length} items, "
                f"got {len(value_list)}"
            )
        
        # Unique items check
        if self.unique_items and len(value_list) != len(set(value_list)):
            self._raise_error(f"{self.field_name} must contain unique items")
        
        # Validate each item
        if self.item_validator:
            validated_items = []
            for i, item in enumerate(value_list):
                try:
                    validated_items.append(self.item_validator.validate(item))
                except ValidationException as e:
                    self._raise_error(
                        f"{self.field_name}[{i}]: {e.message}"
                    )
            return validated_items
        
        return value_list


class DictValidator(Validator):
    """Validate dictionary/object values"""
    
    def __init__(
        self,
        field_name: Optional[str] = None,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        key_validators: Optional[Dict[str, Validator]] = None,
        allow_extra_keys: bool = True
    ):
        super().__init__(field_name)
        self.required_keys = set(required_keys or [])
        self.optional_keys = set(optional_keys or [])
        self.key_validators = key_validators or {}
        self.allow_extra_keys = allow_extra_keys
    
    def validate(self, value: Any) -> Dict[str, Any]:
        # Type check
        if not isinstance(value, dict):
            self._raise_error(
                f"{self.field_name} must be a dictionary, got {type(value).__name__}",
                expected_type="dict",
                actual_value=value
            )
        
        # Check required keys
        missing_keys = self.required_keys - set(value.keys())
        if missing_keys:
            self._raise_error(
                f"{self.field_name} is missing required keys: {sorted(missing_keys)}"
            )
        
        # Check for unexpected keys
        if not self.allow_extra_keys:
            allowed_keys = self.required_keys | self.optional_keys
            extra_keys = set(value.keys()) - allowed_keys
            if extra_keys:
                self._raise_error(
                    f"{self.field_name} contains unexpected keys: {sorted(extra_keys)}"
                )
        
        # Validate each key
        validated_dict = {}
        for key, val in value.items():
            if key in self.key_validators:
                try:
                    validated_dict[key] = self.key_validators[key].validate(val)
                except ValidationException as e:
                    self._raise_error(
                        f"{self.field_name}.{key}: {e.message}"
                    )
            else:
                validated_dict[key] = val
        
        return validated_dict


class PathValidator(Validator):
    """Validate file/directory paths"""
    
    def __init__(
        self,
        field_name: Optional[str] = None,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        create_if_missing: bool = False
    ):
        super().__init__(field_name)
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir
        self.create_if_missing = create_if_missing
    
    def validate(self, value: Any) -> Path:
        # Type check and convert to Path
        if isinstance(value, str):
            path = Path(value)
        elif isinstance(value, Path):
            path = value
        else:
            self._raise_error(
                f"{self.field_name} must be a string or Path, got {type(value).__name__}",
                expected_type="str or Path",
                actual_value=value
            )
        
        # Existence check
        if self.must_exist and not path.exists():
            self._raise_error(
                f"{self.field_name} does not exist: {path}"
            )
        
        # File check
        if self.must_be_file:
            if path.exists() and not path.is_file():
                self._raise_error(
                    f"{self.field_name} must be a file, not a directory: {path}"
                )
        
        # Directory check
        if self.must_be_dir:
            if path.exists() and not path.is_dir():
                self._raise_error(
                    f"{self.field_name} must be a directory, not a file: {path}"
                )
            
            # Create directory if requested
            if self.create_if_missing and not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    self._raise_error(
                        f"Failed to create directory {path}: {e}"
                    )
        
        return path


class EmailValidator(Validator):
    """Validate email addresses"""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def validate(self, value: Any) -> str:
        if not isinstance(value, str):
            self._raise_error(
                f"{self.field_name} must be a string, got {type(value).__name__}",
                expected_type="str",
                actual_value=value
            )
        
        if not self.EMAIL_PATTERN.match(value):
            self._raise_error(
                f"{self.field_name} must be a valid email address, got '{value}'"
            )
        
        return value


class URLValidator(Validator):
    """Validate URLs"""
    
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$',
        re.IGNORECASE
    )
    
    def validate(self, value: Any) -> str:
        if not isinstance(value, str):
            self._raise_error(
                f"{self.field_name} must be a string, got {type(value).__name__}",
                expected_type="str",
                actual_value=value
            )
        
        if not self.URL_PATTERN.match(value):
            self._raise_error(
                f"{self.field_name} must be a valid URL, got '{value}'"
            )
        
        return value


# Convenience validation functions
def validate_required(value: Any, field_name: str) -> Any:
    """Validate that value is not None"""
    if value is None:
        raise InvalidInputException(
            f"{field_name} is required",
            field=field_name
        )
    return value


def validate_string(
    value: Any,
    field_name: str,
    **kwargs
) -> str:
    """Validate string value"""
    return StringValidator(field_name, **kwargs).validate(value)


def validate_number(
    value: Any,
    field_name: str,
    **kwargs
) -> Union[int, float]:
    """Validate numeric value"""
    return NumberValidator(field_name, **kwargs).validate(value)


def validate_list(
    value: Any,
    field_name: str,
    **kwargs
) -> List[Any]:
    """Validate list value"""
    return ListValidator(field_name, **kwargs).validate(value)


def validate_dict(
    value: Any,
    field_name: str,
    **kwargs
) -> Dict[str, Any]:
    """Validate dictionary value"""
    return DictValidator(field_name, **kwargs).validate(value)


def validate_path(
    value: Any,
    field_name: str,
    **kwargs
) -> Path:
    """Validate path value"""
    return PathValidator(field_name, **kwargs).validate(value)


def validate_email(value: Any, field_name: str) -> str:
    """Validate email address"""
    return EmailValidator(field_name).validate(value)


def validate_url(value: Any, field_name: str) -> str:
    """Validate URL"""
    return URLValidator(field_name).validate(value)

