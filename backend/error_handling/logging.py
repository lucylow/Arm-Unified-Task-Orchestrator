"""
Comprehensive Error Logging System

Provides structured logging with error context, filtering, and formatters.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from .exceptions import AutoRLBaseException, ErrorSeverity


class ErrorLogFormatter(logging.Formatter):
    """Custom formatter for error logs with structured output"""
    
    def __init__(self, include_traceback: bool = True, json_format: bool = False):
        super().__init__()
        self.include_traceback = include_traceback
        self.json_format = json_format
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record"""
        
        # Extract error information if available
        error_info = {}
        if hasattr(record, 'exc_info') and record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            
            if isinstance(exc_value, AutoRLBaseException):
                error_info = {
                    'error_code': exc_value.error_code,
                    'error_category': exc_value.category.value,
                    'error_severity': exc_value.severity.value,
                    'recoverable': exc_value.recoverable,
                    'recovery_suggestions': exc_value.recovery_suggestions,
                    'context': exc_value.context.to_dict()
                }
        
        if self.json_format:
            return self._format_json(record, error_info)
        else:
            return self._format_text(record, error_info)
    
    def _format_json(self, record: logging.LogRecord, error_info: Dict) -> str:
        """Format as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if error_info:
            log_data['error'] = error_info
        
        if self.include_traceback and record.exc_info:
            log_data['traceback'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)
    
    def _format_text(self, record: logging.LogRecord, error_info: Dict) -> str:
        """Format as human-readable text"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        parts = [
            f"[{timestamp}]",
            f"[{record.levelname}]",
            f"[{record.name}]",
            record.getMessage()
        ]
        
        if error_info:
            parts.append(f"\n  Error Code: {error_info.get('error_code')}")
            parts.append(f"  Category: {error_info.get('error_category')}")
            parts.append(f"  Severity: {error_info.get('error_severity')}")
            
            if error_info.get('recovery_suggestions'):
                suggestions = ', '.join(error_info['recovery_suggestions'])
                parts.append(f"  Recovery: {suggestions}")
        
        if self.include_traceback and record.exc_info:
            parts.append('\n' + self.formatException(record.exc_info))
        
        return ' '.join(parts)


class ErrorFilter(logging.Filter):
    """Filter logs based on error severity and other criteria"""
    
    def __init__(
        self,
        min_severity: Optional[ErrorSeverity] = None,
        categories: Optional[list] = None,
        exclude_codes: Optional[list] = None
    ):
        super().__init__()
        self.min_severity = min_severity
        self.categories = set(categories) if categories else None
        self.exclude_codes = set(exclude_codes) if exclude_codes else None
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record"""
        
        # Extract exception info
        if hasattr(record, 'exc_info') and record.exc_info:
            exc_value = record.exc_info[1]
            
            if isinstance(exc_value, AutoRLBaseException):
                # Check severity
                if self.min_severity:
                    severity_order = {
                        ErrorSeverity.DEBUG: 0,
                        ErrorSeverity.INFO: 1,
                        ErrorSeverity.WARNING: 2,
                        ErrorSeverity.ERROR: 3,
                        ErrorSeverity.CRITICAL: 4
                    }
                    if severity_order.get(exc_value.severity, 0) < severity_order.get(self.min_severity, 0):
                        return False
                
                # Check category
                if self.categories and exc_value.category.value not in self.categories:
                    return False
                
                # Check excluded codes
                if self.exclude_codes and exc_value.error_code in self.exclude_codes:
                    return False
        
        return True


class ErrorLogger:
    """
    Comprehensive error logging system.
    
    Features:
    - Multiple log handlers (file, console, rotating)
    - Structured logging formats
    - Error filtering
    - Context enrichment
    """
    
    def __init__(
        self,
        name: str = "autorl",
        log_dir: str = "./logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        json_format: bool = False
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler(console_level, json_format)
        self._setup_file_handlers(file_level, max_file_size, backup_count, json_format)
        
        self.logger.info(f"ErrorLogger initialized for '{name}'")
    
    def _setup_console_handler(self, level: int, json_format: bool):
        """Setup console logging handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            ErrorLogFormatter(include_traceback=False, json_format=json_format)
        )
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(
        self,
        level: int,
        max_size: int,
        backup_count: int,
        json_format: bool
    ):
        """Setup file logging handlers"""
        
        # General log file with rotation
        general_log = self.log_dir / "autorl.log"
        file_handler = RotatingFileHandler(
            general_log,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            ErrorLogFormatter(include_traceback=True, json_format=json_format)
        )
        self.logger.addHandler(file_handler)
        
        # Error-only log file
        error_log = self.log_dir / "errors.log"
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=max_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            ErrorLogFormatter(include_traceback=True, json_format=json_format)
        )
        self.logger.addHandler(error_handler)
        
        # Daily rotating log file
        daily_log = self.log_dir / "daily.log"
        daily_handler = TimedRotatingFileHandler(
            daily_log,
            when='midnight',
            interval=1,
            backupCount=30
        )
        daily_handler.setLevel(level)
        daily_handler.setFormatter(
            ErrorLogFormatter(include_traceback=True, json_format=json_format)
        )
        self.logger.addHandler(daily_handler)
    
    def log_error(
        self,
        error: AutoRLBaseException,
        extra_context: Optional[Dict[str, Any]] = None
    ):
        """Log an AutoRL exception with full context"""
        
        # Enrich context
        if extra_context:
            error.context.additional_data.update(extra_context)
        
        # Determine log level from severity
        level_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        level = level_map.get(error.severity, logging.ERROR)
        
        # Log the error
        self.logger.log(
            level,
            f"[{error.error_code}] {error.message}",
            exc_info=(type(error), error, error.__traceback__)
        )
    
    def add_filter(self, log_filter: logging.Filter):
        """Add filter to all handlers"""
        for handler in self.logger.handlers:
            handler.addFilter(log_filter)
    
    def get_logger(self) -> logging.Logger:
        """Get underlying logger instance"""
        return self.logger


# Global error logger instance
_global_error_logger: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get global error logger instance"""
    global _global_error_logger
    if _global_error_logger is None:
        _global_error_logger = ErrorLogger()
    return _global_error_logger


def setup_error_logging(
    log_dir: str = "./logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    json_format: bool = False
) -> ErrorLogger:
    """Setup global error logging"""
    global _global_error_logger
    _global_error_logger = ErrorLogger(
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        json_format=json_format
    )
    return _global_error_logger

