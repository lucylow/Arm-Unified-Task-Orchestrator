"""Advanced Logging System with PII Masking and Structured Output"""
import logging
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib


class PIIMasker:
    """Masks personally identifiable information in logs"""
    
    def __init__(self):
        self.patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b'
        }
    
    def mask(self, text: str) -> str:
        """Mask sensitive data in text"""
        if not isinstance(text, str):
            text = str(text)
        
        masked = text
        for pattern_name, pattern in self.patterns.items():
            masked = re.sub(pattern, f'[REDACTED_{pattern_name.upper()}]', masked)
        
        return masked
    
    def mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask sensitive data in dictionaries"""
        if not isinstance(data, dict):
            return data
        
        masked = {}
        sensitive_keys = {'password', 'token', 'secret', 'api_key', 'auth', 'credential'}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                masked[key] = '[REDACTED]'
            elif isinstance(value, str):
                masked[key] = self.mask(value)
            elif isinstance(value, dict):
                masked[key] = self.mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [self.mask_dict(item) if isinstance(item, dict) else self.mask(str(item)) for item in value]
            else:
                masked[key] = value
        
        return masked


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self):
        super().__init__()
        self.masker = PIIMasker()
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': self.masker.mask(record.getMessage()),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = self.masker.mask_dict(record.extra_data)
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class AutoRLLogger:
    """Advanced logger with PII masking and structured output"""
    
    def __init__(self, name: str = "AutoRL", log_dir: Optional[Path] = None):
        self.logger = self._setup_logger(name, log_dir)
        self.masker = PIIMasker()
        self.session_id = self._generate_session_id()
    
    def _setup_logger(self, name: str, log_dir: Optional[Path]) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Console handler with structured format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(StructuredFormatter())
        logger.addHandler(console_handler)
        
        # File handler for persistent logs
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"autorl_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(file_handler)
        
        return logger
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.sha256(
            f"{datetime.utcnow().isoformat()}_{id(self)}".encode()
        ).hexdigest()[:16]
    
    def log_agent_action(
        self, 
        agent_name: str, 
        action: str,
        context: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        level: str = 'INFO'
    ):
        """Log an agent action with context"""
        log_data = {
            'agent': agent_name,
            'action': action,
            'context': self.masker.mask_dict(context),
            'session_id': self.session_id,
            'trace_id': context.get('trace_id', 'unknown')
        }
        
        if result:
            log_data['result'] = self.masker.mask_dict(result)
        
        log_record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.upper()),
            __file__,
            0,
            f"Agent action: {agent_name}.{action}",
            (),
            None
        )
        log_record.extra_data = log_data
        
        self.logger.handle(log_record)
    
    def log_task_execution(
        self,
        task_id: str,
        status: str,
        duration: float,
        metadata: Dict[str, Any]
    ):
        """Log task execution metrics"""
        log_data = {
            'task_id': task_id,
            'status': status,
            'duration_seconds': duration,
            'metadata': self.masker.mask_dict(metadata),
            'session_id': self.session_id
        }
        
        log_record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            __file__,
            0,
            f"Task execution: {task_id} - {status}",
            (),
            None
        )
        log_record.extra_data = log_data
        
        self.logger.handle(log_record)
    
    def log_error(
        self,
        error_code: str,
        error_message: str,
        context: Dict[str, Any],
        severity: str = 'ERROR'
    ):
        """Log errors with context"""
        log_data = {
            'error_code': error_code,
            'error_message': self.masker.mask(error_message),
            'context': self.masker.mask_dict(context),
            'session_id': self.session_id
        }
        
        log_record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, severity.upper()),
            __file__,
            0,
            f"Error: {error_code}",
            (),
            None
        )
        log_record.extra_data = log_data
        
        self.logger.handle(log_record)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self.masker.mask(message), extra={'extra_data': kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self.masker.mask(message), extra={'extra_data': kwargs})
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self.masker.mask(message), extra={'extra_data': kwargs})
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self.masker.mask(message), extra={'extra_data': kwargs})
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self.masker.mask(message), extra={'extra_data': kwargs})


# Global logger instance
_global_logger: Optional[AutoRLLogger] = None


def get_logger(name: str = "AutoRL", log_dir: Optional[Path] = None) -> AutoRLLogger:
    """Get or create global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = AutoRLLogger(name, log_dir)
    
    return _global_logger
