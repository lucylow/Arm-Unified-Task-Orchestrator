"""Enhanced safety guardrails for production-ready AI agents"""
import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SafetyGuardrails:
    """Enhanced safety guardrails for production-ready AI agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',              # Credit card
            r'\b\+1-\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
    
    def sanitize_screenshot_data(self, ui_state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from UI state before processing"""
        sanitized_state = ui_state.copy()
        
        for element in sanitized_state.get('text_elements', []):
            if 'text' in element:
                element['text'] = self.mask_sensitive_info(element['text'])
        
        return sanitized_state
    
    def mask_sensitive_info(self, text: str) -> str:
        """Mask sensitive information in text"""
        if not text:
            return text
            
        for pattern in self.sensitive_patterns:
            text = re.sub(pattern, '[REDACTED]', text)
        return text
    
    def validate_action_safety(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if an action is safe to execute"""
        validation_result = {
            'safe': True,
            'reason': '',
            'suggested_alternative': None
        }
        
        # Prevent destructive actions in production without confirmation
        destructive_keywords = ['delete', 'remove', 'uninstall', 'format', 'erase']
        action_str = str(action).lower()
        
        if any(keyword in action_str for keyword in destructive_keywords):
            validation_result.update({
                'safe': False,
                'reason': 'Action contains potentially destructive operation',
                'suggested_alternative': {'type': 'require_human_approval', 'original_action': action}
            })
        
        return validation_result
    
    def check_rate_limits(self, action_count: int, time_window: float) -> bool:
        """Check if action rate is within safe limits"""
        max_actions_per_minute = self.config.get('max_actions_per_minute', 30)
        actions_per_minute = (action_count / time_window) * 60
        
        if actions_per_minute > max_actions_per_minute:
            logger.warning(f"Rate limit exceeded: {actions_per_minute:.1f} actions/min")
            return False
        
        return True
