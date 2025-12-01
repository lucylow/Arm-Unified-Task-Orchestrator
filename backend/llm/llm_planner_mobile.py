"""
Enhanced LLM planner with on-device fallback support.

Supports:
- Cloud LLM planning (primary)
- On-device planning model fallback (when offline/cloud unavailable)
- Rule-based fallback (last resort)

For mobile deployment, the on-device planner uses lightweight quantized models
via the runtime abstraction layer.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import runtime abstraction for on-device planning
try:
    import sys
    backend_path = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_path))
    from inference.runtime import Runtime
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False
    Runtime = None

logger = logging.getLogger(__name__)


class LLMPlannerMobile:
    """
    Enhanced LLM planner with on-device fallback.
    
    Planning modes (priority order):
    1. Cloud LLM (if available)
    2. On-device planning model (if model loaded)
    3. Rule-based fallback (always available)
    """
    
    def __init__(
        self,
        cloud_llm_enabled: bool = True,
        on_device_model_path: Optional[str] = None,
        prefer_backend: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM planner with fallback options.
        
        Args:
            cloud_llm_enabled: Enable cloud LLM planning (default: True)
            on_device_model_path: Path to on-device planning model
            prefer_backend: Preferred runtime backend
            device_info: Device capabilities dictionary
        """
        self.cloud_llm_enabled = cloud_llm_enabled
        self.on_device_runtime = None
        self.device_info = device_info or {}
        
        # Load on-device planning model if path provided
        if on_device_model_path and RUNTIME_AVAILABLE:
            try:
                model_path = str(Path(on_device_model_path).expanduser().resolve())
                if os.path.exists(model_path):
                    self.on_device_runtime = Runtime.load(
                        model_path,
                        prefer=prefer_backend,
                        device_info=device_info
                    )
                    logger.info(f"Loaded on-device planning model: {model_path} (backend: {self.on_device_runtime.backend})")
                else:
                    logger.warning(f"On-device model not found: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load on-device planning model: {e}")
    
    def generate_action_plan(
        self,
        instruction: str,
        ui_state: Dict[str, Any],
        force_on_device: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate action plan with automatic fallback.
        
        Args:
            instruction: Natural language task instruction
            ui_state: Current UI state from perception
            force_on_device: Force on-device planning (skip cloud)
            
        Returns:
            List of action dictionaries
        """
        logger.info(f"Generating plan for: '{instruction}'")
        
        # Try cloud LLM first (unless forced on-device or offline)
        if not force_on_device and self.cloud_llm_enabled:
            try:
                plan = self._generate_cloud_plan(instruction, ui_state)
                if plan:
                    logger.info("Plan generated using cloud LLM")
                    return plan
            except Exception as e:
                logger.warning(f"Cloud LLM planning failed: {e}, falling back to on-device")
        
        # Try on-device planning model
        if self.on_device_runtime:
            try:
                plan = self._generate_on_device_plan(instruction, ui_state)
                if plan:
                    logger.info(f"Plan generated using on-device model (backend: {self.on_device_runtime.backend})")
                    return plan
            except Exception as e:
                logger.warning(f"On-device planning failed: {e}, falling back to rule-based")
        
        # Fallback to rule-based planning
        logger.info("Using rule-based planning fallback")
        return self._generate_rule_based_plan(instruction, ui_state)
    
    def _generate_cloud_plan(
        self,
        instruction: str,
        ui_state: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate plan using cloud LLM (e.g., OpenAI, Gemini).
        
        This is a placeholder - implement with actual LLM API calls.
        
        Args:
            instruction: Task instruction
            ui_state: UI state
            
        Returns:
            Action plan or None if unavailable
        """
        # Placeholder for cloud LLM integration
        # In production, this would call OpenAI/Gemini/Claude API
        logger.debug("Cloud LLM planning not implemented, using fallback")
        return None
    
    def _generate_on_device_plan(
        self,
        instruction: str,
        ui_state: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate plan using on-device planning model.
        
        Args:
            instruction: Task instruction
            ui_state: UI state
            
        Returns:
            Action plan or None if model inference fails
        """
        if self.on_device_runtime is None:
            return None
        
        try:
            # Prepare input for planning model
            # This is a simplified version - actual implementation would tokenize properly
            input_text = f"{instruction}\n{json.dumps(ui_state.get('ui_elements', []))}"
            
            # Convert to model input format (simplified)
            # Real implementation would use proper tokenization
            input_tensor = self._prepare_planning_input(input_text, ui_state)
            
            # Run inference
            output = self.on_device_runtime.run(input_tensor)
            
            # Decode output to action plan
            plan = self._decode_planning_output(output, instruction, ui_state)
            
            return plan
            
        except Exception as e:
            logger.error(f"On-device planning inference failed: {e}")
            return None
    
    def _prepare_planning_input(
        self,
        instruction: str,
        ui_state: Dict[str, Any]
    ) -> Any:
        """
        Prepare input tensor for planning model.
        
        Args:
            instruction: Task instruction
            ui_state: UI state
            
        Returns:
            Input tensor for model inference
        """
        import numpy as np
        
        # Simplified: create a feature vector from instruction + UI elements
        # Real implementation would use proper tokenization/embedding
        
        # Extract features from UI elements
        num_elements = len(ui_state.get('ui_elements', []))
        has_button = any(e.get('type') == 'button' for e in ui_state.get('ui_elements', []))
        has_input = any(e.get('type') == 'input' for e in ui_state.get('ui_elements', []))
        
        # Create simple feature vector
        features = [
            len(instruction) / 100.0,  # Normalized instruction length
            num_elements / 10.0,        # Normalized element count
            1.0 if has_button else 0.0,
            1.0 if has_input else 0.0,
        ]
        
        # Pad to expected input size (example: 512 features)
        while len(features) < 512:
            features.append(0.0)
        features = features[:512]
        
        # Reshape for model (batch_size=1, features=512)
        input_tensor = np.array([features], dtype=np.float32)
        
        return input_tensor
    
    def _decode_planning_output(
        self,
        output: Any,
        instruction: str,
        ui_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Decode model output to action plan.
        
        Args:
            output: Model output tensor
            instruction: Original instruction
            ui_state: UI state
            
        Returns:
            List of action dictionaries
        """
        import numpy as np
        
        # Simplified decoding - real implementation would parse model output properly
        # For now, use rule-based fallback based on instruction
        return self._generate_rule_based_plan(instruction, ui_state)
    
    def _generate_rule_based_plan(
        self,
        instruction: str,
        ui_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate plan using rule-based heuristics.
        
        Args:
            instruction: Task instruction
            ui_state: UI state
            
        Returns:
            List of action dictionaries
        """
        instruction_lower = instruction.lower()
        action_plan = []
        
        # Extract UI elements
        ui_elements = ui_state.get('ui_elements', [])
        element_ids = {e.get('id') for e in ui_elements if e.get('id')}
        
        # Login task
        if "login" in instruction_lower:
            if "username_field" in element_ids and "password_field" in element_ids:
                action_plan = [
                    {"action": "type_text", "target_id": "username_field", "value": "testuser"},
                    {"action": "type_text", "target_id": "password_field", "value": "testpassword"},
                ]
            if "login_button" in element_ids:
                action_plan.append({"action": "tap", "target_id": "login_button"})
            if "home_screen_element" in element_ids:
                action_plan.append({"action": "wait_for_displayed", "target_id": "home_screen_element"})
        
        # Navigation tasks
        elif "profile" in instruction_lower or "navigate" in instruction_lower:
            if "profile_button" in element_ids:
                action_plan.append({"action": "tap", "target_id": "profile_button"})
        
        # Tap actions
        elif "tap" in instruction_lower or "click" in instruction_lower:
            # Try to find matching element
            for element in ui_elements:
                element_text = element.get('text', '').lower()
                if any(word in element_text for word in instruction_lower.split()):
                    action_plan.append({"action": "tap", "target_id": element.get('id', 'element')})
                    break
        
        # Scroll actions
        elif "scroll" in instruction_lower or "swipe" in instruction_lower:
            direction = "down" if "down" in instruction_lower else "up"
            action_plan.append({"action": "swipe", "direction": direction})
        
        # Default: wait if no specific action found
        if not action_plan:
            action_plan.append({"action": "wait", "duration": 1})
        
        logger.debug(f"Rule-based plan: {json.dumps(action_plan, indent=2)}")
        return action_plan
    
    def generate_plan(self, instruction: str, ui_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Alias for generate_action_plan for backward compatibility."""
        return self.generate_action_plan(instruction, ui_state)
    
    def reflect_and_correct(
        self,
        instruction: str,
        ui_state: Dict[str, Any],
        failed_action: Dict[str, Any],
        error_message: str
    ) -> List[Dict[str, Any]]:
        """
        Reflect on failed action and generate corrective plan.
        
        Args:
            instruction: Original instruction
            ui_state: Current UI state
            failed_action: Action that failed
            error_message: Error message
            
        Returns:
            Corrective action plan
        """
        logger.info(f"Reflecting on failed action: {failed_action} (error: {error_message})")
        
        # Simple correction heuristics
        target_id = failed_action.get("target_id")
        action_type = failed_action.get("action")
        
        if "not displayed" in error_message.lower() or "not found" in error_message.lower():
            # Element not found - try to wait and retry
            return [
                {"action": "wait", "duration": 2},
                {"action": action_type, "target_id": target_id},
            ]
        elif "timeout" in error_message.lower():
            # Timeout - increase wait time
            return [
                {"action": "wait", "duration": 5},
                {"action": action_type, "target_id": target_id},
            ]
        else:
            # Unknown error - just retry once
            return [failed_action]


# Backward compatibility alias
LLMPlanner = LLMPlannerMobile

