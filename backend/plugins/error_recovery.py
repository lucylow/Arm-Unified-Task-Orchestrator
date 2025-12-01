import traceback
from typing import Dict, Any
from plugins.base_plugin import BasePlugin


class ErrorRecoveryPlugin(BasePlugin):
    """
    ErrorRecoveryPlugin automatically analyzes and recovers from errors.
    It provides structured recovery actions and contextual explanations.
    """

    def __init__(self):
        self.recovery_rules = {}
        self.error_log = []
        self.initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Load or define recovery rule base."""
        self.recovery_rules = config.get("recovery_rules", {
            "ConnectionError": "Retry the API call with exponential backoff.",
            "TimeoutError": "Increase timeout threshold and re-attempt once.",
            "ValueError": "Validate input format and sanitize user data.",
        })
        self.initialized = True
        print("[ErrorRecovery] Rules loaded:", self.recovery_rules)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts:
            {
                "error": Exception or str,
                "context": "Agent failed to parse UI element"
            }
        Returns:
            {
                "root_cause": str,
                "recovery_action": str,
                "confidence": float
            }
        """
        if not self.initialized:
            raise RuntimeError("ErrorRecoveryPlugin must be initialized before use.")

        error_obj = input_data.get("error")
        context = input_data.get("context", "")
        error_type = type(error_obj).__name__ if isinstance(error_obj, Exception) else str(error_obj)

        # Extract trace for debugging
        trace_info = traceback.format_exc(limit=2)
        print(f"[ErrorRecovery] Error captured: {error_type} | Context: {context}")

        # Find best recovery plan
        recovery_action = self.recovery_rules.get(
            error_type,
            "Escalate to human-in-the-loop review."
        )

        # Confidence scoring (mock)
        confidence = 0.9 if error_type in self.recovery_rules else 0.6

        recovery_report = {
            "root_cause": error_type,
            "context": context,
            "recovery_action": recovery_action,
            "confidence": confidence,
            "trace": trace_info,
        }

        self.error_log.append(recovery_report)
        print(f"[ErrorRecovery] Plan: {recovery_action}")

        return recovery_report

    def shutdown(self) -> None:
        print("[ErrorRecovery] Shutdown. Total errors logged:", len(self.error_log))
        self.initialized = False
