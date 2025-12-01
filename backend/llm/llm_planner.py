
import json
from typing import Dict, Any, List

class LLMPlanner:
    """Simulates an LLM-based planner that converts user instructions and UI state into action plans."""

    def __init__(self):
        # In a real scenario, this would initialize an LLM client (e.g., OpenAI, Gemini)
        pass

    def generate_action_plan(self, instruction: str, ui_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generates a simulated action plan based on user instruction and current UI state."""
        print(f"[LLMPlanner] Generating plan for instruction: '{instruction}' with UI state...")
        action_plan = []
        if "login" in instruction.lower():
            action_plan = [
                {"action": "type_text", "target_id": "username_field", "value": "testuser"},
                {"action": "type_text", "target_id": "password_field", "value": "testpassword"},
                {"action": "tap", "target_id": "login_button"},
                {"action": "wait_for_displayed", "target_id": "home_screen_element"},
            ]
        elif "profile" in instruction.lower() or "navigate" in instruction.lower():
            action_plan = [
                {"action": "tap", "target_id": "profile_button"},
            ]
        else:
            print("[LLMPlanner] No specific plan found for instruction. Returning empty plan.")

        print(f"[LLMPlanner] Generated plan: {json.dumps(action_plan, indent=2)}")
        return action_plan

    def generate_plan(self, instruction: str, ui_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Alias for generate_action_plan for backward compatibility."""
        return self.generate_action_plan(instruction, ui_state)

    def reflect_and_correct(self, instruction: str, ui_state: Dict[str, Any], failed_action: Dict[str, Any], error_message: str) -> List[Dict[str, Any]]:
        """Simulates LLM reflection to correct a failed action plan."""
        print(f"[LLMPlanner] Reflecting on failed action: {failed_action} with error: {error_message}")
        print("[LLMPlanner] Attempting to generate a corrective plan...")

        if failed_action.get("target_id") == "login_button" and "not displayed" in error_message.lower():
            print("[LLMPlanner] Suggesting re-tapping login button after a short delay.")
            return [
                {"action": "sleep", "value": 2},
                {"action": "tap", "target_id": "login_button"},
                {"action": "wait_for_displayed", "target_id": "home_screen_element"},
            ]
        else:
            print("[LLMPlanner] No specific corrective action simulated. Returning empty plan.")
            return []

