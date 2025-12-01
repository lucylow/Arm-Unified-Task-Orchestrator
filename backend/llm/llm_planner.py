
import json
from typing import Dict, Any, List, Optional
import time
import random

class LLMPlanner:
    """Enhanced LLM-based planner that converts user instructions and UI state into detailed action plans."""

    def __init__(self):
        # In a real scenario, this would initialize an LLM client (e.g., OpenAI, Gemini)
        self.plan_templates = self._initialize_plan_templates()
        pass

    def _initialize_plan_templates(self) -> Dict[str, Dict]:
        """Initialize plan templates for common tasks"""
        return {
            "login": {
                "steps": [
                    {"action": "type_text", "target_id": "username_field", "value": "testuser", "description": "Enter username", "confidence": 0.95},
                    {"action": "type_text", "target_id": "password_field", "value": "testpassword", "description": "Enter password", "confidence": 0.95},
                    {"action": "tap", "target_id": "login_button", "description": "Tap login button", "confidence": 0.90},
                    {"action": "wait_for_displayed", "target_id": "home_screen_element", "description": "Wait for home screen", "confidence": 0.85},
                ],
                "reasoning": "Standard login flow: enter credentials and authenticate",
                "estimated_duration": 5
            },
            "search": {
                "steps": [
                    {"action": "tap", "target_id": "search_field", "description": "Tap search field", "confidence": 0.92},
                    {"action": "type_text", "target_id": "search_field", "value": "{query}", "description": "Enter search query", "confidence": 0.90},
                    {"action": "tap", "target_id": "search_button", "description": "Submit search", "confidence": 0.88},
                ],
                "reasoning": "Search flow: activate search, enter query, and submit",
                "estimated_duration": 3
            },
            "navigation": {
                "steps": [
                    {"action": "tap", "target_id": "profile_button", "description": "Navigate to profile", "confidence": 0.90},
                ],
                "reasoning": "Direct navigation to target screen",
                "estimated_duration": 2
            }
        }

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

    def generate_detailed_plan(self, instruction: str, ui_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a detailed plan with reasoning, steps, and metadata for interactive visualization."""
        print(f"[LLMPlanner] Generating detailed plan for: '{instruction}'")
        
        instruction_lower = instruction.lower()
        plan_id = f"plan_{int(time.time())}"
        
        # Determine plan type and generate steps
        if "login" in instruction_lower or "sign in" in instruction_lower:
            template = self.plan_templates["login"]
            steps = [step.copy() for step in template["steps"]]
            reasoning = template["reasoning"]
            estimated_duration = template["estimated_duration"]
        elif "search" in instruction_lower:
            template = self.plan_templates["search"]
            steps = [step.copy() for step in template["steps"]]
            # Extract search query from instruction
            query = instruction.replace("search", "").replace("for", "").strip()
            for step in steps:
                if "{query}" in step.get("value", ""):
                    step["value"] = step["value"].replace("{query}", query)
            reasoning = f"Search flow: {template['reasoning']}"
            estimated_duration = template["estimated_duration"]
        elif "navigate" in instruction_lower or "go to" in instruction_lower or "open" in instruction_lower:
            template = self.plan_templates["navigation"]
            steps = [step.copy() for step in template["steps"]]
            reasoning = template["reasoning"]
            estimated_duration = template["estimated_duration"]
        else:
            # Generic plan generation
            steps = [
                {
                    "action": "analyze_screen",
                    "target_id": "current_screen",
                    "description": "Analyze current screen state",
                    "confidence": 0.85
                },
                {
                    "action": "tap",
                    "target_id": "primary_button",
                    "description": f"Execute: {instruction}",
                    "confidence": 0.75
                }
            ]
            reasoning = f"Generic plan for: {instruction}"
            estimated_duration = 3

        # Add step numbers and metadata
        for i, step in enumerate(steps):
            step["step_number"] = i + 1
            step["id"] = f"{plan_id}_step_{i+1}"
            if "confidence" not in step:
                step["confidence"] = round(random.uniform(0.75, 0.95), 2)

        # Calculate overall confidence
        overall_confidence = round(sum(s.get("confidence", 0.8) for s in steps) / len(steps), 2) if steps else 0.8

        detailed_plan = {
            "plan_id": plan_id,
            "instruction": instruction,
            "reasoning": reasoning,
            "steps": steps,
            "total_steps": len(steps),
            "estimated_duration_seconds": estimated_duration,
            "overall_confidence": overall_confidence,
            "generated_at": time.time(),
            "ui_state": ui_state or {},
            "alternatives": self._generate_alternatives(instruction, steps)
        }

        print(f"[LLMPlanner] Generated detailed plan with {len(steps)} steps, confidence: {overall_confidence}")
        return detailed_plan

    def _generate_alternatives(self, instruction: str, primary_steps: List[Dict]) -> List[Dict]:
        """Generate alternative plan approaches"""
        alternatives = []
        
        if len(primary_steps) > 1:
            # Alternative: fewer steps (more direct)
            alt_steps = primary_steps[:len(primary_steps)//2 + 1] if len(primary_steps) > 2 else primary_steps
            alternatives.append({
                "approach": "direct",
                "description": "More direct approach with fewer steps",
                "steps": alt_steps,
                "confidence": round(random.uniform(0.70, 0.85), 2)
            })
        
        # Alternative: more cautious (with additional verification)
        cautious_steps = []
        for step in primary_steps:
            cautious_steps.append(step)
            if step.get("action") in ["tap", "type_text"]:
                cautious_steps.append({
                    "action": "verify",
                    "target_id": step.get("target_id"),
                    "description": f"Verify {step.get('description', 'action')} completed",
                    "confidence": 0.90
                })
        
        if len(cautious_steps) > len(primary_steps):
            alternatives.append({
                "approach": "cautious",
                "description": "More cautious approach with verification steps",
                "steps": cautious_steps,
                "confidence": round(random.uniform(0.85, 0.95), 2)
            })

        return alternatives[:2]  # Return max 2 alternatives

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

    def get_plan_explanation(self, plan: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the plan"""
        if not plan or "steps" not in plan:
            return "No plan available"
        
        explanation = f"Plan for: {plan.get('instruction', 'Unknown task')}\n\n"
        explanation += f"Reasoning: {plan.get('reasoning', 'No reasoning provided')}\n\n"
        explanation += f"Steps ({plan.get('total_steps', 0)} total, ~{plan.get('estimated_duration_seconds', 0)}s):\n"
        
        for i, step in enumerate(plan.get("steps", []), 1):
            action = step.get("action", "unknown")
            desc = step.get("description", f"Step {i}")
            conf = step.get("confidence", 0.8)
            explanation += f"  {i}. {desc} ({action}) - Confidence: {conf*100:.0f}%\n"
        
        return explanation

