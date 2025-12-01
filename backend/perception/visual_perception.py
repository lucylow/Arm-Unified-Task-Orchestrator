
import os
from PIL import Image
import pytesseract
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from PIL import ImageDraw # Added for MockDriver

# Mock function for UI element detection (replace with actual ML model integration)
def detect_ui_elements(image_path: str) -> List[Dict[str, Any]]:
    """Simulates UI element detection, returning mock bounding boxes and labels."""
    # In a real scenario, this would use an ML model (e.g., YOLO, custom CNN)
    # to detect buttons, input fields, text, etc.
    print(f"[Perception] Simulating UI element detection for {image_path}")
    mock_elements = [
        {"text": "Username", "bbox": [50, 100, 150, 120], "type": "label"},
        {"text": "", "bbox": [160, 95, 300, 125], "type": "input", "id": "username_field"},
        {"text": "Password", "bbox": [50, 150, 150, 170], "type": "label"},
        {"text": "", "bbox": [160, 145, 300, 175], "type": "input", "id": "password_field"},
        {"text": "Login", "bbox": [100, 200, 250, 230], "type": "button", "id": "login_button"},
        {"text": "Home", "bbox": [10, 10, 50, 30], "type": "button", "id": "home_button"},
        {"text": "Profile", "bbox": [300, 10, 380, 30], "type": "button", "id": "profile_button"},
        {"text": "Welcome to Dashboard", "bbox": [50, 50, 350, 80], "type": "text", "id": "home_screen_element"},
    ]
    return mock_elements

def perform_ocr(image_path: str) -> str:
    """Performs OCR on the given image path and returns the extracted text."""
    print(f"[Perception] Performing OCR on {image_path}")
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        print(f"[Perception] OCR failed: {e}. Returning empty string.")
        return ""

class VisualPerception:
    """Handles visual perception tasks like OCR and UI element detection."""
    def __init__(self, screenshot_dir: str = "./screenshots"):
        self.screenshot_dir = screenshot_dir
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def capture_and_analyze(self, driver) -> Dict[str, Any]:
        """Captures a screenshot, performs OCR and UI element detection, and returns structured UI state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        
        print(f"[Perception] Capturing screenshot to {screenshot_path}")
        driver.save_screenshot(screenshot_path)

        full_text = perform_ocr(screenshot_path)
        ui_elements = detect_ui_elements(screenshot_path)

        ui_state = {
            "timestamp": timestamp,
            "screenshot_path": screenshot_path,
            "full_text_ocr": full_text,
            "ui_elements": ui_elements,
            "device_info": driver.get_capabilities()
        }
        print(f"[Perception] Analysis complete. Found {len(ui_elements)} UI elements.")
        return ui_state

    def get_element_locator(self, ui_state: Dict[str, Any], element_id: str) -> Optional[Tuple[str, str]]:
        """Helper to get Appium locator from UI state based on a logical ID."""
        for element in ui_state.get("ui_elements", []):
            if element.get("id") == element_id:
                return ("accessibility id", element_id)
        return None

