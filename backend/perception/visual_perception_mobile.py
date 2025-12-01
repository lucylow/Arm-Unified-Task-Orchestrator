"""
Enhanced visual perception module with on-device AI inference using runtime abstraction.

This module integrates with the runtime abstraction layer to support:
- ExecuTorch for on-device perception models
- ONNX Runtime for NNAPI-accelerated inference
- PyTorch Mobile as fallback

The module can run entirely on-device when models are available,
or fall back to mock detection for development.
"""

import os
import logging
from PIL import Image
import pytesseract
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import runtime abstraction
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


def preprocess_image_for_model(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image as numpy array (C, H, W, normalized)
    """
    # Resize
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32)
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array / 255.0 - mean) / std
    
    # Transpose to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def detect_ui_elements_with_model(image: Image.Image, runtime: Optional[Runtime]) -> List[Dict[str, Any]]:
    """
    Detect UI elements using on-device AI model.
    
    Args:
        image: PIL Image to analyze
        runtime: Runtime instance for inference (None for fallback)
        
    Returns:
        List of detected UI elements
    """
    if runtime is None:
        # Fallback to mock detection
        logger.debug("No runtime available, using mock UI element detection")
        return detect_ui_elements_mock(image)
    
    try:
        # Preprocess image
        input_tensor = preprocess_image_for_model(image)
        
        # Run inference
        logger.debug(f"Running inference with backend: {runtime.backend}")
        output = runtime.run(input_tensor)
        
        # Post-process output to extract UI elements
        # This is a simplified version - actual implementation would parse model output
        ui_elements = postprocess_model_output(output, image.size)
        
        logger.info(f"Detected {len(ui_elements)} UI elements using {runtime.backend}")
        return ui_elements
        
    except Exception as e:
        logger.warning(f"Model inference failed: {e}, falling back to mock detection")
        return detect_ui_elements_mock(image)


def postprocess_model_output(output: np.ndarray, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Post-process model output to extract UI element detections.
    
    This is a placeholder - actual implementation would depend on model architecture.
    For now, we convert raw output to a structured format.
    
    Args:
        output: Model output tensor
        image_size: Original image size (width, height)
        
    Returns:
        List of UI element dictionaries
    """
    # Simplified post-processing
    # In reality, this would decode bounding boxes, class predictions, etc.
    
    elements = []
    output = np.squeeze(output)
    
    # Example: if output is class predictions, convert to mock elements
    if len(output.shape) == 1 and len(output) == 10:
        # Assume output is class probabilities
        top_indices = np.argsort(output)[::-1][:5]
        
        element_types = ['button', 'input', 'text', 'image', 'label']
        for idx, class_idx in enumerate(top_indices):
            if output[class_idx] > 0.1:  # Threshold
                # Generate mock bbox
                x = 50 + (idx * 100) % 300
                y = 50 + (idx * 50) % 200
                elements.append({
                    'type': element_types[class_idx % len(element_types)],
                    'bbox': [x, y, x + 100, y + 40],
                    'confidence': float(output[class_idx]),
                    'id': f'element_{idx}'
                })
    
    # If no elements detected, return empty list (will fall back to mock)
    if not elements:
        logger.debug("Model output did not yield UI elements, using mock fallback")
        return detect_ui_elements_mock(Image.new('RGB', image_size))
    
    return elements


def detect_ui_elements_mock(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Mock UI element detection (fallback when model not available).
    
    Args:
        image: PIL Image (used for size info)
        
    Returns:
        Mock UI elements
    """
    width, height = image.size
    
    mock_elements = [
        {"text": "Username", "bbox": [50, 100, 150, 120], "type": "label", "id": "username_label"},
        {"text": "", "bbox": [160, 95, 300, 125], "type": "input", "id": "username_field"},
        {"text": "Password", "bbox": [50, 150, 150, 170], "type": "label", "id": "password_label"},
        {"text": "", "bbox": [160, 145, 300, 175], "type": "input", "id": "password_field"},
        {"text": "Login", "bbox": [100, 200, 250, 230], "type": "button", "id": "login_button"},
    ]
    
    return mock_elements


def perform_ocr(image_path: str) -> str:
    """
    Perform OCR on image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


class VisualPerceptionMobile:
    """
    Enhanced visual perception with on-device AI inference support.
    
    This class supports:
    - On-device model inference via runtime abstraction
    - Fallback to mock detection for development
    - OCR for text extraction
    - Runtime backend auto-detection
    """
    
    def __init__(
        self,
        screenshot_dir: str = "./screenshots",
        model_path: Optional[str] = None,
        prefer_backend: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize visual perception module.
        
        Args:
            screenshot_dir: Directory to save screenshots
            model_path: Path to perception model (None to use mock)
            prefer_backend: Preferred backend ('executorch', 'onnx', 'pytorch')
            device_info: Device capabilities dictionary
        """
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True)
        
        self.runtime = None
        self.device_info = device_info or {}
        
        # Load model if path provided and runtime available
        if model_path and RUNTIME_AVAILABLE:
            try:
                model_path = str(Path(model_path).expanduser().resolve())
                if os.path.exists(model_path):
                    self.runtime = Runtime.load(
                        model_path,
                        prefer=prefer_backend,
                        device_info=device_info
                    )
                    logger.info(f"Loaded perception model: {model_path} (backend: {self.runtime.backend})")
                else:
                    logger.warning(f"Model path not found: {model_path}, using mock detection")
            except Exception as e:
                logger.warning(f"Failed to load perception model: {e}, using mock detection")
        else:
            if not RUNTIME_AVAILABLE:
                logger.info("Runtime abstraction not available, using mock detection")
            else:
                logger.info("No model path provided, using mock detection")
    
    def capture_and_analyze(self, driver) -> Dict[str, Any]:
        """
        Capture screenshot and analyze UI state.
        
        Args:
            driver: Appium driver or mock driver
            
        Returns:
            Dictionary with UI state including elements and OCR text
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        
        logger.info(f"Capturing screenshot: {screenshot_path}")
        driver.save_screenshot(screenshot_path)
        
        # Load image for analysis
        image = Image.open(screenshot_path)
        
        # Perform OCR
        full_text = perform_ocr(screenshot_path)
        
        # Detect UI elements (with on-device model if available)
        ui_elements = detect_ui_elements_with_model(image, self.runtime)
        
        ui_state = {
            "timestamp": timestamp,
            "screenshot_path": screenshot_path,
            "full_text_ocr": full_text,
            "ui_elements": ui_elements,
            "device_info": driver.get_capabilities() if hasattr(driver, 'get_capabilities') else {},
            "inference_backend": self.runtime.backend if self.runtime else "mock",
        }
        
        logger.info(f"Analysis complete: {len(ui_elements)} UI elements detected")
        return ui_state
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a static image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with UI state
        """
        image = Image.open(image_path)
        
        # Perform OCR
        full_text = perform_ocr(image_path)
        
        # Detect UI elements
        ui_elements = detect_ui_elements_with_model(image, self.runtime)
        
        return {
            "screenshot_path": image_path,
            "full_text_ocr": full_text,
            "ui_elements": ui_elements,
            "inference_backend": self.runtime.backend if self.runtime else "mock",
        }
    
    def get_element_locator(self, ui_state: Dict[str, Any], element_id: str) -> Optional[Tuple[str, str]]:
        """Get Appium locator from UI state based on element ID."""
        for element in ui_state.get("ui_elements", []):
            if element.get("id") == element_id:
                return ("accessibility id", element_id)
        return None
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get information about loaded runtime."""
        if self.runtime:
            return self.runtime.get_info()
        return {"runtime": None, "backend": "mock"}


# Backward compatibility: alias for existing code
VisualPerception = VisualPerceptionMobile

