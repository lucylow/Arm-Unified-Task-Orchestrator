import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any

from plugins.base_plugin import BasePlugin


class VisionBoostPlugin(BasePlugin):
    """
    VisionBoostPlugin enhances the agent's visual perception.
    Converts image input -> descriptive text + embedding vectors.
    """

    def __init__(self):
        self.model_caption = None
        self.model_detector = None
        self.initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """Load lightweight mock models (can replace with real APIs)."""
        print("[VisionBoost] Initializing models...")
        # Mock initialization
        self.model_caption = lambda img: "A person sitting at a desk using a computer."
        self.model_detector = lambda img: [
            {"object": "person", "confidence": 0.97},
            {"object": "laptop", "confidence": 0.95}
        ]
        self.initialized = True
        print("[VisionBoost] Initialization complete.")

    def _decode_image(self, image_b64: str) -> Image.Image:
        """Helper: Decode base64 image to PIL Image."""
        img_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(img_bytes))

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts:
          input_data = {
              "frame_base64": "<base64>",
              "timestamp": float,
              "context": "observing screen"
          }

        Returns:
          {
              "caption": "...",
              "detected_objects": [...],
              "visual_embedding": np.ndarray (mocked)
          }
        """
        if not self.initialized:
            raise RuntimeError("VisionBoostPlugin must be initialized before use.")

        print("[VisionBoost] Processing frame...")

        # Decode base64 image
        img = self._decode_image(input_data["frame_base64"])

        # Run models
        caption = self.model_caption(img)
        detected_objects = self.model_detector(img)

        # Mock vector embedding
        visual_embedding = np.random.rand(1, 512)

        output = {
            "caption": caption,
            "detected_objects": detected_objects,
            "visual_embedding": visual_embedding.tolist(),
            "timestamp": input_data.get("timestamp"),
        }

        print(f"[VisionBoost] Output: {output['caption']}, {len(detected_objects)} objects.")
        return output

    def shutdown(self) -> None:
        """Release resources."""
        print("[VisionBoost] Shutting down...")
        self.initialized = False


def create(config: Dict[str, Any] = None):
    p = VisionBoostPlugin()
    if config:
        p.initialize(config)
    return p
