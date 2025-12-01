"""plugins/vision_boost.py

Example plugin that demonstrates the plugin interface expected by
`src/plugins/registry.py`. This is a lightweight, paste-ready plugin that
enhances perception output by applying a configurable confidence threshold
and optionally adding synthetic detections (for demo purposes).

This module is intentionally small and well-commented so you can replace
the internals with a real model (YOLOv8/TensorRT/TFLite) later.
"""
from typing import Dict, Any, List
import logging
import random

logger = logging.getLogger("vision_boost")


class VisionBoost:
    """A minimal plugin class with an `on_perception` hook.

    The PluginRegistry will import the module and can instantiate the
    `VisionBoost` class using the `entrypoint` value from plugin YAML.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # demo threshold controls whether to keep low-confidence detections
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.6))

    def on_perception(self, ui_state: Dict[str, Any]) -> Dict[str, Any]:
        """Called after the core perception agent produces a ui_state dict.

        This hook may modify the ui_state in-place or return an enriched
        copy. Here we filter detections and optionally add a synthetic
        'highlight' detection to demonstrate plugin effects in the UI.
        """
        elements: List[Dict[str, Any]] = ui_state.get("elements", [])
        if not elements:
            # nothing to do
            return ui_state

        # Apply confidence filtering
        filtered = [el for el in elements if (el.get("confidence", 0.0) >= self.confidence_threshold)]

        # For demo: randomly add a visual hint detection 10% of the time
        if random.random() < 0.1:
            hint = {
                "id": "vision_boost_hint",
                "type": "overlay",
                "text": "hint: try tapping 'Send'",
                "bbox": [50, 50, 180, 120],
                "confidence": 0.99,
            }
            filtered.append(hint)

        new_state = dict(ui_state)
        new_state["elements"] = filtered
        logger.info("vision_boost applied: kept %d/%d elements", len(filtered), len(elements))
        return new_state


def create(config: Dict[str, Any] = None):
    """Factory used by dynamic loaders: returns an instance configured.
    The `PluginRegistry` can import module and call create() to get the plugin.
    """
    return VisionBoost(config=config)
