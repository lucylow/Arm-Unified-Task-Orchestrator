"""
ARM AI Integration Module for AutoRL

This module provides ARM-optimized AI inference capabilities for on-device
mobile automation. It includes model loading, quantization, and performance
monitoring specifically designed for ARM architecture.
"""

from .arm_inference_engine import ARMInferenceEngine
from .device_detector import ARMDeviceDetector
from .performance_monitor import ARMPerformanceMonitor
from .model_loader import ARMModelLoader

__all__ = [
    'ARMInferenceEngine',
    'ARMDeviceDetector',
    'ARMPerformanceMonitor',
    'ARMModelLoader',
]

__version__ = '1.0.0'
__author__ = 'AutoRL ARM Edition Team'
