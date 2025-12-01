"""
ARM AI Integration Module for AutoRL

This module provides ARM-optimized AI inference capabilities for on-device
mobile automation. It includes model loading, quantization, and performance
monitoring specifically designed for ARM architecture.
"""

from .arm_inference_engine import ARMInferenceEngine, get_arm_inference_engine
from .device_detector import ARMDeviceDetector, get_arm_device_detector
from .performance_monitor import ARMPerformanceMonitor, get_arm_performance_monitor
from .model_loader import ARMModelLoader, get_arm_model_loader
from .arm_compute_integration import (
    ARMComputeOptimizer,
    ARMNeuralAccelerator,
    get_arm_compute_optimizer,
    get_arm_neural_accelerator,
)

__all__ = [
    'ARMInferenceEngine',
    'ARMDeviceDetector',
    'ARMPerformanceMonitor',
    'ARMModelLoader',
    'ARMComputeOptimizer',
    'ARMNeuralAccelerator',
    'get_arm_inference_engine',
    'get_arm_device_detector',
    'get_arm_performance_monitor',
    'get_arm_model_loader',
    'get_arm_compute_optimizer',
    'get_arm_neural_accelerator',
]

__version__ = '1.0.0'
__author__ = 'AutoRL ARM Edition Team'
