"""
Inference runtime abstraction layer for mobile AI on Arm devices.

Supports multiple backends:
- ExecuTorch (preferred for on-device LLM/policy)
- ONNX Runtime Mobile (NNAPI/XNNPACK optimized)
- PyTorch Mobile (XNNPACK/NEON optimized)
"""

from .runtime import Runtime
from .executorch_wrapper import ExecuTorchRuntime
from .onnx_wrapper import OnnxRuntimeWrapper
from .pytorch_mobile_wrapper import PyTorchMobileWrapper

__all__ = [
    'Runtime',
    'ExecuTorchRuntime',
    'OnnxRuntimeWrapper',
    'PyTorchMobileWrapper',
]

