"""
Runtime abstraction layer for mobile AI inference on ARM devices.

Automatically selects the best available backend based on device capabilities:
Priority: ExecuTorch → ONNX (NNAPI/XNNPACK) → PyTorch Mobile (XNNPACK/NEON)

This abstraction allows the perception and planning modules to work seamlessly
across different mobile runtimes without code changes.

ARM Optimizations:
- ExecuTorch: Edge-optimized format with ARM NEON support
- ONNX Runtime: NNAPI acceleration on Android, XNNPACK CPU optimizations
- PyTorch Mobile: XNNPACK and ARM NEON SIMD instructions
"""

import os
import logging
import platform
from typing import Optional, Dict, Any, Union, List
import numpy as np
from pathlib import Path

from .executorch_wrapper import ExecuTorchRuntime
from .onnx_wrapper import OnnxRuntimeWrapper
from .pytorch_mobile_wrapper import PyTorchMobileWrapper

log = logging.getLogger("inference.runtime")


class Runtime:
    """
    Unified runtime interface for on-device AI inference.
    
    Automatically detects and uses the best available backend based on:
    - Hardware capabilities (NPU via NNAPI, GPU, CPU features)
    - Runtime availability (ExecuTorch, ONNX Runtime, PyTorch Mobile)
    - Model format compatibility
    """
    
    def __init__(self, backend: str, model_path: str, device_info: Optional[Dict[str, Any]] = None):
        """
        Initialize runtime.
        
        Args:
            backend: Backend name ('executorch', 'onnx', 'pytorch')
            model_path: Path to model file
            device_info: Optional device capabilities dictionary
        """
        self.backend = backend
        self.model_path = model_path
        self.device_info = device_info or {}
        self.impl = None
        
    @staticmethod
    def detect_arm_capabilities() -> Dict[str, Any]:
        """
        Detect ARM-specific capabilities and optimizations.
        
        Returns:
            Dictionary with ARM capabilities (NEON, NNAPI, etc.)
        """
        capabilities = {
            'platform': platform.machine().lower(),
            'is_arm': False,
            'has_neon': False,
            'has_nnapi': False,
            'cpu_count': os.cpu_count() or 1,
        }
        
        # Check if running on ARM architecture
        arch = platform.machine().lower()
        if 'arm' in arch or 'aarch64' in arch:
            capabilities['is_arm'] = True
            capabilities['has_neon'] = True  # ARMv7+ and AArch64 have NEON
            log.info(f"ARM architecture detected: {arch}")
        
        # Check for NNAPI (Android Neural Networks API)
        # This is typically available on Android devices
        if os.getenv('ANDROID_ROOT') or os.getenv('ANDROID_DATA'):
            capabilities['has_nnapi'] = True
            log.info("Android environment detected, NNAPI may be available")
        
        return capabilities
    
    @staticmethod
    def detect_best_backend(model_path: Optional[str] = None, device_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Detect the best available backend for the current platform.
        
        Priority order:
        1. ExecuTorch (if available and model format matches) - Best for ARM edge devices
        2. ONNX Runtime (if NNAPI/XNNPACK available) - Good ARM optimization
        3. PyTorch Mobile (fallback, always available on desktop) - ARM NEON support
        
        Args:
            model_path: Optional model path to check format compatibility
            device_info: Optional device capabilities dictionary
            
        Returns:
            Backend name: 'executorch', 'onnx', or 'pytorch'
        """
        # Merge device info with detected capabilities
        if device_info is None:
            device_info = {}
        arm_caps = Runtime.detect_arm_capabilities()
        device_info = {**arm_caps, **device_info}
        
        # Check model format if path provided
        if model_path:
            model_path = Path(model_path)
            if model_path.suffix == '.pte':
                # ExecuTorch format
                if ExecuTorchRuntime.is_available():
                    log.info("Model format is .pte, preferring ExecuTorch (ARM-optimized)")
                    return "executorch"
            elif model_path.suffix == '.onnx':
                # ONNX format
                if OnnxRuntimeWrapper.is_available():
                    log.info("Model format is .onnx, preferring ONNX Runtime (NNAPI/XNNPACK)")
                    return "onnx"
            elif model_path.suffix in ['.pt', '.ptl']:
                # PyTorch Mobile format
                if PyTorchMobileWrapper.is_available():
                    log.info("Model format is .pt/.ptl, preferring PyTorch Mobile (ARM NEON)")
                    return "pytorch"
        
        # Priority 1: ExecuTorch (best for on-device LLM/policy on ARM)
        if ExecuTorchRuntime.is_available():
            log.debug("ExecuTorch is available (ARM edge-optimized)")
            return "executorch"
        
        # Priority 2: ONNX Runtime (good ARM optimization via NNAPI/XNNPACK)
        if OnnxRuntimeWrapper.is_available():
            if device_info.get('has_nnapi'):
                log.debug("ONNX Runtime is available with NNAPI acceleration")
            else:
                log.debug("ONNX Runtime is available (XNNPACK CPU optimizations)")
            return "onnx"
        
        # Priority 3: PyTorch Mobile (fallback, ARM NEON support)
        if PyTorchMobileWrapper.is_available():
            if device_info.get('has_neon'):
                log.debug("PyTorch Mobile is available (ARM NEON optimized)")
            else:
                log.debug("PyTorch Mobile is available (fallback)")
            return "pytorch"
        
        # No runtime available
        raise RuntimeError(
            "No compatible inference runtime found. "
            "Please install one of: ExecuTorch, ONNX Runtime, or PyTorch"
        )
    
    @classmethod
    def load(
        cls,
        model_path: str,
        prefer: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None
    ) -> 'Runtime':
        """
        Load a model using the best available runtime.
        
        Args:
            model_path: Path to model file
            prefer: Preferred backend ('executorch', 'onnx', 'pytorch'), None for auto-detect
            device_info: Optional device capabilities dictionary
            
        Returns:
            Runtime instance ready for inference
            
        Example:
            >>> runtime = Runtime.load("model_quant.pt")
            >>> output = runtime.run(input_tensor)
        """
        model_path = str(Path(model_path).expanduser().resolve())
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Merge device info with detected ARM capabilities
        if device_info is None:
            device_info = {}
        arm_caps = cls.detect_arm_capabilities()
        device_info = {**arm_caps, **device_info}
        
        # Detect or use preferred backend
        backend = prefer or cls.detect_best_backend(model_path, device_info)
        log.info(f"Loading model with backend: {backend} (path: {model_path})")
        if device_info.get('is_arm'):
            log.info(f"ARM optimizations enabled: NEON={device_info.get('has_neon')}, "
                    f"NNAPI={device_info.get('has_nnapi')}")
        
        # Initialize appropriate wrapper
        if backend == "executorch":
            impl = ExecuTorchRuntime(model_path, device_info)
        elif backend == "onnx":
            impl = OnnxRuntimeWrapper(model_path, device_info)
        elif backend == "pytorch":
            impl = PyTorchMobileWrapper(model_path, device_info)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Load model
        impl.load()
        
        # Create and return runtime instance
        r = cls(backend, model_path, device_info)
        r.impl = impl
        return r
    
    def run(
        self,
        input_tensor: Union[np.ndarray, Any],
        **kwargs
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Input tensor (numpy array or backend-specific tensor)
            **kwargs: Additional backend-specific arguments
            
        Returns:
            Inference output (numpy array or backend-specific format)
        """
        if self.impl is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        return self.impl.run(input_tensor, **kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get runtime and model information.
        
        Returns:
            Dictionary with backend, model path, and device info
        """
        info = {
            'backend': self.backend,
            'model_path': str(self.model_path),
            'device_info': self.device_info,
        }
        
        if self.impl:
            impl_info = self.impl.get_info()
            info.update(impl_info)
        
        return info
    
    def warmup(self, input_shape: tuple = (1, 3, 224, 224), num_runs: int = 3):
        """
        Warmup the model with dummy inputs to optimize first inference.
        
        Args:
            input_shape: Shape of dummy input (batch, channels, height, width)
            num_runs: Number of warmup runs
        """
        if self.impl is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        log.info(f"Warming up model with {num_runs} runs (input shape: {input_shape})")
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run warmup iterations
        for i in range(num_runs):
            try:
                _ = self.run(dummy_input)
            except Exception as e:
                log.warning(f"Warmup run {i+1} failed: {e}")
        
        log.info("Warmup complete")
    
    def benchmark(
        self,
        input_shape: tuple = (1, 3, 224, 224),
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_shape: Shape of input tensor
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs before benchmarking
            
        Returns:
            Dictionary with latency statistics (mean, min, max, p50, p90, p99)
        """
        import time
        
        if self.impl is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        log.info(f"Benchmarking {num_runs} runs (warmup: {warmup_runs})")
        
        # Warmup
        self.warmup(input_shape, warmup_runs)
        
        # Create input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Benchmark
        latencies = []
        for i in range(num_runs):
            start = time.perf_counter()
            _ = self.run(dummy_input)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        latencies.sort()
        stats = {
            'mean_ms': sum(latencies) / len(latencies),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'p50_ms': latencies[len(latencies) // 2],
            'p90_ms': latencies[int(len(latencies) * 0.9)],
            'p99_ms': latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
        }
        
        log.info(f"Benchmark complete: mean={stats['mean_ms']:.2f}ms, p50={stats['p50_ms']:.2f}ms")
        return stats
    
    def get_arm_info(self) -> Dict[str, Any]:
        """
        Get ARM-specific runtime information.
        
        Returns:
            Dictionary with ARM capabilities and optimizations
        """
        info = self.get_info()
        arm_info = {
            'arm_platform': info.get('device_info', {}).get('is_arm', False),
            'neon_support': info.get('device_info', {}).get('has_neon', False),
            'nnapi_support': info.get('device_info', {}).get('has_nnapi', False),
            'backend_arm_optimized': self.backend in ['executorch', 'onnx', 'pytorch'],
        }
        
        # Backend-specific ARM optimizations
        if self.backend == 'executorch':
            arm_info['optimizations'] = ['ARM NEON', 'Edge-optimized format', 'Quantization support']
        elif self.backend == 'onnx':
            arm_info['optimizations'] = ['NNAPI (if available)', 'XNNPACK CPU', 'INT8 quantization']
        elif self.backend == 'pytorch':
            arm_info['optimizations'] = ['ARM NEON', 'XNNPACK', 'TorchScript optimization']
        
        return arm_info

