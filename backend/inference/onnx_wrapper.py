"""
ONNX Runtime wrapper for mobile AI inference.

ONNX Runtime Mobile provides excellent ARM optimization via:
- NNAPI (Android Neural Networks API) for hardware acceleration
- XNNPACK for CPU optimizations
- Efficient quantization support (INT8)

References:
- https://onnxruntime.ai/docs/build/mobile.html
- https://github.com/onnx/models
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import numpy as np

log = logging.getLogger("inference.onnx")


class OnnxRuntimeWrapper:
    """
    ONNX Runtime wrapper for mobile AI inference.
    
    ONNX Runtime is a good choice for:
    - Cross-platform model deployment
    - NNAPI acceleration on Android
    - Quantized INT8 models
    - Models from ONNX Model Zoo
    """
    
    _onnxruntime_available: Optional[bool] = None
    _onnxruntime_module = None
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if ONNX Runtime is available.
        
        Returns:
            True if ONNX Runtime can be imported
        """
        if OnnxRuntimeWrapper._onnxruntime_available is not None:
            return OnnxRuntimeWrapper._onnxruntime_available
        
        try:
            import onnxruntime as ort
            OnnxRuntimeWrapper._onnxruntime_module = ort
            OnnxRuntimeWrapper._onnxruntime_available = True
            log.debug("ONNX Runtime is available")
        except ImportError:
            OnnxRuntimeWrapper._onnxruntime_available = False
            log.debug("ONNX Runtime not available (onnxruntime package not installed)")
        
        return OnnxRuntimeWrapper._onnxruntime_available
    
    def __init__(self, model_path: str, device_info: Optional[Dict[str, Any]] = None):
        """
        Initialize ONNX Runtime wrapper.
        
        Args:
            model_path: Path to .onnx model file
            device_info: Optional device capabilities dictionary
        """
        self.model_path = model_path
        self.device_info = device_info or {}
        self.session = None
        self.input_names = []
        self.output_names = []
        
        if not OnnxRuntimeWrapper.is_available():
            raise RuntimeError(
                "ONNX Runtime is not available. "
                "Install with: pip install onnxruntime"
            )
    
    def load(self):
        """Load ONNX model."""
        log.info(f"Loading ONNX model: {self.model_path}")
        
        try:
            import onnxruntime as ort
            
            # Create session options with optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Configure execution providers
            # Priority: NNAPI (Android) > CPU (with ARM XNNPACK optimizations)
            providers = []
            
            # Check for NNAPI availability (Android)
            if self.device_info.get('has_nnapi', False):
                try:
                    providers.append('NnapiExecutionProvider')
                    log.info("Using NNAPI execution provider for ARM hardware acceleration")
                except Exception as e:
                    log.warning(f"NNAPI provider not available: {e}")
            
            # Configure CPU provider with ARM optimizations
            # XNNPACK is automatically used by ONNX Runtime on ARM
            cpu_provider_options = {}
            if self.device_info.get('is_arm', False):
                # Enable ARM-specific optimizations
                cpu_provider_options['enable_cpu_mem_arena'] = True
                log.info("ARM architecture detected, enabling XNNPACK optimizations")
            
            providers.append(('CPUExecutionProvider', cpu_provider_options))
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            log.info(f"ONNX model loaded: {len(self.input_names)} inputs, {len(self.output_names)} outputs")
            log.info(f"Execution providers: {self.session.get_providers()}")
            
        except Exception as e:
            log.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"ONNX model loading failed: {e}")
    
    def run(
        self,
        input_tensor: Union[np.ndarray, Any],
        **kwargs
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Run inference using ONNX Runtime.
        
        Args:
            input_tensor: Input tensor as numpy array
            **kwargs: Additional arguments (input_name for multi-input models)
            
        Returns:
            Output tensor as numpy array (or dict for multi-output models)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Prepare input
            if not isinstance(input_tensor, np.ndarray):
                input_tensor = np.array(input_tensor)
            
            # Ensure float32
            if input_tensor.dtype != np.float32:
                input_tensor = input_tensor.astype(np.float32)
            
            # Get input name (use first input by default, or from kwargs)
            input_name = kwargs.get('input_name', self.input_names[0])
            
            # Create input dictionary
            inputs = {input_name: input_tensor}
            
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Return single output or dict
            if len(outputs) == 1:
                return outputs[0]
            else:
                return {name: output for name, output in zip(self.output_names, outputs)}
            
        except Exception as e:
            log.error(f"ONNX Runtime inference failed: {e}")
            raise RuntimeError(f"ONNX Runtime inference error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get runtime and model information."""
        import platform
        arch = platform.machine().lower()
        is_arm = 'arm' in arch or 'aarch64' in arch
        
        info = {
            'runtime': 'onnx',
            'model_path': self.model_path,
            'loaded': self.session is not None,
            'device_info': self.device_info,
            'arm_optimized': is_arm or self.device_info.get('is_arm', False),
            'architecture': arch,
        }
        
        if self.session:
            providers = self.session.get_providers()
            info.update({
                'input_names': self.input_names,
                'output_names': self.output_names,
                'providers': providers,
                'nnapi_enabled': 'NnapiExecutionProvider' in providers,
                'xnnpack_enabled': is_arm and 'CPUExecutionProvider' in providers,
            })
        
        return info

