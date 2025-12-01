"""
PyTorch Mobile wrapper for mobile AI inference on ARM devices.

PyTorch Mobile provides:
- TorchScript model support (.pt, .ptl)
- Quantization (dynamic and static)
- XNNPACK and NEON optimizations on ARM
- Fallback option when other runtimes aren't available

ARM Optimizations:
- ARM NEON SIMD instructions for efficient vector operations
- XNNPACK for optimized CPU operations on ARM
- TorchScript JIT compilation with ARM-specific optimizations
- Quantized INT8 operations optimized for ARM Cortex-A processors

References:
- https://pytorch.org/mobile/
- https://pytorch.org/tutorials/recipes/mobile/interpreter_for_mobile_tutorial.html
"""

import os
import logging
import platform
from typing import Optional, Dict, Any, Union
import numpy as np

log = logging.getLogger("inference.pytorch_mobile")


class PyTorchMobileWrapper:
    """
    PyTorch Mobile runtime wrapper for mobile AI inference.
    
    PyTorch Mobile is a fallback option that should always be available.
    It provides good performance with quantized models on ARM CPUs via:
    - XNNPACK for efficient CPU operations
    - NEON SIMD instructions on ARM
    - TorchScript for optimized execution
    """
    
    _pytorch_available: Optional[bool] = None
    _torch_module = None
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if PyTorch is available.
        
        Returns:
            True if PyTorch can be imported
        """
        if PyTorchMobileWrapper._pytorch_available is not None:
            return PyTorchMobileWrapper._pytorch_available
        
        try:
            import torch
            PyTorchMobileWrapper._torch_module = torch
            PyTorchMobileWrapper._pytorch_available = True
            log.debug("PyTorch is available")
        except ImportError:
            PyTorchMobileWrapper._pytorch_available = False
            log.debug("PyTorch not available (torch package not installed)")
        
        return PyTorchMobileWrapper._pytorch_available
    
    def __init__(self, model_path: str, device_info: Optional[Dict[str, Any]] = None):
        """
        Initialize PyTorch Mobile wrapper.
        
        Args:
            model_path: Path to .pt or .ptl TorchScript model file
            device_info: Optional device capabilities dictionary
        """
        self.model_path = model_path
        self.device_info = device_info or {}
        self.model = None
        
        if not PyTorchMobileWrapper.is_available():
            raise RuntimeError(
                "PyTorch is not available. "
                "Install with: pip install torch"
            )
    
    def load(self):
        """Load PyTorch Mobile model."""
        log.info(f"Loading PyTorch Mobile model: {self.model_path}")
        
        try:
            import torch
            
            # Load TorchScript model
            self.model = torch.jit.load(self.model_path, map_location='cpu')
            self.model.eval()
            
            # Optimize for mobile if possible (ARM-optimized)
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                # Check ARM architecture
                arch = platform.machine().lower()
                is_arm = 'arm' in arch or 'aarch64' in arch
                
                if is_arm:
                    log.info(f"ARM architecture detected ({arch}), applying ARM-optimized mobile optimizations")
                else:
                    log.info("Applying mobile optimizations")
                
                self.model = optimize_for_mobile(self.model)
                log.info("Model optimized for mobile deployment (ARM NEON/XNNPACK enabled)")
            except Exception as e:
                log.debug(f"Mobile optimization not available: {e}")
            
            # Warmup run (optional, but recommended)
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                log.debug("Model warmup completed")
            except Exception as e:
                log.warning(f"Model warmup failed: {e}")
            
            log.info("PyTorch Mobile model loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load PyTorch Mobile model: {e}")
            raise RuntimeError(f"PyTorch Mobile model loading failed: {e}")
    
    def run(
        self,
        input_tensor: Union[np.ndarray, Any],
        **kwargs
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Run inference using PyTorch Mobile.
        
        Args:
            input_tensor: Input tensor as numpy array
            **kwargs: Additional arguments (not used currently)
            
        Returns:
            Output tensor as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            import torch
            
            # Convert numpy to torch tensor if needed
            if isinstance(input_tensor, np.ndarray):
                input_tensor = torch.from_numpy(input_tensor).float()
            elif not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Convert to numpy
            if isinstance(output, torch.Tensor):
                output = output.numpy()
            elif isinstance(output, (list, tuple)):
                output = [t.numpy() if isinstance(t, torch.Tensor) else t for t in output]
            
            return output
            
        except Exception as e:
            log.error(f"PyTorch Mobile inference failed: {e}")
            raise RuntimeError(f"PyTorch Mobile inference error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get runtime and model information."""
        arch = platform.machine().lower()
        is_arm = 'arm' in arch or 'aarch64' in arch
        
        info = {
            'runtime': 'pytorch_mobile',
            'model_path': self.model_path,
            'loaded': self.model is not None,
            'device_info': self.device_info,
            'arm_optimized': is_arm or self.device_info.get('is_arm', False),
            'architecture': arch,
            'neon_support': is_arm or self.device_info.get('has_neon', False),
            'xnnpack_enabled': True,  # PyTorch Mobile uses XNNPACK by default
        }
        
        if self.model:
            try:
                # Try to get model info
                if hasattr(self.model, 'graph'):
                    info['torchscript'] = True
            except Exception:
                pass
        
        return info

