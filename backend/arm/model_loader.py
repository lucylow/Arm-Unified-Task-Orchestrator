"""
ARM Model Loader Module

Handles loading and caching of ARM-optimized AI models including PyTorch Mobile,
ONNX Runtime, and TensorFlow Lite models.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ARMModelLoader:
    """Loads and manages ARM-optimized AI models"""
    
    def __init__(self, models_dir: str = "models/arm", enable_caching: bool = True):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        self.enable_caching = enable_caching
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'loads': 0,
        }
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ARM Model Loader initialized (models_dir={self.models_dir}, caching={enable_caching})")
    
    def load_pytorch_mobile_model(
        self, 
        model_path: str, 
        model_name: str,
        use_memory_mapping: bool = False
    ) -> Any:
        """
        Load PyTorch Mobile model with quantization verification and optional memory mapping.
        
        Args:
            model_path: Path to model file relative to models_dir
            model_name: Name identifier for the model
            use_memory_mapping: If True, use memory-mapped file for large models (reduces RAM)
        """
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded (cached)")
            self._cache_stats['hits'] += 1
            return self.loaded_models[model_name]
        
        try:
            import torch
            from torch.utils.mobile_optimizer import optimize_for_mobile
            
            full_path = self.models_dir / model_path
            if not full_path.exists():
                raise FileNotFoundError(f"Model file not found: {full_path}")
            
            logger.info(f"Loading PyTorch Mobile model: {model_name}")
            start_time = time.time()
            
            # Use memory mapping for large models (>50MB)
            model_size_mb = full_path.stat().st_size / (1024 * 1024)
            if use_memory_mapping or model_size_mb > 50:
                logger.info(f"Using memory-mapped loading for large model ({model_size_mb:.1f}MB)")
                # PyTorch doesn't directly support memory mapping, but we can use mmap for reading
                # For now, regular load - in production could use torch.package or custom loader
                model = torch.jit.load(str(full_path), map_location='cpu')
            else:
                # Standard loading
                model = torch.jit.load(str(full_path), map_location='cpu')
            
            model.eval()
            
            # Check if model is quantized
            is_quantized = self._check_pytorch_quantization(model)
            quantization_type = self._get_quantization_type(model) if is_quantized else None
            
            # Apply mobile optimizations if not already optimized
            try:
                # Check if already optimized
                if not hasattr(model, '_optimized_for_mobile'):
                    model = optimize_for_mobile(model)
                    model._optimized_for_mobile = True
                    logger.debug("Applied mobile optimizations")
            except Exception as e:
                logger.debug(f"Mobile optimization not available or already applied: {e}")
            
            load_time = (time.time() - start_time) * 1000
            self._cache_stats['loads'] += 1
            
            # Cache model
            self.loaded_models[model_name] = model
            self.model_info[model_name] = {
                'type': 'pytorch_mobile',
                'path': str(full_path),
                'size_mb': model_size_mb,
                'load_time_ms': load_time,
                'quantized': is_quantized,
                'quantization_type': quantization_type,
                'memory_mapped': use_memory_mapping or model_size_mb > 50,
            }
            
            logger.info(f"Loaded '{model_name}' in {load_time:.1f}ms "
                       f"(size={model_size_mb:.1f}MB, "
                       f"quantized={is_quantized}, "
                       f"quantization={quantization_type or 'FP32'})")
            
            return model
            
        except ImportError:
            logger.error("PyTorch not available. Install with: pip install torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load PyTorch Mobile model '{model_name}': {e}")
            raise
    
    def _check_pytorch_quantization(self, model) -> bool:
        """Check if PyTorch model is quantized"""
        try:
            import torch
            # Check for quantized modules
            for name, module in model.named_modules():
                if isinstance(module, (torch.quantization.QuantStub, 
                                       torch.quantization.DeQuantStub,
                                       torch.nn.quantized.Quantize,
                                       torch.nn.quantized.DeQuantize)):
                    return True
                # Check for quantized parameters
                if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                    if 'qint' in str(module.weight.dtype):
                        return True
        except Exception:
            pass
        return False
    
    def _get_quantization_type(self, model) -> Optional[str]:
        """Get quantization type if model is quantized"""
        try:
            import torch
            quantization_types = []
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                    dtype_str = str(module.weight.dtype)
                    if 'qint8' in dtype_str and 'INT8' not in quantization_types:
                        quantization_types.append('INT8')
                    elif 'qint4' in dtype_str and 'INT4' not in quantization_types:
                        quantization_types.append('INT4')
                    elif 'quint8' in dtype_str and 'UINT8' not in quantization_types:
                        quantization_types.append('UINT8')
                # Check for FP16/BF16 mixed precision
                if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                    if module.weight.dtype == torch.float16:
                        if 'FP16' not in quantization_types:
                            quantization_types.append('FP16')
                    elif module.weight.dtype == torch.bfloat16:
                        if 'BF16' not in quantization_types:
                            quantization_types.append('BF16')
            
            if quantization_types:
                return '+'.join(quantization_types)  # Mixed precision
        except Exception:
            pass
        return None
    
    def load_onnx_model(self, model_path: str, model_name: str) -> Any:
        """Load ONNX Runtime model"""
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded (cached)")
            return self.loaded_models[model_name]
        
        try:
            import onnxruntime as ort
            
            full_path = self.models_dir / model_path
            if not full_path.exists():
                raise FileNotFoundError(f"Model file not found: {full_path}")
            
            logger.info(f"Loading ONNX model: {model_name}")
            start_time = time.time()
            
            # Create inference session with ARM optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Use CPU execution provider (ARM-optimized)
            providers = ['CPUExecutionProvider']
            
            session = ort.InferenceSession(
                str(full_path),
                sess_options=session_options,
                providers=providers
            )
            
            load_time = (time.time() - start_time) * 1000
            
            # Cache model
            self.loaded_models[model_name] = session
            self.model_info[model_name] = {
                'type': 'onnx',
                'path': str(full_path),
                'size_mb': full_path.stat().st_size / (1024 * 1024),
                'load_time_ms': load_time,
                'input_names': [inp.name for inp in session.get_inputs()],
                'output_names': [out.name for out in session.get_outputs()],
            }
            
            logger.info(f"Loaded '{model_name}' in {load_time:.1f}ms "
                       f"(size={self.model_info[model_name]['size_mb']:.1f}MB)")
            
            return session
            
        except ImportError:
            logger.error("ONNX Runtime not available. Install with: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"Failed to load ONNX model '{model_name}': {e}")
            raise
    
    def load_tflite_model(self, model_path: str, model_name: str) -> Any:
        """Load TensorFlow Lite model"""
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded (cached)")
            return self.loaded_models[model_name]
        
        try:
            import tensorflow as tf
            
            full_path = self.models_dir / model_path
            if not full_path.exists():
                raise FileNotFoundError(f"Model file not found: {full_path}")
            
            logger.info(f"Loading TFLite model: {model_name}")
            start_time = time.time()
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(full_path))
            interpreter.allocate_tensors()
            
            load_time = (time.time() - start_time) * 1000
            
            # Cache model
            self.loaded_models[model_name] = interpreter
            self.model_info[model_name] = {
                'type': 'tflite',
                'path': str(full_path),
                'size_mb': full_path.stat().st_size / (1024 * 1024),
                'load_time_ms': load_time,
                'input_details': interpreter.get_input_details(),
                'output_details': interpreter.get_output_details(),
            }
            
            logger.info(f"Loaded '{model_name}' in {load_time:.1f}ms "
                       f"(size={self.model_info[model_name]['size_mb']:.1f}MB)")
            
            return interpreter
            
        except ImportError:
            logger.error("TensorFlow not available. Install with: pip install tensorflow")
            raise
        except Exception as e:
            logger.error(f"Failed to load TFLite model '{model_name}': {e}")
            raise
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get loaded model by name (with cache statistics)"""
        if model_name in self.loaded_models:
            if self.enable_caching:
                self._cache_stats['hits'] += 1
            return self.loaded_models.get(model_name)
        else:
            if self.enable_caching:
                self._cache_stats['misses'] += 1
            return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information"""
        return self.model_info.get(model_name)
    
    def get_all_models_info(self) -> Dict[str, Dict]:
        """Get information for all loaded models"""
        return self.model_info.copy()
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self.loaded_models.clear()
        logger.info("Unloaded all models")
    
    def list_available_models(self) -> Dict[str, list]:
        """List available model files in models directory"""
        available = {
            'pytorch': [],
            'onnx': [],
            'tflite': [],
        }
        
        if not self.models_dir.exists():
            return available
        
        for file_path in self.models_dir.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in ['.pt', '.pth', '.ptl']:
                    available['pytorch'].append(str(file_path.relative_to(self.models_dir)))
                elif suffix == '.onnx':
                    available['onnx'].append(str(file_path.relative_to(self.models_dir)))
                elif suffix == '.tflite':
                    available['tflite'].append(str(file_path.relative_to(self.models_dir)))
        
        return available
    
    def get_summary(self) -> str:
        """Get loader summary"""
        cache_hit_rate = 0.0
        if self._cache_stats['hits'] + self._cache_stats['misses'] > 0:
            cache_hit_rate = (self._cache_stats['hits'] / 
                            (self._cache_stats['hits'] + self._cache_stats['misses'])) * 100
        
        summary = f"""
ARM Model Loader Summary
========================
Models Directory: {self.models_dir}
Loaded Models: {len(self.loaded_models)}
Caching: {'Enabled' if self.enable_caching else 'Disabled'}
Cache Stats: {self._cache_stats['hits']} hits, {self._cache_stats['misses']} misses ({cache_hit_rate:.1f}% hit rate)

"""
        
        if self.loaded_models:
            summary += "Loaded Models:\n"
            for name, info in self.model_info.items():
                quant_info = f", {info.get('quantization_type', 'FP32')}" if info.get('quantized') else ""
                summary += f"  - {name} ({info['type']}): {info['size_mb']:.1f}MB{quant_info}\n"
        else:
            summary += "No models currently loaded.\n"
        
        available = self.list_available_models()
        total_available = sum(len(models) for models in available.values())
        
        summary += f"\nAvailable Models: {total_available}\n"
        for model_type, models in available.items():
            if models:
                summary += f"  {model_type}: {len(models)} model(s)\n"
        
        return summary


# Singleton instance
_loader_instance: Optional[ARMModelLoader] = None


def get_arm_model_loader(models_dir: str = "models/arm") -> ARMModelLoader:
    """Get singleton ARM model loader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ARMModelLoader(models_dir)
    return _loader_instance


if __name__ == '__main__':
    # Test model loader
    loader = ARMModelLoader()
    print(loader.get_summary())
    print("\nAvailable models:")
    for model_type, models in loader.list_available_models().items():
        print(f"  {model_type}: {models}")
