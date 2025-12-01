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
    
    def __init__(self, models_dir: str = "models/arm"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ARM Model Loader initialized (models_dir={self.models_dir})")
    
    def load_pytorch_mobile_model(self, model_path: str, model_name: str) -> Any:
        """Load PyTorch Mobile model"""
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded (cached)")
            return self.loaded_models[model_name]
        
        try:
            import torch
            from torch.utils.mobile_optimizer import optimize_for_mobile
            
            full_path = self.models_dir / model_path
            if not full_path.exists():
                raise FileNotFoundError(f"Model file not found: {full_path}")
            
            logger.info(f"Loading PyTorch Mobile model: {model_name}")
            start_time = time.time()
            
            # Load model
            model = torch.jit.load(str(full_path))
            model.eval()
            
            load_time = (time.time() - start_time) * 1000
            
            # Cache model
            self.loaded_models[model_name] = model
            self.model_info[model_name] = {
                'type': 'pytorch_mobile',
                'path': str(full_path),
                'size_mb': full_path.stat().st_size / (1024 * 1024),
                'load_time_ms': load_time,
            }
            
            logger.info(f"Loaded '{model_name}' in {load_time:.1f}ms "
                       f"(size={self.model_info[model_name]['size_mb']:.1f}MB)")
            
            return model
            
        except ImportError:
            logger.error("PyTorch not available. Install with: pip install torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load PyTorch Mobile model '{model_name}': {e}")
            raise
    
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
        """Get loaded model by name"""
        return self.loaded_models.get(model_name)
    
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
        summary = f"""
ARM Model Loader Summary
========================
Models Directory: {self.models_dir}
Loaded Models: {len(self.loaded_models)}

"""
        
        if self.loaded_models:
            summary += "Loaded Models:\n"
            for name, info in self.model_info.items():
                summary += f"  - {name} ({info['type']}): {info['size_mb']:.1f}MB\n"
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
