"""
ARM Inference Engine

Main inference engine for running ARM-optimized AI models on mobile devices.
Integrates device detection, model loading, and performance monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

from .device_detector import get_arm_device_detector
from .model_loader import get_arm_model_loader
from .performance_monitor import get_arm_performance_monitor
from .arm_compute_integration import get_arm_compute_optimizer

logger = logging.getLogger(__name__)


class ARMInferenceEngine:
    """Main ARM inference engine for on-device AI"""
    
    def __init__(self, models_dir: str = "models/arm"):
        self.device_detector = get_arm_device_detector()
        self.model_loader = get_arm_model_loader(models_dir)
        self.performance_monitor = get_arm_performance_monitor()
        self.compute_optimizer = get_arm_compute_optimizer()
        
        self.is_arm = self.device_detector.is_arm_device()
        self.optimization_flags = self.device_detector.get_optimization_flags()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(interval=1.0)
        
        # Cache optimal batch size for common operations
        self._optimal_batch_size = None
        
        logger.info("ARM Inference Engine initialized")
        logger.info(f"ARM Device: {self.is_arm}")
        logger.info(f"Optimizations: {self.optimization_flags}")
    
    def load_models(self, model_configs: Dict[str, Dict]):
        """Load multiple models from configuration
        
        Args:
            model_configs: Dict of {model_name: {type, path}}
            Example:
                {
                    'vision_model': {'type': 'pytorch', 'path': 'vision_int8.pt'},
                    'planning_model': {'type': 'onnx', 'path': 'planning.onnx'},
                }
        """
        for model_name, config in model_configs.items():
            model_type = config['type']
            model_path = config['path']
            
            try:
                if model_type == 'pytorch':
                    self.model_loader.load_pytorch_mobile_model(model_path, model_name)
                elif model_type == 'onnx':
                    self.model_loader.load_onnx_model(model_path, model_name)
                elif model_type == 'tflite':
                    self.model_loader.load_tflite_model(model_path, model_name)
                else:
                    logger.warning(f"Unknown model type '{model_type}' for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
    
    def infer_vision_model(self, image: Image.Image, model_name: str = 'vision_model', 
                          use_optimization: bool = True) -> Dict:
        """Run vision model inference on image with ARM optimizations
        
        Args:
            image: PIL Image
            model_name: Name of vision model
            use_optimization: Whether to use ARM compute optimizations
            
        Returns:
            Dict with inference results
        """
        start_time = time.time()
        
        try:
            model = self.model_loader.get_model(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not loaded")
            
            model_info = self.model_loader.get_model_info(model_name)
            model_type = model_info['type']
            
            # Preprocess image with ARM-optimized operations
            input_tensor = self._preprocess_image(image, target_size=(224, 224), 
                                                 use_optimization=use_optimization)
            
            # Run inference based on model type
            if model_type == 'pytorch_mobile':
                output = self._infer_pytorch(model, input_tensor, use_optimization)
            elif model_type == 'onnx':
                output = self._infer_onnx(model, input_tensor, model_info, use_optimization)
            elif model_type == 'tflite':
                output = self._infer_tflite(model, input_tensor, model_info)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Record performance
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_inference(model_name, latency_ms)
            
            logger.info(f"Vision inference completed in {latency_ms:.1f}ms")
            
            return {
                'success': True,
                'output': output,
                'latency_ms': latency_ms,
                'model': model_name,
                'optimization_used': use_optimization,
            }
            
        except Exception as e:
            logger.error(f"Vision inference failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model_name,
            }
    
    def infer_vision_batch(self, images: List[Image.Image], model_name: str = 'vision_model') -> List[Dict]:
        """Run batch vision inference with ARM optimizations
        
        Args:
            images: List of PIL Images
            model_name: Name of vision model
            
        Returns:
            List of inference result dicts
        """
        if not images:
            return []
        
        start_time = time.time()
        
        try:
            model = self.model_loader.get_model(model_name)
            if model is None:
                raise ValueError(f"Model '{model_name}' not loaded")
            
            model_info = self.model_loader.get_model_info(model_name)
            model_type = model_info['type']
            
            # Preprocess all images
            input_tensors = [self._preprocess_image(img, target_size=(224, 224)) for img in images]
            
            # Use compute optimizer for batch processing
            if model_type == 'pytorch_mobile':
                def inference_fn(tensor):
                    return self._infer_pytorch(model, tensor, use_optimization=True)
                
                outputs = self.compute_optimizer.optimize_batch_inference(input_tensors, inference_fn)
            else:
                # Fallback to sequential for other model types
                outputs = []
                for tensor in input_tensors:
                    if model_type == 'onnx':
                        output = self._infer_onnx(model, tensor, model_info, use_optimization=True)
                    elif model_type == 'tflite':
                        output = self._infer_tflite(model, tensor, model_info)
                    else:
                        raise ValueError(f"Unsupported model type for batch: {model_type}")
                    outputs.append(output)
            
            # Record performance
            total_latency_ms = (time.time() - start_time) * 1000
            avg_latency_ms = total_latency_ms / len(images)
            self.performance_monitor.record_inference(model_name, avg_latency_ms)
            
            logger.info(f"Batch vision inference: {len(images)} images in {total_latency_ms:.1f}ms "
                       f"(avg: {avg_latency_ms:.1f}ms per image)")
            
            return [
                {
                    'success': True,
                    'output': output,
                    'latency_ms': avg_latency_ms,
                    'model': model_name,
                    'batch_index': i,
                }
                for i, output in enumerate(outputs)
            ]
            
        except Exception as e:
            logger.error(f"Batch vision inference failed: {e}")
            return [
                {
                    'success': False,
                    'error': str(e),
                    'model': model_name,
                }
                for _ in images
            ]
    
    def infer_planning_model(self, input_text: str, model_name: str = 'planning_model') -> Dict:
        """Run planning model inference
        
        Args:
            input_text: Input text/prompt
            model_name: Name of planning model
            
        Returns:
            Dict with inference results
        """
        start_time = time.time()
        
        try:
            model = self.model_loader.get_model(model_name)
            if model is None:
                # Fallback to rule-based planning
                logger.warning(f"Model '{model_name}' not loaded, using rule-based fallback")
                return self._rule_based_planning(input_text)
            
            model_info = self.model_loader.get_model_info(model_name)
            model_type = model_info['type']
            
            # Tokenize input (simplified - would need proper tokenizer)
            input_ids = self._tokenize_text(input_text)
            
            # Run inference based on model type
            if model_type == 'pytorch_mobile':
                output = self._infer_pytorch(model, input_ids)
            elif model_type == 'onnx':
                output = self._infer_onnx(model, input_ids, model_info)
            else:
                raise ValueError(f"Unsupported model type for planning: {model_type}")
            
            # Decode output (simplified)
            plan = self._decode_planning_output(output)
            
            # Record performance
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_inference(model_name, latency_ms)
            
            logger.info(f"Planning inference completed in {latency_ms:.1f}ms")
            
            return {
                'success': True,
                'plan': plan,
                'latency_ms': latency_ms,
                'model': model_name,
            }
            
        except Exception as e:
            logger.error(f"Planning inference failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model_name,
            }
    
    def _preprocess_image(self, image: Image.Image, target_size=(224, 224), 
                         use_optimization: bool = True) -> np.ndarray:
        """Preprocess image for model input with ARM optimizations"""
        # Resize
        image = image.resize(target_size, Image.BILINEAR)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        
        # Normalize (ImageNet stats) - use optimized operations if enabled
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if use_optimization and self.compute_optimizer.neon_available:
            # Use vectorized operations that leverage NEON
            img_array = img_array / 255.0
            img_array = (img_array - mean) / std
        else:
            img_array = (img_array / 255.0 - mean) / std
        
        # Transpose to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Align memory for optimal NEON access
        if use_optimization:
            img_array = self.compute_optimizer._align_memory(img_array)
        
        return img_array
    
    def _infer_pytorch(self, model, input_tensor, use_optimization: bool = True):
        """Run PyTorch model inference with ARM optimizations"""
        import torch
        
        # Convert to torch tensor
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).float()
        
        # Set thread count for ARM multi-core
        if use_optimization:
            torch.set_num_threads(self.compute_optimizer.cpu_count)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert back to numpy
        if isinstance(output, torch.Tensor):
            output = output.numpy()
        
        return output
    
    def _infer_onnx(self, session, input_tensor, model_info, use_optimization: bool = True):
        """Run ONNX model inference with ARM optimizations"""
        input_name = model_info['input_names'][0]
        output_names = model_info['output_names']
        
        # Ensure correct dtype and alignment
        if isinstance(input_tensor, np.ndarray):
            input_tensor = input_tensor.astype(np.float32)
            if use_optimization:
                input_tensor = self.compute_optimizer._align_memory(input_tensor)
        
        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _infer_tflite(self, interpreter, input_tensor, model_info):
        """Run TFLite model inference"""
        input_details = model_info['input_details'][0]
        output_details = model_info['output_details'][0]
        
        # Set input tensor
        interpreter.set_tensor(input_details['index'], input_tensor.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output = interpreter.get_tensor(output_details['index'])
        
        return output
    
    def _tokenize_text(self, text: str) -> np.ndarray:
        """Simple tokenization (placeholder - would use proper tokenizer)"""
        # This is a placeholder - real implementation would use proper tokenizer
        tokens = [ord(c) % 256 for c in text[:512]]  # Simple char-level
        tokens = tokens + [0] * (512 - len(tokens))  # Pad to 512
        return np.array([tokens], dtype=np.int64)
    
    def _decode_planning_output(self, output: np.ndarray) -> List[Dict]:
        """Decode planning model output to action plan"""
        # This is a placeholder - real implementation would decode properly
        # For now, return a simple plan structure
        return [
            {'action': 'tap', 'target': 'button', 'confidence': 0.95},
            {'action': 'swipe', 'direction': 'up', 'confidence': 0.87},
        ]
    
    def _rule_based_planning(self, input_text: str) -> Dict:
        """Fallback rule-based planning when model not available"""
        logger.info("Using rule-based planning fallback")
        
        # Simple keyword-based planning
        plan = []
        text_lower = input_text.lower()
        
        if 'open' in text_lower:
            app_name = text_lower.split('open')[-1].strip().split()[0]
            plan.append({'action': 'launch_app', 'app': app_name, 'confidence': 0.9})
        
        if 'tap' in text_lower or 'click' in text_lower:
            plan.append({'action': 'tap', 'target': 'element', 'confidence': 0.85})
        
        if 'scroll' in text_lower or 'swipe' in text_lower:
            direction = 'down' if 'down' in text_lower else 'up'
            plan.append({'action': 'swipe', 'direction': direction, 'confidence': 0.8})
        
        if not plan:
            plan.append({'action': 'wait', 'duration': 1, 'confidence': 0.5})
        
        return {
            'success': True,
            'plan': plan,
            'latency_ms': 5.0,  # Rule-based is very fast
            'model': 'rule_based_fallback',
        }
    
    def get_device_info(self) -> Dict:
        """Get ARM device information"""
        return self.device_detector.get_device_info()
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return self.performance_monitor.get_summary()
    
    def get_loaded_models_info(self) -> Dict:
        """Get information about loaded models"""
        return self.model_loader.get_all_models_info()
    
    def get_status(self) -> Dict:
        """Get complete engine status"""
        return {
            'device': self.device_detector.get_device_info(),
            'optimization_flags': self.optimization_flags,
            'loaded_models': self.model_loader.get_all_models_info(),
            'performance': self.performance_monitor.get_summary(),
            'is_arm': self.is_arm,
        }
    
    def get_compute_optimization_report(self) -> str:
        """Get ARM compute optimization report"""
        return self.compute_optimizer.get_optimization_report()
    
    def shutdown(self):
        """Shutdown engine and cleanup resources"""
        self.performance_monitor.stop_monitoring()
        self.model_loader.unload_all_models()
        self.compute_optimizer.shutdown()
        logger.info("ARM Inference Engine shutdown complete")


# Singleton instance
_engine_instance: Optional[ARMInferenceEngine] = None


def get_arm_inference_engine(models_dir: str = "models/arm") -> ARMInferenceEngine:
    """Get singleton ARM inference engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ARMInferenceEngine(models_dir)
    return _engine_instance


if __name__ == '__main__':
    # Test inference engine
    engine = ARMInferenceEngine()
    
    print("Device Info:")
    print(engine.device_detector.get_device_summary())
    
    print("\nEngine Status:")
    status = engine.get_status()
    print(f"ARM Device: {status['is_arm']}")
    print(f"Loaded Models: {len(status['loaded_models'])}")
    
    # Test rule-based planning
    result = engine.infer_planning_model("Open Instagram and like 3 posts")
    print(f"\nPlanning Result: {result}")
    
    engine.shutdown()
