"""
ARM Performance Monitoring Module

Monitors and tracks ARM-specific performance metrics including CPU usage,
memory footprint, inference latency, and power consumption.
"""

import time
import psutil
import threading
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    inference_latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    power_mw: Optional[float] = None
    temperature_c: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ARMPerformanceMonitor:
    """Monitors ARM device performance metrics"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.inference_times: Dict[str, List[float]] = {}
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Get process for monitoring
        self.process = psutil.Process()
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started ARM performance monitoring (interval={interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped ARM performance monitoring")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._capture_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def _capture_metrics(self) -> PerformanceMetrics:
        """Capture current performance metrics including power and thermal"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        # Get power consumption estimate (if available)
        power_mw = self._estimate_power_consumption(cpu_percent)
        
        # Get CPU temperature (if available)
        temperature_c = self._get_cpu_temperature()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            power_mw=power_mw,
            temperature_c=temperature_c,
        )
    
    def _estimate_power_consumption(self, cpu_percent: float) -> Optional[float]:
        """Estimate power consumption in milliwatts"""
        try:
            # Try to read from power monitoring (Android-specific)
            power_paths = [
                '/sys/class/power_supply/battery/power_now',
                '/sys/class/power_supply/battery/current_now',
            ]
            
            for path in power_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        power_uw = int(f.read().strip())
                        # Convert to milliwatts and scale by CPU usage
                        power_mw = (power_uw / 1000.0) * (cpu_percent / 100.0)
                        return power_mw
            
            # Fallback: estimate based on CPU usage
            # Typical ARM mobile CPU: ~2W at 100% load
            base_power_mw = 500  # Base power consumption
            cpu_power_mw = 2000 * (cpu_percent / 100.0)  # CPU power scaling
            return base_power_mw + cpu_power_mw
            
        except Exception:
            return None
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature in Celsius"""
        try:
            import os
            thermal_base = '/sys/class/thermal'
            if os.path.exists(thermal_base):
                for zone in os.listdir(thermal_base):
                    if zone.startswith('thermal_zone'):
                        type_path = os.path.join(thermal_base, zone, 'type')
                        temp_path = os.path.join(thermal_base, zone, 'temp')
                        
                        if os.path.exists(type_path) and os.path.exists(temp_path):
                            with open(type_path, 'r') as f:
                                zone_type = f.read().strip().lower()
                                if 'cpu' in zone_type or 'soc' in zone_type:
                                    with open(temp_path, 'r') as f:
                                        temp_millidegrees = int(f.read().strip())
                                        return temp_millidegrees / 1000.0
        except Exception:
            pass
        return None
    
    def is_thermal_throttling(self, threshold_c: float = 80.0) -> bool:
        """
        Check if CPU is thermal throttling.
        
        Args:
            threshold_c: Temperature threshold in Celsius (default: 80°C)
            
        Returns:
            True if temperature exceeds threshold
        """
        temp = self._get_cpu_temperature()
        if temp is None:
            return False
        return temp >= threshold_c
    
    def get_thermal_state(self) -> Dict:
        """
        Get current thermal state information.
        
        Returns:
            Dictionary with thermal state, temperature, and throttling status
        """
        temp = self._get_cpu_temperature()
        is_throttling = self.is_thermal_throttling() if temp else False
        
        return {
            'temperature_c': temp,
            'is_throttling': is_throttling,
            'threshold_c': 80.0,
            'status': 'throttling' if is_throttling else ('normal' if temp else 'unknown'),
        }
    
    def record_inference(self, model_name: str, latency_ms: float):
        """Record inference latency for a model"""
        with self._lock:
            if model_name not in self.inference_times:
                self.inference_times[model_name] = []
            
            self.inference_times[model_name].append(latency_ms)
            
            # Keep only recent measurements
            if len(self.inference_times[model_name]) > self.history_size:
                self.inference_times[model_name] = \
                    self.inference_times[model_name][-self.history_size:]
            
            # Also add to metrics history
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=self.process.cpu_percent(interval=0),
                memory_mb=self.process.memory_info().rss / (1024 * 1024),
                inference_latency_ms=latency_ms,
                model_name=model_name,
            )
            self.metrics_history.append(metrics)
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = self._capture_metrics()
        
        return {
            'cpu_percent': metrics.cpu_percent,
            'memory_mb': metrics.memory_mb,
            'timestamp': metrics.timestamp,
        }
    
    def get_inference_stats(self, model_name: str) -> Dict:
        """Get inference statistics for a model"""
        with self._lock:
            if model_name not in self.inference_times:
                return {
                    'model': model_name,
                    'count': 0,
                    'avg_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0,
                    'p50_ms': 0,
                    'p95_ms': 0,
                    'p99_ms': 0,
                }
            
            times = self.inference_times[model_name]
            sorted_times = sorted(times)
            count = len(times)
            
            return {
                'model': model_name,
                'count': count,
                'avg_ms': sum(times) / count if count > 0 else 0,
                'min_ms': min(times) if times else 0,
                'max_ms': max(times) if times else 0,
                'p50_ms': sorted_times[count // 2] if count > 0 else 0,
                'p95_ms': sorted_times[int(count * 0.95)] if count > 0 else 0,
                'p99_ms': sorted_times[int(count * 0.99)] if count > 0 else 0,
            }
    
    def get_all_inference_stats(self) -> Dict[str, Dict]:
        """Get inference statistics for all models"""
        with self._lock:
            return {
                model: self.get_inference_stats(model)
                for model in self.inference_times.keys()
            }
    
    def get_metrics_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get historical metrics"""
        with self._lock:
            history = list(self.metrics_history)
        
        if last_n:
            history = history[-last_n:]
        
        return [m.to_dict() for m in history]
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        with self._lock:
            if not self.metrics_history:
                return {
                    'cpu_avg': 0,
                    'cpu_max': 0,
                    'memory_avg_mb': 0,
                    'memory_max_mb': 0,
                    'power_avg_mw': 0,
                    'power_max_mw': 0,
                    'temperature_avg_c': 0,
                    'temperature_max_c': 0,
                    'inference_stats': {},
                }
            
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            memory_values = [m.memory_mb for m in self.metrics_history]
            power_values = [m.power_mw for m in self.metrics_history if m.power_mw is not None]
            temp_values = [m.temperature_c for m in self.metrics_history if m.temperature_c is not None]
            
            return {
                'cpu_avg': sum(cpu_values) / len(cpu_values),
                'cpu_max': max(cpu_values),
                'memory_avg_mb': sum(memory_values) / len(memory_values),
                'memory_max_mb': max(memory_values),
                'power_avg_mw': sum(power_values) / len(power_values) if power_values else 0,
                'power_max_mw': max(power_values) if power_values else 0,
                'temperature_avg_c': sum(temp_values) / len(temp_values) if temp_values else 0,
                'temperature_max_c': max(temp_values) if temp_values else 0,
                'inference_stats': self.get_all_inference_stats(),
                'samples': len(self.metrics_history),
            }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics_history.clear()
            self.inference_times.clear()
        logger.info("Reset ARM performance metrics")
    
    def predict_performance(self, model_name: str, lookahead_samples: int = 10) -> Dict[str, Any]:
        """
        Predict future performance based on historical data.
        
        Uses simple linear regression on recent latency measurements to predict
        future inference times and identify performance trends.
        
        Args:
            model_name: Name of the model to predict for
            lookahead_samples: Number of future samples to predict
            
        Returns:
            Dictionary with predictions and trend information
        """
        with self._lock:
            if model_name not in self.inference_times or len(self.inference_times[model_name]) < 5:
                return {
                    'model': model_name,
                    'predicted_avg_ms': None,
                    'trend': 'insufficient_data',
                    'trend_strength': 0.0,
                    'warnings': ['Insufficient historical data for prediction']
                }
            
            times = self.inference_times[model_name]
            recent_times = times[-20:]  # Use last 20 samples
            
            # Simple linear regression to detect trend
            x = np.arange(len(recent_times))
            y = np.array(recent_times)
            
            # Calculate trend
            coeffs = np.polyfit(x, y, 1)  # Linear fit
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Predict next values
            future_x = np.arange(len(recent_times), len(recent_times) + lookahead_samples)
            predicted_values = slope * future_x + intercept
            
            # Determine trend
            if abs(slope) < 0.5:
                trend = 'stable'
            elif slope > 0:
                trend = 'degrading'
            else:
                trend = 'improving'
            
            # Calculate trend strength (R²)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            warnings = []
            if trend == 'degrading' and abs(slope) > 1.0:
                warnings.append('Performance degradation detected - consider optimization')
            if predicted_values[-1] > np.mean(recent_times) * 1.5:
                warnings.append('Predicted latency significantly higher than average')
            
            return {
                'model': model_name,
                'predicted_avg_ms': float(np.mean(predicted_values)),
                'predicted_min_ms': float(np.min(predicted_values)),
                'predicted_max_ms': float(np.max(predicted_values)),
                'trend': trend,
                'trend_strength': float(r_squared),
                'slope_ms_per_sample': float(slope),
                'current_avg_ms': float(np.mean(recent_times)),
                'warnings': warnings,
            }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on current performance metrics.
        
        Returns:
            List of recommendation dictionaries with priority and actions
        """
        recommendations = []
        summary = self.get_summary()
        thermal_state = self.get_thermal_state()
        
        # Thermal recommendations
        if thermal_state.get('is_throttling', False):
            recommendations.append({
                'priority': 'high',
                'category': 'thermal',
                'issue': 'Thermal throttling active',
                'action': 'Reduce batch size, pause heavy inference, or improve cooling',
                'impact': 'Prevents performance degradation and device damage'
            })
        
        # Memory recommendations
        if summary['memory_max_mb'] > 800:
            recommendations.append({
                'priority': 'medium',
                'category': 'memory',
                'issue': f"High memory usage ({summary['memory_max_mb']:.1f} MB)",
                'action': 'Use memory-mapped model loading or reduce batch size',
                'impact': 'Prevents OOM errors and improves stability'
            })
        
        # CPU recommendations
        if summary['cpu_avg'] > 80:
            recommendations.append({
                'priority': 'medium',
                'category': 'cpu',
                'issue': f"High CPU usage ({summary['cpu_avg']:.1f}%)",
                'action': 'Reduce parallelism or use more efficient models',
                'impact': 'Improves battery life and reduces thermal load'
            })
        
        # Inference performance recommendations
        for model, stats in summary['inference_stats'].items():
            if stats['count'] > 10:
                if stats['avg_ms'] > 100:
                    recommendations.append({
                        'priority': 'low',
                        'category': 'performance',
                        'issue': f"Model {model} has high latency ({stats['avg_ms']:.1f}ms avg)",
                        'action': f'Consider quantizing {model} or using FP16',
                        'impact': 'Can reduce inference time by 2-4x'
                    })
                
                if stats['p95_ms'] > stats['avg_ms'] * 2:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'consistency',
                        'issue': f"Model {model} has inconsistent latency (high P95: {stats['p95_ms']:.1f}ms)",
                        'action': 'Check for resource contention or thermal throttling',
                        'impact': 'Improves predictable performance'
                    })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def get_arm_optimization_report(self) -> str:
        """Generate comprehensive ARM optimization report with predictions"""
        summary = self.get_summary()
        thermal_state = self.get_thermal_state()
        recommendations = self.get_optimization_recommendations()
        
        report = f"""
ARM Performance Report
======================
CPU Usage:
  Average: {summary['cpu_avg']:.1f}%
  Maximum: {summary['cpu_max']:.1f}%

Memory Usage:
  Average: {summary['memory_avg_mb']:.1f} MB
  Maximum: {summary['memory_max_mb']:.1f} MB

Power Consumption:
  Average: {summary['power_avg_mw']:.1f} mW
  Maximum: {summary['power_max_mw']:.1f} mW

Thermal:
  Average Temperature: {summary['temperature_avg_c']:.1f}°C
  Maximum Temperature: {summary['temperature_max_c']:.1f}°C
  Status: {thermal_state.get('status', 'unknown')}
  Throttling: {'Yes' if thermal_state.get('is_throttling', False) else 'No'}

Inference Performance:
"""
        
        for model, stats in summary['inference_stats'].items():
            report += f"""
  {model}:
    Inferences: {stats['count']}
    Average: {stats['avg_ms']:.1f} ms
    P50: {stats['p50_ms']:.1f} ms
    P95: {stats['p95_ms']:.1f} ms
    P99: {stats['p99_ms']:.1f} ms
    Min: {stats['min_ms']:.1f} ms
    Max: {stats['max_ms']:.1f} ms
"""
            
            # Add predictions
            prediction = self.predict_performance(model)
            if prediction.get('predicted_avg_ms'):
                report += f"    Predicted Avg: {prediction['predicted_avg_ms']:.1f} ms (trend: {prediction['trend']})\n"
        
        report += f"\nTotal Samples: {summary['samples']}\n"
        
        # Add recommendations
        if recommendations:
            report += "\nOptimization Recommendations:\n"
            for rec in recommendations[:5]:  # Top 5 recommendations
                report += f"\n  [{rec['priority'].upper()}] {rec['category'].upper()}: {rec['issue']}\n"
                report += f"    Action: {rec['action']}\n"
                report += f"    Impact: {rec['impact']}\n"
        
        return report


# Singleton instance
_monitor_instance: Optional[ARMPerformanceMonitor] = None


def get_arm_performance_monitor() -> ARMPerformanceMonitor:
    """Get singleton ARM performance monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ARMPerformanceMonitor()
    return _monitor_instance


if __name__ == '__main__':
    # Test performance monitoring
    monitor = ARMPerformanceMonitor()
    monitor.start_monitoring(interval=0.5)
    
    # Simulate some inferences
    for i in range(10):
        time.sleep(0.1)
        monitor.record_inference('vision_model', 45.0 + i * 2)
        monitor.record_inference('planning_model', 180.0 + i * 5)
    
    time.sleep(2)
    monitor.stop_monitoring()
    
    print(monitor.get_arm_optimization_report())
