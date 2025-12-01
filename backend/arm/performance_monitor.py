"""
ARM Performance Monitoring Module

Monitors and tracks ARM-specific performance metrics including CPU usage,
memory footprint, inference latency, and power consumption.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    inference_latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    
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
        """Capture current performance metrics"""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
        )
    
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
                    'inference_stats': {},
                }
            
            cpu_values = [m.cpu_percent for m in self.metrics_history]
            memory_values = [m.memory_mb for m in self.metrics_history]
            
            return {
                'cpu_avg': sum(cpu_values) / len(cpu_values),
                'cpu_max': max(cpu_values),
                'memory_avg_mb': sum(memory_values) / len(memory_values),
                'memory_max_mb': max(memory_values),
                'inference_stats': self.get_all_inference_stats(),
                'samples': len(self.metrics_history),
            }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics_history.clear()
            self.inference_times.clear()
        logger.info("Reset ARM performance metrics")
    
    def get_arm_optimization_report(self) -> str:
        """Generate ARM optimization report"""
        summary = self.get_summary()
        
        report = f"""
ARM Performance Report
======================
CPU Usage:
  Average: {summary['cpu_avg']:.1f}%
  Maximum: {summary['cpu_max']:.1f}%

Memory Usage:
  Average: {summary['memory_avg_mb']:.1f} MB
  Maximum: {summary['memory_max_mb']:.1f} MB

Inference Performance:
"""
        
        for model, stats in summary['inference_stats'].items():
            report += f"""
  {model}:
    Inferences: {stats['count']}
    Average: {stats['avg_ms']:.1f} ms
    P50: {stats['p50_ms']:.1f} ms
    P95: {stats['p95_ms']:.1f} ms
    Min: {stats['min_ms']:.1f} ms
    Max: {stats['max_ms']:.1f} ms
"""
        
        report += f"\nTotal Samples: {summary['samples']}\n"
        
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
