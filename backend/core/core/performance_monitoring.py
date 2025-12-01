"""Real-time Performance Monitoring with Prometheus Integration"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Centralized performance metrics collection"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Task metrics
        self.task_counter = Counter(
            'autorl_tasks_total',
            'Total tasks processed',
            ['status', 'agent', 'platform'],
            registry=self.registry
        )
        
        self.task_duration = Histogram(
            'autorl_task_duration_seconds',
            'Task execution time in seconds',
            ['agent', 'platform'],
            registry=self.registry
        )
        
        self.task_errors = Counter(
            'autorl_task_errors_total',
            'Total task errors',
            ['error_type', 'agent', 'severity'],
            registry=self.registry
        )
        
        # Device metrics
        self.active_devices = Gauge(
            'autorl_active_devices',
            'Number of active devices',
            ['platform'],
            registry=self.registry
        )
        
        self.device_utilization = Gauge(
            'autorl_device_utilization_percent',
            'Device utilization percentage',
            registry=self.registry
        )
        
        # Agent metrics
        self.agent_invocations = Counter(
            'autorl_agent_invocations_total',
            'Total agent invocations',
            ['agent_type', 'operation'],
            registry=self.registry
        )
        
        self.agent_latency = Histogram(
            'autorl_agent_latency_seconds',
            'Agent operation latency',
            ['agent_type', 'operation'],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_calls = Counter(
            'autorl_llm_calls_total',
            'Total LLM API calls',
            ['model', 'operation'],
            registry=self.registry
        )
        
        self.llm_tokens = Counter(
            'autorl_llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'token_type'],
            registry=self.registry
        )
        
        self.llm_latency = Histogram(
            'autorl_llm_latency_seconds',
            'LLM API call latency',
            ['model'],
            registry=self.registry
        )
        
        # Recovery metrics
        self.recovery_attempts = Counter(
            'autorl_recovery_attempts_total',
            'Total recovery attempts',
            ['error_type', 'success'],
            registry=self.registry
        )


class PerformanceMonitor:
    """Performance monitoring with automatic metrics collection"""
    
    def __init__(self, metrics_port: int = 9000, enable_server: bool = True):
        self.metrics = PerformanceMetrics()
        self.metrics_port = metrics_port
        self.server_started = False
        
        if enable_server:
            self.start_metrics_server()
    
    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        if not self.server_started:
            try:
                start_http_server(self.metrics_port, registry=self.metrics.registry)
                self.server_started = True
                logger.info(f"Metrics server started on port {self.metrics_port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    @asynccontextmanager
    async def measure_task(
        self, 
        task_id: str, 
        agent_name: str,
        platform: str = 'unknown'
    ):
        """Context manager to measure task execution"""
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
            self.metrics.task_counter.labels(
                status='success',
                agent=agent_name,
                platform=platform
            ).inc()
            
        except Exception as e:
            error_type = type(e).__name__
            self.metrics.task_counter.labels(
                status='error',
                agent=agent_name,
                platform=platform
            ).inc()
            
            self.metrics.task_errors.labels(
                error_type=error_type,
                agent=agent_name,
                severity='high'
            ).inc()
            
            raise
            
        finally:
            duration = time.time() - start_time
            self.metrics.task_duration.labels(
                agent=agent_name,
                platform=platform
            ).observe(duration)
            
            logger.info(
                f"Task {task_id} completed in {duration:.2f}s "
                f"(success={success}, agent={agent_name})"
            )
    
    @asynccontextmanager
    async def measure_agent_operation(
        self,
        agent_type: str,
        operation: str
    ):
        """Measure agent operation performance"""
        start_time = time.time()
        
        self.metrics.agent_invocations.labels(
            agent_type=agent_type,
            operation=operation
        ).inc()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.agent_latency.labels(
                agent_type=agent_type,
                operation=operation
            ).observe(duration)
    
    @asynccontextmanager
    async def measure_llm_call(
        self,
        model: str,
        operation: str = 'generate'
    ):
        """Measure LLM API call performance"""
        start_time = time.time()
        
        self.metrics.llm_calls.labels(
            model=model,
            operation=operation
        ).inc()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.llm_latency.labels(model=model).observe(duration)
    
    def record_llm_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Record LLM token usage"""
        self.metrics.llm_tokens.labels(
            model=model,
            token_type='prompt'
        ).inc(prompt_tokens)
        
        self.metrics.llm_tokens.labels(
            model=model,
            token_type='completion'
        ).inc(completion_tokens)
    
    def update_device_count(self, count: int, platform: str = 'all'):
        """Update active device count"""
        self.metrics.active_devices.labels(platform=platform).set(count)
    
    def update_device_utilization(self, utilization_percent: float):
        """Update device utilization percentage"""
        self.metrics.device_utilization.set(utilization_percent)
    
    def record_recovery_attempt(
        self,
        error_type: str,
        success: bool
    ):
        """Record error recovery attempt"""
        self.metrics.recovery_attempts.labels(
            error_type=error_type,
            success='true' if success else 'false'
        ).inc()


def monitored_async(agent_type: str, operation: str):
    """Decorator to automatically monitor async function performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = getattr(args[0], '_performance_monitor', None)
            
            if monitor:
                async with monitor.measure_agent_operation(agent_type, operation):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor(metrics_port: int = 9000) -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(metrics_port=metrics_port)
    
    return _global_monitor
