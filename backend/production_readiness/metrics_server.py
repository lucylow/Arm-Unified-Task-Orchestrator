import time
import random
import logging
from prometheus_client import start_http_server, Gauge, Counter

logger = logging.getLogger("MetricsServer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

task_success_counter = Counter("autorl_success_total", "Successful tasks")
task_failure_counter = Counter("autorl_failure_total", "Failed tasks")
current_active_tasks_gauge = Gauge("autorl_active_tasks", "Current number of active tasks")
avg_runtime_gauge = Gauge("autorl_avg_runtime_sec", "Average runtime per task")

def start_metrics_server(port: int = 9000):
    """Starts the Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus metrics server: {e}")

def record_task_outcome(success: bool, duration: float = 0.0):
    """Records a task outcome and updates metrics."""
    if success:
        task_success_counter.inc()
        logger.info(f"Task completed successfully. Duration: {duration:.2f}s")
    else:
        task_failure_counter.inc()
        logger.warning(f"Task failed. Duration: {duration:.2f}s")
    
    # For average runtime, in a real system, you'd calculate a moving average
    # For simplicity here, we'll just set it (not truly an average across all tasks)
    if duration > 0:
        avg_runtime_gauge.set(duration)

def increment_active_tasks():
    current_active_tasks_gauge.inc()
    logger.debug("Active tasks incremented.")

def decrement_active_tasks():
    current_active_tasks_gauge.dec()
    logger.debug("Active tasks decremented.")


