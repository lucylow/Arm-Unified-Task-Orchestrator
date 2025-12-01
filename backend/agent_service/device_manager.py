import asyncio
import random
import logging
from typing import List, Dict, Optional
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prometheus metrics
TASKS_TOTAL = Counter('autorl_tasks_total', 'Total number of tasks scheduled')
TASKS_SUCCESS = Counter('autorl_tasks_success_total', 'Total successful tasks')
TASKS_FAILURE = Counter('autorl_tasks_failure_total', 'Total failed tasks')
TASK_DURATION = Histogram('autorl_task_duration_seconds', 'Task execution duration seconds')
DEVICES_AVAILABLE = Gauge('autorl_devices_available', 'Number of available devices')


class AsyncDeviceManager:
    """Asynchronous Device Manager for scheduling tasks across devices.

    Features:
    - Register/update devices
    - Schedule tasks to devices with concurrency control
    - Per-action retries with exponential backoff and timeouts
    - Simple round-robin device selection for load distribution
    """

    def __init__(self, devices: Optional[List[Dict]] = None, max_concurrent: int = 4):
        self.devices = devices or []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rr_index = 0

    def register_device(self, device: Dict):
        """Register a new device or update an existing device by id."""
        existing = next((d for d in self.devices if d.get("device_id") == device.get("device_id")), None)
        if existing:
            existing.update(device)
        else:
            self.devices.append(device)

    def update_device_status(self, device_id: str, status: str):
        d = next((d for d in self.devices if d.get("device_id") == device_id), None)
        if d:
            d["status"] = status

    def get_available_devices(self, platform: Optional[str] = None) -> List[Dict]:
        """Return devices considered available for scheduling (status == active)."""
        ds = [d for d in self.devices if d.get("status") in ("active", "idle")]
        if platform:
            ds = [d for d in ds if d.get("platform") == platform]
        return ds

    async def _select_device(self, preferred_id: Optional[str] = None) -> Optional[Dict]:
        async with self._lock:
            if preferred_id:
                d = next((d for d in self.devices if d.get("device_id") == preferred_id), None)
                if d and d.get("status") in ("active", "idle"):
                    return d
                return None
            avail = self.get_available_devices()
            if not avail:
                return None
            # round-robin selection
            self._rr_index = (self._rr_index + 1) % len(avail)
            return avail[self._rr_index]

    async def schedule_task(self, instruction: str, device_id: Optional[str] = None, timeout: int = 30, retries: int = 3) -> Dict:
        """Schedule a single task to run on a selected device.

        Returns a dict with result metadata.
        """
        device = await self._select_device(device_id)
        if not device:
            return {"success": False, "msg": "No available device"}

        # Acquire concurrency slot
        async with self._semaphore:
            TASKS_TOTAL.inc()
            DEVICES_AVAILABLE.set(len(self.get_available_devices()))
            # mark device busy
            prev_status = device.get("status")
            device["status"] = "busy"
            try:
                with TASK_DURATION.time():
                    result = await self._execute_with_retries(device, instruction, timeout, retries)
                if result.get('success'):
                    TASKS_SUCCESS.inc()
                else:
                    TASKS_FAILURE.inc()
                return result
            finally:
                # restore device status (if device still exists)
                device["status"] = prev_status or "active"

    async def _execute_with_retries(self, device: Dict, instruction: str, timeout: int, retries: int) -> Dict:
        attempt = 0
        backoff = 0.5
        last_exc = None
        while attempt <= retries:
            attempt += 1
            try:
                coro = self._simulate_device_execution(device, instruction)
                res = await asyncio.wait_for(coro, timeout=timeout)
                return {"success": True, "attempts": attempt, "result": res}
            except asyncio.TimeoutError as te:
                last_exc = te
                logger.warning("Timeout on device %s attempt %d", device.get("device_id"), attempt)
            except Exception as e:
                last_exc = e
                logger.warning("Execution error on device %s attempt %d: %s", device.get("device_id"), attempt, e)

            # retry backoff
            if attempt <= retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5)

        return {"success": False, "attempts": attempt - 1, "error": str(last_exc)}

    async def _simulate_device_execution(self, device: Dict, instruction: str) -> Dict:
        """Simulate running an instruction on a device. Replace with real executor in production."""
        # simulate variable execution time
        duration = random.uniform(0.5, 2.5)
        await asyncio.sleep(duration)
        # simulate flaky behavior based on battery and status
        battery = device.get("battery", 100)
        base_success = 0.95 if device.get("status") in ("active", "idle") else 0.5
        battery_penalty = 0.0
        if battery < 20:
            battery_penalty = 0.25

        success_prob = max(0.1, base_success - battery_penalty - random.random() * 0.1)
        success = random.random() < success_prob
        return {
            "device_id": device.get("device_id"),
            "instruction": instruction,
            "duration": duration,
            "success": success,
            "battery": battery
        }
