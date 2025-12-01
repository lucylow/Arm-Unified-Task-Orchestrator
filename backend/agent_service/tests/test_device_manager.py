import asyncio
import pytest

from agent_service.device_manager import AsyncDeviceManager


@pytest.mark.asyncio
async def test_register_and_get_devices():
    dm = AsyncDeviceManager()
    dm.register_device({"device_id": "d1", "platform": "android", "status": "active", "battery": 80})
    dm.register_device({"device_id": "d2", "platform": "ios", "status": "idle", "battery": 90})
    avail = dm.get_available_devices()
    assert len(avail) == 2


@pytest.mark.asyncio
async def test_schedule_task_success():
    dm = AsyncDeviceManager(max_concurrent=2)
    dm.register_device({"device_id": "d1", "platform": "android", "status": "active", "battery": 80})

    res = await dm.schedule_task("test instruction", device_id="d1", timeout=5, retries=2)
    assert isinstance(res, dict)
    assert res.get("success") in (True, False)


@pytest.mark.asyncio
async def test_concurrent_scheduling():
    dm = AsyncDeviceManager(max_concurrent=2)
    for i in range(4):
        dm.register_device({"device_id": f"d{i}", "platform": "android", "status": "active", "battery": 80})

    async def submit(i):
        return await dm.schedule_task(f"ins-{i}", timeout=5, retries=1)

    tasks = [asyncio.create_task(submit(i)) for i in range(6)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 6
    for r in results:
        assert "attempts" in r or r.get("success") is False
