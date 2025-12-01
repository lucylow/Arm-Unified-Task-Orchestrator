"""Infrastructure Layer - Concrete Device Repository Implementation"""
from typing import Optional
from src.domain.repositories import DeviceRepository
from src.runtime.device_manager import Device, DeviceManager


class DeviceRepositoryImpl(DeviceRepository):
    """Concrete implementation of DeviceRepository using DeviceManager"""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
    
    async def get_available(self) -> Optional[Device]:
        """Get an available device from the pool"""
        try:
            return await self.device_manager.acquire_device()
        except Exception as e:
            print(f"Error acquiring device: {e}")
            return None
    
    async def release(self, device: Device):
        """Release a device back to the pool"""
        await self.device_manager.release_device(device)
    
    def get_by_id(self, device_id: str) -> Optional[Device]:
        """Get a device by its ID"""
        return self.device_manager.get_device_by_id(device_id)
    
    async def shutdown_all(self):
        """Shutdown all devices"""
        await self.device_manager.shutdown()
