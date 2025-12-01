"""Domain Layer - Repository Interfaces (Abstract)"""
from abc import ABC, abstractmethod
from typing import Optional, List
from src.domain.entities import MobileTask, TaskResult
from src.runtime.device_manager import Device


class DeviceRepository(ABC):
    """Abstract interface for device management"""
    
    @abstractmethod
    async def get_available(self) -> Optional[Device]:
        """Get an available device from the pool"""
        pass
    
    @abstractmethod
    async def release(self, device: Device):
        """Release a device back to the pool"""
        pass
    
    @abstractmethod
    def get_by_id(self, device_id: str) -> Optional[Device]:
        """Get a device by its ID"""
        pass
    
    @abstractmethod
    async def shutdown_all(self):
        """Shutdown all devices"""
        pass


class TaskRepository(ABC):
    """Abstract interface for task management"""
    
    @abstractmethod
    async def save(self, task: MobileTask) -> bool:
        """Save or update a task"""
        pass
    
    @abstractmethod
    async def get_by_id(self, task_id: str) -> Optional[MobileTask]:
        """Retrieve a task by ID"""
        pass
    
    @abstractmethod
    async def get_pending(self) -> List[MobileTask]:
        """Get all pending tasks"""
        pass


class ResultRepository(ABC):
    """Abstract interface for task result storage"""
    
    @abstractmethod
    async def save_result(self, result: TaskResult):
        """Save task execution result"""
        pass
    
    @abstractmethod
    async def get_results(self, task_id: str) -> List[TaskResult]:
        """Get all results for a task"""
        pass
