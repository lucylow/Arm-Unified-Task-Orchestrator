"""Infrastructure Layer - In-Memory Task Repository Implementation"""
from typing import Optional, List, Dict
from src.domain.repositories import TaskRepository
from src.domain.entities import MobileTask, TaskStatus


class InMemoryTaskRepository(TaskRepository):
    """In-memory implementation of TaskRepository for development"""
    
    def __init__(self):
        self._tasks: Dict[str, MobileTask] = {}
    
    async def save(self, task: MobileTask) -> bool:
        """Save or update a task"""
        self._tasks[task.task_id] = task
        return True
    
    async def get_by_id(self, task_id: str) -> Optional[MobileTask]:
        """Retrieve a task by ID"""
        return self._tasks.get(task_id)
    
    async def get_pending(self) -> List[MobileTask]:
        """Get all pending tasks"""
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.PENDING
        ]
