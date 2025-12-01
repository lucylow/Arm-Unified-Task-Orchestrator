"""Infrastructure Layer - In-Memory Result Repository Implementation"""
from typing import List, Dict
from collections import defaultdict
from src.domain.repositories import ResultRepository
from src.domain.entities import TaskResult


class InMemoryResultRepository(ResultRepository):
    """In-memory implementation of ResultRepository for development"""
    
    def __init__(self):
        self._results: Dict[str, List[TaskResult]] = defaultdict(list)
    
    async def save_result(self, result: TaskResult):
        """Save task execution result"""
        self._results[result.task_id].append(result)
    
    async def get_results(self, task_id: str) -> List[TaskResult]:
        """Get all results for a task"""
        return self._results.get(task_id, [])
