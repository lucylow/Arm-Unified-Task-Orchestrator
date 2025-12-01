"""Domain Layer - Core Business Entities"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass
class MobileTask:
    """Core business entity representing a mobile automation task"""
    task_id: str
    instruction: str
    target_app: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    device_id: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def validate(self) -> bool:
        """Core business validation logic"""
        return bool(self.instruction and self.target_app)
    
    def mark_in_progress(self, device_id: str):
        """Transition to in-progress state"""
        self.status = TaskStatus.IN_PROGRESS
        self.device_id = device_id
    
    def mark_completed(self):
        """Transition to completed state"""
        self.status = TaskStatus.COMPLETED
    
    def mark_failed(self):
        """Transition to failed state"""
        self.status = TaskStatus.FAILED
    
    def mark_recovered(self):
        """Transition to recovered state after failure"""
        self.status = TaskStatus.RECOVERED


@dataclass
class UIElement:
    """Core entity representing a UI element"""
    element_id: str
    locator_type: str
    locator_value: str
    is_displayed: bool = False
    is_enabled: bool = False
    text_content: Optional[str] = None


@dataclass
class TaskResult:
    """Entity representing the result of a task execution"""
    task_id: str
    success: bool
    runtime: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
