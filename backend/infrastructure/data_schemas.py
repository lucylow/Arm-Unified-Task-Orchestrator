"""
Data Schemas for AI Agent ETL Pipeline
========================================
Defines standard data schemas for agent data structures
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ElementType(Enum):
    """UI element types"""
    BUTTON = "button"
    INPUT = "input"
    TEXT = "text"
    IMAGE = "image"
    LINK = "link"
    LABEL = "label"
    CONTAINER = "container"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Action types"""
    TAP = "tap"
    TYPE_TEXT = "type_text"
    SWIPE = "swipe"
    SCROLL = "scroll"
    WAIT = "wait"
    VERIFY = "verify"
    NAVIGATE = "navigate"
    UNKNOWN = "unknown"


@dataclass
class UIElementSchema:
    """Schema for UI element"""
    id: str
    type: ElementType
    text: str = ""
    bbox: List[float] = field(default_factory=list)
    confidence: float = 1.0
    is_interactable: bool = False
    locator_type: Optional[str] = None
    locator_value: Optional[str] = None
    normalized_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value if isinstance(self.type, ElementType) else self.type,
            'text': self.text,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'is_interactable': self.is_interactable,
            'locator_type': self.locator_type,
            'locator_value': self.locator_value,
            'normalized_at': self.normalized_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIElementSchema':
        """Create from dictionary"""
        return cls(
            id=data.get('id', ''),
            type=ElementType(data.get('type', 'unknown')),
            text=data.get('text', ''),
            bbox=data.get('bbox', []),
            confidence=float(data.get('confidence', 1.0)),
            is_interactable=bool(data.get('is_interactable', False)),
            locator_type=data.get('locator_type'),
            locator_value=data.get('locator_value'),
            normalized_at=data.get('normalized_at')
        )


@dataclass
class UIStateSchema:
    """Schema for UI state"""
    elements: List[UIElementSchema] = field(default_factory=list)
    text_content: str = ""
    screenshot_path: str = ""
    timestamp: str = ""
    element_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'elements': [elem.to_dict() for elem in self.elements],
            'text_content': self.text_content,
            'screenshot_path': self.screenshot_path,
            'timestamp': self.timestamp,
            'element_count': len(self.elements),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UIStateSchema':
        """Create from dictionary"""
        elements = [UIElementSchema.from_dict(elem) if isinstance(elem, dict) else elem 
                   for elem in data.get('elements', [])]
        return cls(
            elements=elements,
            text_content=data.get('text_content', ''),
            screenshot_path=data.get('screenshot_path', ''),
            timestamp=data.get('timestamp', ''),
            element_count=data.get('element_count', len(elements)),
            metadata=data.get('metadata', {})
        )


@dataclass
class ActionStepSchema:
    """Schema for action step"""
    step_id: str
    step_number: int
    action: ActionType
    target_id: str = ""
    value: str = ""
    description: str = ""
    confidence: float = 0.8
    estimated_duration_ms: int = 1000
    dependencies: List[str] = field(default_factory=list)
    normalized_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'step_id': self.step_id,
            'step_number': self.step_number,
            'action': self.action.value if isinstance(self.action, ActionType) else self.action,
            'target_id': self.target_id,
            'value': self.value,
            'description': self.description,
            'confidence': self.confidence,
            'estimated_duration_ms': self.estimated_duration_ms,
            'dependencies': self.dependencies,
            'normalized_at': self.normalized_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionStepSchema':
        """Create from dictionary"""
        action = data.get('action', 'unknown')
        if isinstance(action, str):
            try:
                action = ActionType(action)
            except ValueError:
                action = ActionType.UNKNOWN
        
        return cls(
            step_id=data.get('step_id', ''),
            step_number=int(data.get('step_number', 0)),
            action=action,
            target_id=data.get('target_id', ''),
            value=data.get('value', ''),
            description=data.get('description', ''),
            confidence=float(data.get('confidence', 0.8)),
            estimated_duration_ms=int(data.get('estimated_duration_ms', 1000)),
            dependencies=data.get('dependencies', []),
            normalized_at=data.get('normalized_at')
        )


@dataclass
class ActionPlanSchema:
    """Schema for action plan"""
    plan_id: str
    instruction: str
    steps: List[ActionStepSchema] = field(default_factory=list)
    total_steps: int = 0
    estimated_duration_seconds: float = 0.0
    overall_confidence: float = 0.8
    reasoning: str = ""
    normalized_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'plan_id': self.plan_id,
            'instruction': self.instruction,
            'steps': [step.to_dict() for step in self.steps],
            'total_steps': len(self.steps),
            'estimated_duration_seconds': self.estimated_duration_seconds,
            'overall_confidence': self.overall_confidence,
            'reasoning': self.reasoning,
            'normalized_at': self.normalized_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionPlanSchema':
        """Create from dictionary"""
        steps = [ActionStepSchema.from_dict(step) if isinstance(step, dict) else step 
                for step in data.get('steps', [])]
        return cls(
            plan_id=data.get('plan_id', ''),
            instruction=data.get('instruction', ''),
            steps=steps,
            total_steps=data.get('total_steps', len(steps)),
            estimated_duration_seconds=float(data.get('estimated_duration_seconds', 0.0)),
            overall_confidence=float(data.get('overall_confidence', 0.8)),
            reasoning=data.get('reasoning', ''),
            normalized_at=data.get('normalized_at')
        )


@dataclass
class TaskSchema:
    """Schema for task"""
    task_id: str
    instruction: str
    device_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"
    created_at: str = ""
    instruction_normalized: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalized_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'instruction': self.instruction,
            'device_id': self.device_id,
            'parameters': self.parameters,
            'priority': self.priority,
            'created_at': self.created_at,
            'instruction_normalized': self.instruction_normalized,
            'metadata': self.metadata,
            'normalized_at': self.normalized_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSchema':
        """Create from dictionary"""
        return cls(
            task_id=data.get('task_id', ''),
            instruction=data.get('instruction', ''),
            device_id=data.get('device_id', ''),
            parameters=data.get('parameters', {}),
            priority=data.get('priority', 'medium'),
            created_at=data.get('created_at', ''),
            instruction_normalized=data.get('instruction_normalized', ''),
            metadata=data.get('metadata', {}),
            normalized_at=data.get('normalized_at')
        )


@dataclass
class AgentDataSchema:
    """Complete schema for agent data"""
    task: Optional[TaskSchema] = None
    ui_state: Optional[UIStateSchema] = None
    action_plan: Optional[ActionPlanSchema] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    etl_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'metadata': self.metadata,
            'etl_metadata': self.etl_metadata
        }
        if self.task:
            result['task'] = self.task.to_dict()
        if self.ui_state:
            result['ui_state'] = self.ui_state.to_dict()
        if self.action_plan:
            result['action_plan'] = self.action_plan.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentDataSchema':
        """Create from dictionary"""
        task = None
        ui_state = None
        action_plan = None
        
        if 'normalized_task' in data or 'task' in data:
            task_data = data.get('normalized_task') or data.get('task', {})
            if task_data:
                task = TaskSchema.from_dict(task_data)
        
        if 'normalized_ui_state' in data or 'ui_state' in data:
            ui_data = data.get('normalized_ui_state') or data.get('ui_state', {})
            if ui_data:
                ui_state = UIStateSchema.from_dict(ui_data)
        
        if 'normalized_action_plan' in data or 'action_plan' in data:
            plan_data = data.get('normalized_action_plan') or data.get('action_plan', {})
            if plan_data:
                action_plan = ActionPlanSchema.from_dict(plan_data)
        
        return cls(
            task=task,
            ui_state=ui_state,
            action_plan=action_plan,
            metadata=data.get('_metadata', {}),
            etl_metadata=data.get('etl_metadata', {})
        )

