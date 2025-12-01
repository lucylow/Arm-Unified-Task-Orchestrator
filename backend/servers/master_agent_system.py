"""
Master AI Agent System
Consolidates all agent implementations into one unified system
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MasterAgentSystem")

# ============================================================================
# AGENT BASE CLASS
# ============================================================================

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    PERCEPTION = "perception"
    PLANNING = "planning"
    EXECUTION = "execution"
    RECOVERY = "recovery"
    REFLECTION = "reflection"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentContext:
    """Context shared between agents"""
    task_id: str
    instruction: str
    device_id: str
    driver: Any = None
    ui_state: Dict[str, Any] = None
    action_plan: List[Dict[str, Any]] = None
    execution_result: Dict[str, Any] = None
    error: Optional[str] = None
    state: AgentState = AgentState.IDLE
    handoff_to: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentContext:
        """Execute agent logic and return updated context"""
        pass
    
    def handoff(self, context: AgentContext, next_agent: str) -> AgentContext:
        """Signal handoff to another agent"""
        context.handoff_to = next_agent
        self.logger.info(f"Handing off to: {next_agent}")
        return context

# ============================================================================
# PERCEPTION AGENT
# ============================================================================

class PerceptionAgent(BaseAgent):
    """Analyzes UI state and extracts visual information"""
    
    def __init__(self, visual_perception=None):
        super().__init__("Perception")
        self.visual_perception = visual_perception
    
    async def execute(self, context: AgentContext) -> AgentContext:
        """Capture and analyze UI state"""
        self.logger.info(f"Analyzing UI for task: {context.instruction}")
        context.state = AgentState.PERCEPTION
        
        try:
            if self.visual_perception and context.driver:
                # Real visual perception
                ui_state = await self._analyze_with_perception(context)
            else:
                # Mock visual perception
                ui_state = self._mock_ui_analysis(context)
            
            context.ui_state = ui_state
            context.metadata['perception_confidence'] = ui_state.get('confidence', 0.95)
            
            # Hand off to planning
            return self.handoff(context, 'planning')
            
        except Exception as e:
            self.logger.error(f"Perception failed: {e}")
            context.error = str(e)
            return self.handoff(context, 'recovery')
    
    async def _analyze_with_perception(self, context: AgentContext) -> Dict[str, Any]:
        """Real UI analysis using visual perception"""
        screenshot_path = self.visual_perception.capture_screenshot(
            context.driver,
            f"perception_{context.task_id}.png"
        )
        ui_state = self.visual_perception.analyze_screenshot(screenshot_path)
        return ui_state
    
    def _mock_ui_analysis(self, context: AgentContext) -> Dict[str, Any]:
        """Mock UI analysis for demo mode"""
        return {
            'elements': [
                {'id': 'button_1', 'type': 'button', 'text': 'Submit', 'bounds': [100, 200, 200, 250]},
                {'id': 'input_1', 'type': 'input', 'hint': 'Enter text', 'bounds': [50, 100, 300, 150]},
            ],
            'screen_size': {'width': 1080, 'height': 1920},
            'confidence': 0.97
        }

# ============================================================================
# PLANNING AGENT
# ============================================================================

class PlanningAgent(BaseAgent):
    """Generates action plans using LLM"""
    
    def __init__(self, llm_planner=None):
        super().__init__("Planning")
        self.llm_planner = llm_planner
    
    async def execute(self, context: AgentContext) -> AgentContext:
        """Generate action plan based on instruction and UI state"""
        self.logger.info(f"Planning actions for: {context.instruction}")
        context.state = AgentState.PLANNING
        
        try:
            if self.llm_planner:
                # Real LLM planning
                action_plan = await self._plan_with_llm(context)
            else:
                # Mock planning
                action_plan = self._mock_planning(context)
            
            context.action_plan = action_plan
            context.metadata['plan_confidence'] = action_plan.get('confidence', 0.94)
            context.metadata['estimated_duration'] = action_plan.get('estimated_duration', 10)
            
            # Hand off to execution
            return self.handoff(context, 'execution')
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            context.error = str(e)
            return self.handoff(context, 'recovery')
    
    async def _plan_with_llm(self, context: AgentContext) -> Dict[str, Any]:
        """Real planning using LLM"""
        plan = self.llm_planner.generate_plan(
            context.instruction,
            context.ui_state
        )
        return plan
    
    def _mock_planning(self, context: AgentContext) -> Dict[str, Any]:
        """Mock action planning"""
        return {
            'actions': [
                {'action': 'tap', 'target_id': 'input_1', 'value': None},
                {'action': 'type_text', 'target_id': 'input_1', 'value': 'test input'},
                {'action': 'tap', 'target_id': 'button_1', 'value': None},
            ],
            'confidence': 0.94,
            'estimated_duration': 8
        }

# ============================================================================
# EXECUTION AGENT
# ============================================================================

class ExecutionAgent(BaseAgent):
    """Executes actions on the device"""
    
    def __init__(self, action_executor=None, visual_perception=None):
        super().__init__("Execution")
        self.action_executor = action_executor
        self.visual_perception = visual_perception
        self.max_retries = 3
    
    async def execute(self, context: AgentContext) -> AgentContext:
        """Execute the action plan"""
        self.logger.info(f"Executing {len(context.action_plan.get('actions', []))} actions")
        context.state = AgentState.EXECUTION
        
        try:
            if self.action_executor:
                # Real execution
                result = await self._execute_with_appium(context)
            else:
                # Mock execution
                result = await self._mock_execution(context)
            
            context.execution_result = result
            context.metadata['execution_time'] = result.get('duration', 0)
            
            if result.get('success', False):
                # Task completed successfully
                context.state = AgentState.COMPLETED
                context.handoff_to = None
            else:
                # Execution failed, try recovery
                context.error = result.get('error', 'Execution failed')
                return self.handoff(context, 'recovery')
            
            return context
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            context.error = str(e)
            return self.handoff(context, 'recovery')
    
    async def _execute_with_appium(self, context: AgentContext) -> Dict[str, Any]:
        """Real action execution"""
        import time
        start_time = time.time()
        
        for action in context.action_plan.get('actions', []):
            action_type = action.get('action')
            target_id = action.get('target_id')
            value = action.get('value')
            
            # Get element locator
            locator = self.visual_perception.get_element_locator(context.ui_state, target_id)
            
            if not locator:
                raise ValueError(f"Element {target_id} not found")
            
            # Execute action with retries
            for attempt in range(self.max_retries):
                try:
                    if action_type == 'tap':
                        await self.action_executor.tap(locator[0], locator[1])
                    elif action_type == 'type_text':
                        await self.action_executor.type_text(locator[0], locator[1], value)
                    elif action_type == 'wait_for_displayed':
                        await self.action_executor.wait_for_displayed(locator[0], locator[1])
                    elif action_type == 'sleep':
                        await asyncio.sleep(value)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(1)
                    else:
                        raise e
        
        duration = time.time() - start_time
        return {'success': True, 'duration': duration}
    
    async def _mock_execution(self, context: AgentContext) -> Dict[str, Any]:
        """Mock action execution with delays"""
        import random
        
        actions = context.action_plan.get('actions', [])
        for i, action in enumerate(actions):
            await asyncio.sleep(0.5)  # Simulate action delay
            self.logger.info(f"Executing action {i+1}/{len(actions)}: {action['action']}")
            
            # Simulate occasional failure
            if random.random() < 0.1:  # 10% failure rate
                return {
                    'success': False,
                    'error': f"Element {action.get('target_id')} not found",
                    'failed_action': action
                }
        
        return {'success': True, 'duration': len(actions) * 0.5}

# ============================================================================
# RECOVERY AGENT
# ============================================================================

class RecoveryAgent(BaseAgent):
    """Handles errors and recovery"""
    
    def __init__(self, recovery_manager=None):
        super().__init__("Recovery")
        self.recovery_manager = recovery_manager
        self.max_recovery_attempts = 2
    
    async def execute(self, context: AgentContext) -> AgentContext:
        """Attempt to recover from error"""
        self.logger.warning(f"Recovering from error: {context.error}")
        context.state = AgentState.RECOVERY
        
        recovery_attempts = context.metadata.get('recovery_attempts', 0)
        
        if recovery_attempts >= self.max_recovery_attempts:
            self.logger.error("Max recovery attempts reached")
            context.state = AgentState.FAILED
            context.handoff_to = None
            return context
        
        context.metadata['recovery_attempts'] = recovery_attempts + 1
        
        try:
            if self.recovery_manager:
                # Real recovery
                await self._recover_with_manager(context)
            else:
                # Mock recovery
                await self._mock_recovery(context)
            
            # Try reflection to generate new plan
            return self.handoff(context, 'reflection')
            
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            context.state = AgentState.FAILED
            context.handoff_to = None
            return context
    
    async def _recover_with_manager(self, context: AgentContext):
        """Real recovery using recovery manager"""
        await self.recovery_manager.recover("Task execution", "home_screen")
    
    async def _mock_recovery(self, context: AgentContext):
        """Mock recovery"""
        await asyncio.sleep(1)
        self.logger.info("Mock recovery: navigating to safe state")

# ============================================================================
# REFLECTION AGENT
# ============================================================================

class ReflectionAgent(BaseAgent):
    """Reflects on failures and generates corrective plans"""
    
    def __init__(self, llm_planner=None):
        super().__init__("Reflection")
        self.llm_planner = llm_planner
    
    async def execute(self, context: AgentContext) -> AgentContext:
        """Reflect and generate corrective plan"""
        self.logger.info("Reflecting on failure and generating corrective plan")
        context.state = AgentState.REFLECTION
        
        try:
            if self.llm_planner:
                # Real reflection with LLM
                corrective_plan = await self._reflect_with_llm(context)
            else:
                # Mock reflection
                corrective_plan = self._mock_reflection(context)
            
            if corrective_plan:
                context.action_plan = corrective_plan
                context.error = None  # Clear error
                # Try execution again with new plan
                return self.handoff(context, 'execution')
            else:
                # Reflection failed
                context.state = AgentState.FAILED
                context.handoff_to = None
                return context
                
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            context.state = AgentState.FAILED
            context.handoff_to = None
            return context
    
    async def _reflect_with_llm(self, context: AgentContext) -> Optional[Dict[str, Any]]:
        """Real reflection using LLM"""
        failed_action = context.execution_result.get('failed_action') if context.execution_result else None
        corrective_plan = self.llm_planner.reflect_and_correct(
            context.instruction,
            context.ui_state,
            failed_action,
            context.error
        )
        return corrective_plan
    
    def _mock_reflection(self, context: AgentContext) -> Optional[Dict[str, Any]]:
        """Mock reflection - simplify the plan"""
        if context.action_plan:
            # Return a simplified version of the plan
            actions = context.action_plan.get('actions', [])
            if len(actions) > 1:
                # Try with fewer actions
                return {
                    'actions': actions[:-1],  # Remove last action
                    'confidence': 0.85,
                    'estimated_duration': 6
                }
        return None

# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================

class MasterAgentOrchestrator:
    """
    Master orchestrator that manages all agents
    Implements handoff pattern for multi-agent workflow
    """
    
    def __init__(
        self,
        visual_perception=None,
        llm_planner=None,
        action_executor=None,
        recovery_manager=None
    ):
        # Initialize all agents
        self.agents = {
            'perception': PerceptionAgent(visual_perception),
            'planning': PlanningAgent(llm_planner),
            'execution': ExecutionAgent(action_executor, visual_perception),
            'recovery': RecoveryAgent(recovery_manager),
            'reflection': ReflectionAgent(llm_planner)
        }
        
        self.logger = logging.getLogger("MasterOrchestrator")
        self.max_handoffs = 15
    
    async def execute_task(
        self,
        task_id: str,
        instruction: str,
        device_id: str,
        driver: Any = None
    ) -> AgentContext:
        """
        Execute a task through the multi-agent workflow
        
        Args:
            task_id: Unique task identifier
            instruction: Natural language task instruction
            device_id: Target device ID
            driver: Appium driver (optional, for production mode)
        
        Returns:
            AgentContext with execution results
        """
        self.logger.info(f"Starting task {task_id}: {instruction}")
        
        # Create context
        context = AgentContext(
            task_id=task_id,
            instruction=instruction,
            device_id=device_id,
            driver=driver
        )
        
        # Start with perception agent
        current_agent_name = 'perception'
        handoff_count = 0
        
        # Execute agent workflow with handoffs
        while current_agent_name and handoff_count < self.max_handoffs:
            agent = self.agents.get(current_agent_name)
            
            if not agent:
                self.logger.error(f"Unknown agent: {current_agent_name}")
                context.state = AgentState.FAILED
                context.error = f"Unknown agent: {current_agent_name}"
                break
            
            self.logger.info(f"Executing agent: {current_agent_name}")
            
            # Execute agent
            context = await agent.execute(context)
            
            # Check for handoff
            next_agent = context.handoff_to
            if next_agent:
                self.logger.info(f"Handoff: {current_agent_name} → {next_agent}")
                current_agent_name = next_agent
                handoff_count += 1
            else:
                # Workflow complete
                self.logger.info(f"Workflow complete. Final state: {context.state.value}")
                break
        
        if handoff_count >= self.max_handoffs:
            self.logger.warning(f"Max handoffs ({self.max_handoffs}) reached")
            context.state = AgentState.FAILED
            context.error = "Max handoffs exceeded"
        
        return context
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'agents': list(self.agents.keys()),
            'total_agents': len(self.agents),
            'max_handoffs': self.max_handoffs
        }

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MasterAgentOrchestrator',
    'AgentContext',
    'AgentState',
    'PerceptionAgent',
    'PlanningAgent',
    'ExecutionAgent',
    'RecoveryAgent',
    'ReflectionAgent'
]

