"""Multi-Agent Orchestration with Handoff Pattern"""
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import asyncio
from src.runtime.logging import MaskedLogger


class Agent(ABC):
    """Base class for specialized agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = MaskedLogger(name=f"Agent_{name}")
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic and return updated context with optional handoff"""
        pass
    
    async def handoff_to(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Signal handoff to another agent"""
        context['handoff_to'] = agent_name
        return context


class PerceptionAgent(Agent):
    """Agent responsible for UI perception and analysis"""
    
    def __init__(self, visual_perception):
        super().__init__("Perception")
        self.visual_perception = visual_perception
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Capture and analyze UI state"""
        self.logger.info("Analyzing UI state...")
        
        driver = context.get('driver')
        if not driver:
            raise ValueError("No driver in context")
        
        ui_state = self.visual_perception.capture_and_analyze(driver)
        context['ui_state'] = ui_state
        
        # Always handoff to planning after perception
        return await self.handoff_to('planning', context)


class PlanningAgent(Agent):
    """Agent responsible for generating action plans"""
    
    def __init__(self, llm_planner):
        super().__init__("Planning")
        self.llm_planner = llm_planner
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action plan based on instruction and UI state"""
        self.logger.info("Generating action plan...")
        
        instruction = context.get('instruction')
        ui_state = context.get('ui_state')
        
        if not instruction or not ui_state:
            raise ValueError("Missing instruction or ui_state in context")
        
        action_plan = self.llm_planner.generate_action_plan(instruction, ui_state)
        context['action_plan'] = action_plan
        
        if not action_plan:
            # Handoff to reflection if plan is empty
            return await self.handoff_to('reflection', context)
        
        # Handoff to execution
        return await self.handoff_to('execution', context)


class ExecutionAgent(Agent):
    """Agent responsible for executing action plans"""
    
    def __init__(self, executor, visual_perception):
        super().__init__("Execution")
        self.executor = executor
        self.visual_perception = visual_perception
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action plan"""
        self.logger.info("Executing action plan...")
        
        action_plan = context.get('action_plan')
        ui_state = context.get('ui_state')
        
        if not action_plan:
            raise ValueError("No action_plan in context")
        
        try:
            await self._execute_actions(action_plan, ui_state)
            context['execution_success'] = True
            context['handoff_to'] = None  # End workflow
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            context['execution_success'] = False
            context['execution_error'] = str(e)
            context['failed_action'] = action_plan[-1] if action_plan else None
            # Handoff to reflection for recovery
            return await self.handoff_to('reflection', context)
        
        return context
    
    async def _execute_actions(self, action_plan: List[Dict], ui_state: Dict):
        """Execute individual actions"""
        for action_step in action_plan:
            action_type = action_step.get("action")
            target_id = action_step.get("target_id")
            value = action_step.get("value")
            
            locator = self.visual_perception.get_element_locator(ui_state, target_id)
            
            if not locator:
                self.logger.warning(f"Element '{target_id}' not found. Skipping.")
                continue
            
            if action_type == "type_text":
                await self.executor.type_text(locator[0], locator[1], value)
            elif action_type == "tap":
                await self.executor.tap(locator[0], locator[1])
            elif action_type == "wait_for_displayed":
                await self.executor.wait_for_displayed(locator[0], locator[1])
            elif action_type == "sleep":
                await asyncio.sleep(value)


class ReflectionAgent(Agent):
    """Agent responsible for reflecting on failures and generating corrective plans"""
    
    def __init__(self, llm_planner):
        super().__init__("Reflection")
        self.llm_planner = llm_planner
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on failure and generate corrective plan"""
        self.logger.info("Reflecting on failure...")
        
        instruction = context.get('instruction')
        ui_state = context.get('ui_state')
        failed_action = context.get('failed_action')
        error_message = context.get('execution_error', '')
        
        if not failed_action:
            # No specific failure, might be empty plan
            self.logger.warning("No failed action to reflect on")
            context['handoff_to'] = None
            return context
        
        corrective_plan = self.llm_planner.reflect_and_correct(
            instruction, ui_state, failed_action, error_message
        )
        
        if corrective_plan:
            context['action_plan'] = corrective_plan
            # Handoff back to execution with corrective plan
            return await self.handoff_to('execution', context)
        else:
            # Reflection failed, end workflow
            self.logger.warning("Reflection did not yield corrective plan")
            context['handoff_to'] = None
            return context


class AgentOrchestrator:
    """Orchestrates multiple specialized agents using handoff pattern"""
    
    def __init__(self, visual_perception, llm_planner, executor):
        self.agents: Dict[str, Agent] = {
            'perception': PerceptionAgent(visual_perception),
            'planning': PlanningAgent(llm_planner),
            'execution': ExecutionAgent(executor, visual_perception),
            'reflection': ReflectionAgent(llm_planner)
        }
        self.logger = MaskedLogger(name="AgentOrchestrator")
    
    async def execute_workflow(
        self,
        driver,
        instruction: str,
        max_handoffs: int = 10
    ) -> Dict[str, Any]:
        """Execute multi-agent workflow with handoff pattern"""
        context = {
            'instruction': instruction,
            'driver': driver,
            'state': 'init'
        }
        
        current_agent_name = 'perception'
        handoff_count = 0
        
        while current_agent_name and handoff_count < max_handoffs:
            agent = self.agents.get(current_agent_name)
            if not agent:
                self.logger.error(f"Unknown agent: {current_agent_name}")
                break
            
            self.logger.info(f"Executing agent: {current_agent_name}")
            context = await agent.execute(context)
            
            next_agent = context.get('handoff_to')
            if next_agent:
                self.logger.info(f"Handoff from {current_agent_name} to {next_agent}")
                current_agent_name = next_agent
                handoff_count += 1
            else:
                # Workflow complete
                self.logger.info("Workflow complete, no more handoffs")
                break
        
        if handoff_count >= max_handoffs:
            self.logger.warning(f"Max handoffs ({max_handoffs}) reached")
        
        return context
