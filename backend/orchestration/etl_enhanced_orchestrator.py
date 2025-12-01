"""
ETL-Enhanced Agent Orchestrator
================================
Integrates data ETL pipeline with agent orchestration for improved data quality
"""

from typing import Dict, Any, Optional
import logging
from .agent_orchestrator import AgentOrchestrator, Agent
from ..infrastructure.data_etl_pipeline import DataETLPipeline, ETLResult
from ..infrastructure.data_schemas import AgentDataSchema

logger = logging.getLogger(__name__)


class ETLEnhancedAgentOrchestrator(AgentOrchestrator):
    """Agent orchestrator with integrated ETL pipeline"""
    
    def __init__(self, visual_perception, llm_planner, executor, etl_config: Dict[str, Any] = None):
        super().__init__(visual_perception, llm_planner, executor)
        self.etl_pipeline = DataETLPipeline(etl_config or {})
        self.logger = logging.getLogger("ETLEnhancedOrchestrator")
        self.etl_enabled = etl_config.get('enabled', True) if etl_config else True
    
    async def execute_workflow(
        self,
        driver,
        instruction: str,
        max_handoffs: int = 10,
        use_etl: bool = True
    ) -> Dict[str, Any]:
        """Execute workflow with ETL-enhanced data processing"""
        
        # Initial context
        context = {
            'instruction': instruction,
            'driver': driver,
            'state': 'init',
            'etl_results': {}
        }
        
        # Process initial task data through ETL if enabled
        if use_etl and self.etl_enabled:
            try:
                self.logger.info("Processing initial task data through ETL pipeline")
                etl_result = self.etl_pipeline.process(
                    source_type='task_instruction',
                    source_data=instruction,
                    load_targets=['memory']
                )
                
                if etl_result.success:
                    context['normalized_task'] = etl_result.data.get('normalized_task', {})
                    context['etl_results']['initial'] = {
                        'success': True,
                        'quality': etl_result.metadata.quality.value,
                        'warnings': etl_result.warnings
                    }
                else:
                    self.logger.warning(f"Initial ETL processing had errors: {etl_result.errors}")
                    context['etl_results']['initial'] = {
                        'success': False,
                        'errors': etl_result.errors
                    }
            except Exception as e:
                self.logger.error(f"ETL processing failed: {e}")
                # Continue without ETL if it fails
        
        # Execute standard workflow
        context = await super().execute_workflow(driver, instruction, max_handoffs)
        
        # Post-process results through ETL
        if use_etl and self.etl_enabled:
            try:
                # Process UI state if available
                if 'ui_state' in context:
                    self.logger.info("Processing UI state through ETL pipeline")
                    etl_result = self.etl_pipeline.process(
                        source_type='ui_state',
                        source_data=context['ui_state'],
                        load_targets=['memory', 'cache']
                    )
                    
                    if etl_result.success:
                        context['normalized_ui_state'] = etl_result.data.get('normalized_ui_state', {})
                        context['etl_results']['ui_state'] = {
                            'success': True,
                            'quality': etl_result.metadata.quality.value,
                            'cache_hit': etl_result.cache_hit
                        }
                
                # Process action plan if available
                if 'action_plan' in context:
                    self.logger.info("Processing action plan through ETL pipeline")
                    etl_result = self.etl_pipeline.process(
                        source_type='action_plan',
                        source_data={'action_plan': context['action_plan']},
                        load_targets=['memory']
                    )
                    
                    if etl_result.success:
                        context['normalized_action_plan'] = etl_result.data.get('normalized_action_plan', {})
                        context['etl_results']['action_plan'] = {
                            'success': True,
                            'quality': etl_result.metadata.quality.value
                        }
            except Exception as e:
                self.logger.error(f"Post-processing ETL failed: {e}")
        
        return context


class ETLEnhancedPerceptionAgent(Agent):
    """Perception agent with ETL integration"""
    
    def __init__(self, visual_perception, etl_pipeline: DataETLPipeline):
        super().__init__("ETLPerception")
        self.visual_perception = visual_perception
        self.etl_pipeline = etl_pipeline
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Capture and analyze UI state with ETL processing"""
        self.logger.info("Analyzing UI state with ETL...")
        
        driver = context.get('driver')
        if not driver:
            raise ValueError("No driver in context")
        
        # Capture UI state
        ui_state = self.visual_perception.capture_and_analyze(driver)
        
        # Process through ETL pipeline
        try:
            etl_result = self.etl_pipeline.process(
                source_type='ui_state',
                source_data=ui_state,
                load_targets=['memory', 'cache']
            )
            
            if etl_result.success:
                # Use normalized data
                context['ui_state'] = ui_state  # Keep original for compatibility
                context['normalized_ui_state'] = etl_result.data.get('normalized_ui_state', {})
                context['etl_metadata'] = {
                    'quality': etl_result.metadata.quality.value,
                    'validation_passed': etl_result.metadata.validation_passed,
                    'cache_hit': etl_result.cache_hit
                }
                
                if etl_result.warnings:
                    self.logger.warning(f"ETL warnings: {etl_result.warnings}")
            else:
                self.logger.error(f"ETL processing failed: {etl_result.errors}")
                context['ui_state'] = ui_state  # Fallback to original
        except Exception as e:
            self.logger.error(f"ETL processing exception: {e}")
            context['ui_state'] = ui_state  # Fallback to original
        
        # Handoff to planning
        return await self.handoff_to('planning', context)


class ETLEnhancedPlanningAgent(Agent):
    """Planning agent with ETL integration"""
    
    def __init__(self, llm_planner, etl_pipeline: DataETLPipeline):
        super().__init__("ETLPlanning")
        self.llm_planner = llm_planner
        self.etl_pipeline = etl_pipeline
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action plan with ETL processing"""
        self.logger.info("Generating action plan with ETL...")
        
        instruction = context.get('instruction')
        ui_state = context.get('ui_state') or context.get('normalized_ui_state', {})
        
        if not instruction or not ui_state:
            raise ValueError("Missing instruction or ui_state in context")
        
        # Generate action plan
        action_plan = self.llm_planner.generate_action_plan(instruction, ui_state)
        
        # Process through ETL pipeline
        if action_plan:
            try:
                etl_result = self.etl_pipeline.process(
                    source_type='action_plan',
                    source_data={'action_plan': action_plan, 'instruction': instruction},
                    load_targets=['memory']
                )
                
                if etl_result.success:
                    context['action_plan'] = action_plan  # Keep original for compatibility
                    context['normalized_action_plan'] = etl_result.data.get('normalized_action_plan', {})
                    
                    # Use normalized plan if validation passed
                    if etl_result.metadata.validation_passed:
                        normalized_plan = context['normalized_action_plan']
                        if normalized_plan and 'steps' in normalized_plan:
                            # Convert normalized steps back to original format for execution
                            context['action_plan'] = [
                                {
                                    'action': step.get('action', ''),
                                    'target_id': step.get('target_id', ''),
                                    'value': step.get('value', ''),
                                    'description': step.get('description', '')
                                }
                                for step in normalized_plan.get('steps', [])
                            ]
                else:
                    self.logger.warning(f"Action plan ETL validation failed: {etl_result.errors}")
            except Exception as e:
                self.logger.error(f"Action plan ETL processing exception: {e}")
        
        context['action_plan'] = action_plan
        
        if not action_plan:
            return await self.handoff_to('reflection', context)
        
        return await self.handoff_to('execution', context)

