"""Application Layer - Use Cases (Business Logic)"""
import time
import asyncio
from typing import Optional
from src.domain.entities import MobileTask, TaskResult, TaskStatus
from src.domain.repositories import DeviceRepository, TaskRepository, ResultRepository
from src.tools.action_execution import ActionExecutor
from src.perception.visual_perception import VisualPerception
from src.llm.llm_planner import LLMPlanner
from src.runtime.recovery import RecoveryManager
from src.runtime.logging import MaskedLogger
from src.production_readiness.logging_utils import log_event, store_secure_log
from src.production_readiness.metrics_server import record_task_metrics
from selenium.common.exceptions import WebDriverException, TimeoutException


class NoDeviceAvailableError(Exception):
    """Raised when no device is available for task execution"""
    pass


class ExecuteTaskUseCase:
    """Use case for executing a mobile automation task"""
    
    def __init__(
        self,
        device_repo: DeviceRepository,
        task_repo: TaskRepository,
        result_repo: ResultRepository
    ):
        self.device_repo = device_repo
        self.task_repo = task_repo
        self.result_repo = result_repo
        self.logger = MaskedLogger(name="ExecuteTaskUseCase")
    
    async def execute(self, task: MobileTask) -> TaskResult:
        """Main execution logic for a task"""
        if not task.validate():
            raise ValueError(f"Invalid task: {task.task_id}")
        
        # Acquire device
        device = await self.device_repo.get_available()
        if not device:
            raise NoDeviceAvailableError(f"No device available for task {task.task_id}")
        
        task.mark_in_progress(device.device_id)
        await self.task_repo.save(task)
        
        self.logger.info(f"Starting task {task.task_id} on device {device.device_id}")
        log_event("TASK_START", f"Task '{task.instruction}' started on {device.device_id}", "INFO")
        
        task_start_time = time.time()
        
        try:
            result = await self._execute_on_device(task, device)
            
            if result.success:
                task.mark_completed()
            else:
                task.mark_failed()
            
            await self.task_repo.save(task)
            await self.result_repo.save_result(result)
            
            return result
            
        finally:
            await self.device_repo.release(device)
            runtime = time.time() - task_start_time
            record_task_metrics(task.status == TaskStatus.COMPLETED, runtime)
    
    async def _execute_on_device(self, task: MobileTask, device) -> TaskResult:
        """Execute task on a specific device"""
        driver = device.session
        executor = ActionExecutor(driver)
        recovery_manager = RecoveryManager(executor)
        visual_perception = VisualPerception()
        llm_planner = LLMPlanner()
        
        task_start_time = time.time()
        success = False
        error_type = None
        error_message = None
        recovery_attempted = False
        recovery_successful = False
        
        max_planning_attempts = 3
        current_instruction = task.instruction
        
        try:
            for attempt in range(max_planning_attempts):
                self.logger.info(f"[Attempt {attempt + 1}] Planning for: {current_instruction}")
                
                ui_state = visual_perception.capture_and_analyze(driver)
                action_plan = llm_planner.generate_action_plan(current_instruction, ui_state)
                
                if not action_plan:
                    self.logger.warning(f"Empty plan generated for '{current_instruction}'")
                    if attempt < max_planning_attempts - 1:
                        current_instruction = f"Re-evaluate and plan for: {task.instruction}"
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise Exception("Failed to generate valid plan after multiple attempts")
                
                try:
                    await self._execute_action_plan(
                        executor, visual_perception, ui_state, action_plan
                    )
                    self.logger.info(f"Successfully completed task {task.task_id}")
                    log_event("TASK_SUCCESS", f"Task '{task.instruction}' completed", "INFO")
                    success = True
                    break
                    
                except (WebDriverException, TimeoutException) as e:
                    error_type = "UI_INTERACTION_ERROR"
                    error_message = str(e)
                    self.logger.error(f"Execution failed: {e}")
                    
                    if attempt < max_planning_attempts - 1:
                        # Attempt LLM reflection
                        corrective_plan = llm_planner.reflect_and_correct(
                            current_instruction, ui_state, action_plan[-1], str(e)
                        )
                        if corrective_plan:
                            self.logger.info("Executing corrective plan...")
                            await self._execute_action_plan(
                                executor, visual_perception, ui_state, corrective_plan
                            )
                            success = True
                            recovery_successful = True
                            break
                    raise
            
        except (WebDriverException, TimeoutException) as e:
            error_type = "UI_INTERACTION_ERROR"
            error_message = str(e)
            recovery_attempted = True
            
            if await recovery_manager.recover(f"Task {task.task_id} execution"):
                recovery_successful = True
                self.logger.info(f"Recovery successful for task {task.task_id}")
            else:
                self.logger.critical(f"Recovery failed for task {task.task_id}")
                
        except Exception as e:
            error_type = "UNEXPECTED_ERROR"
            error_message = str(e)
            self.logger.critical(f"Unexpected error: {e}")
            store_secure_log(
                f"task_failure_{task.task_id}_{int(time.time())}.log",
                f"Task: {task.instruction}\nError: {error_type}\nDetails: {e}"
            )
        
        runtime = time.time() - task_start_time
        
        return TaskResult(
            task_id=task.task_id,
            success=success,
            runtime=runtime,
            error_type=error_type,
            error_message=error_message,
            recovery_attempted=recovery_attempted,
            recovery_successful=recovery_successful
        )
    
    async def _execute_action_plan(
        self,
        executor: ActionExecutor,
        visual_perception: VisualPerception,
        ui_state: dict,
        action_plan: list
    ):
        """Execute a list of actions"""
        for action_step in action_plan:
            action_type = action_step.get("action")
            target_id = action_step.get("target_id")
            value = action_step.get("value")
            
            locator = visual_perception.get_element_locator(ui_state, target_id)
            
            if not locator:
                self.logger.warning(f"Element '{target_id}' not found. Skipping.")
                continue
            
            if action_type == "type_text":
                await executor.type_text(locator[0], locator[1], value)
            elif action_type == "tap":
                await executor.tap(locator[0], locator[1])
            elif action_type == "wait_for_displayed":
                await executor.wait_for_displayed(locator[0], locator[1])
            elif action_type == "sleep":
                await asyncio.sleep(value)
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
