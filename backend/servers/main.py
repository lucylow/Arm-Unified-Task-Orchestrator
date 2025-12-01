
import asyncio
import os
from appium.webdriver import Remote
from appium.options.common import AppiumOptions
from selenium.common.exceptions import TimeoutException
from typing import List, Dict, Any

from runtime.device_manager import Device, DeviceManager
from tools.action_execution import ActionExecutor
from error_handling.recovery import RecoveryManager
# PolicyManager and shadow_run not available - using mock
try:
    from rl.policy_manager import PolicyManager, shadow_run
except ImportError:
    # Mock PolicyManager for now
    class PolicyManager:
        def __init__(self):
            self.policies = {}
        def register_policy(self, name, policy):
            self.policies[name] = policy
    
    async def shadow_run(policy, func, *args, **kwargs):
        return await func(*args, **kwargs)

# New imports for enhanced functionality
from perception.visual_perception import VisualPerception
from llm.llm_planner import LLMPlanner
from production_readiness.logging_utils import log_event, mask_sensitive, store_secure_log, decrypt_secure_log
from production_readiness.metrics_server import task_success_counter, task_failure_counter, current_active_tasks_gauge, avg_runtime_gauge

# Configuration
APPIUM_SERVER_URL = os.getenv("APPIUM_SERVER_URL", "http://localhost:4723/wd/hub")
SCREENSHOT_DIR = "./screenshots"
LOG_DIR = "./logs"

# Ensure directories exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize components
device_manager = DeviceManager()
policy_manager = PolicyManager()
visual_perception = VisualPerception(screenshot_dir=SCREENSHOT_DIR)
llm_planner = LLMPlanner()

async def initialize_appium_driver(device: Device):
    options = AppiumOptions()
    options.set_capability("platformName", device.platform)
    options.set_capability("appium:deviceName", device.device_id)
    options.set_capability("appium:automationName", "UiAutomator2" if device.platform == "android" else "XCUITest")
    options.set_capability("appium:newCommandTimeout", 60)
    # For a real app, you'd add 'appium:app' capability with the path to your .apk or .ipa
    # options.set_capability("appium:app", "/path/to/your/app.apk")

    log_event("AppiumInit", f"Initializing Appium driver for {device.device_id} on {APPIUM_SERVER_URL}")
    driver = Remote(APPIUM_SERVER_URL, options=options)
    device.session = driver
    return driver

async def execute_action_plan(executor: ActionExecutor, plan: List[Dict[str, Any]], ui_state: Dict[str, Any]):
    for action_item in plan:
        action_type = action_item["action"]
        target_id = action_item.get("target_id")
        value = action_item.get("value")

        # Get actual locator from UI state if available
        locator_tuple = visual_perception.get_element_locator(ui_state, target_id)
        if not locator_tuple and target_id:
            log_event("ActionError", f"Locator not found for target_id: {target_id}", level="ERROR")
            raise ValueError(f"Locator not found for target_id: {target_id}")

        locator_type, locator_value = locator_tuple if locator_tuple else (None, None)

        try:
            if action_type == "tap":
                log_event("ActionExecute", f"Tapping element: {target_id}")
                await executor.tap(locator_type, locator_value)
            elif action_type == "type_text":
                log_event("ActionExecute", f"Typing text into {target_id}")
                await executor.type_text(locator_type, locator_value, value)
            elif action_type == "wait_for_displayed":
                log_event("ActionExecute", f"Waiting for element: {target_id}")
                await executor.wait_for_displayed(locator_type, locator_value)
            elif action_type == "sleep":
                log_event("ActionExecute", f"Sleeping for {value} seconds")
                await asyncio.sleep(value)
            else:
                log_event("ActionError", f"Unknown action type: {action_type}", level="ERROR")
                raise ValueError(f"Unknown action type: {action_type}")
        except TimeoutException as e:
            log_event("ActionFailure", f"Action {action_type} failed for {target_id}: {e}", level="ERROR")
            raise e
        except Exception as e:
            log_event("ActionFailure", f"Action {action_type} failed for {target_id}: {e}", level="ERROR")
            raise e

async def run_agent_task(device: Device, task_instruction: str):
    start_time = asyncio.get_event_loop().time()
    current_active_tasks_gauge.inc() # Increment active tasks metric
    log_event("TaskStart", f"Starting task '{task_instruction}' on device {device.device_id}")
    driver = None
    task_success = False
    try:
        driver = await initialize_appium_driver(device)
        executor = ActionExecutor(driver)
        recovery_manager = RecoveryManager(executor)

        # 1. Perception: Capture and analyze UI state
        screenshot_path = visual_perception.capture_screenshot(driver, f"initial_state_{device.device_id}.png")
        ui_state = visual_perception.analyze_screenshot(screenshot_path)
        log_event("Perception", f"UI state captured for {device.device_id}")

        # 2. LLM Planning: Generate action plan
        action_plan = llm_planner.generate_plan(task_instruction, ui_state)
        if not action_plan:
            raise ValueError("LLM Planner failed to generate an action plan.")

        # 3. Execute Action Plan with Error Recovery
        try:
            await execute_action_plan(executor, action_plan, ui_state)
            task_success = True
        except Exception as e:
            log_event("TaskExecutionFailure", f"Initial plan execution failed: {e}", level="WARNING")
            # Attempt reflection and correction
            corrected_plan = llm_planner.reflect_and_correct(task_instruction, ui_state, action_plan[0], str(e))
            if corrected_plan:
                log_event("LLMReflection", "Executing corrected plan.")
                await execute_action_plan(executor, corrected_plan, ui_state)
                task_success = True
            else:
                log_event("LLMReflection", "No corrective plan generated. Attempting generic recovery.", level="WARNING")
                await recovery_manager.recover("Task execution", "home_screen")
                task_success = False # Generic recovery doesn't guarantee task success

        log_event("TaskEnd", f"Task '{task_instruction}' completed. Success: {task_success}")
        return task_success

    except Exception as e:
        log_event("TaskFailure", f"Task '{task_instruction}' failed on device {device.device_id}: {e}", level="ERROR")
        return False

    finally:
        if driver:
            log_event("AppiumQuit", f"Quitting Appium driver for {device.device_id}")
            driver.quit()
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        if task_success:
            task_success_counter.inc()
        else:
            task_failure_counter.inc()
        avg_runtime_gauge.set(duration) # Set average runtime (consider using a summary/histogram for real avg)
        current_active_tasks_gauge.dec() # Decrement active tasks metric

async def main():
    log_event("OrchestratorStart", "Starting AutoRL Agent Orchestrator")

    # Add some dummy devices
    device_manager.add_device(Device("emulator-5554", "android", False))
    device_manager.add_device(Device("iPhone 15", "ios", False))
    device_manager.add_device(Device("emulator-5556", "android", False))

    log_event("DeviceManagement", f"Registered {len(device_manager.devices)} devices.")

    # Register a dummy policy
    dummy_policy = {"strategy": "explore", "params": {"epsilon": 0.1}}
    policy_manager.register_policy("initial_policy", dummy_policy)
    log_event("PolicyManagement", f"Registered initial_policy: {policy_manager.policies['initial_policy']}")

    # Simulate running multiple tasks
    tasks_to_run = [
        "Login to the app",
        "Navigate to profile settings",
        "Login to the app", # This one might trigger recovery if the first one fails
        "Perform a search"
    ]

    # For demonstration, we'll run them sequentially to simplify the mock Appium interaction
    # In a real scenario, you'd use parallel_runner.py's run_all_tasks
    for task_instruction in tasks_to_run:
        device = await device_manager.acquire_device()
        log_event("DeviceAcquisition", f"Acquired device {device.device_id} for task '{task_instruction}'")
        try:
            # Simulate shadow run for a specific task if needed
            if "profile" in task_instruction.lower():
                log_event("ShadowRun", f"Running '{task_instruction}' in shadow mode with a new policy.")
                new_policy = {"strategy": "exploit", "params": {"epsilon": 0.01}}
                policy_manager.register_policy("new_profile_policy", new_policy)
                await shadow_run(new_policy, run_agent_task, device, task_instruction)
            else:
                await run_agent_task(device, task_instruction)
        finally:
            await device_manager.release_device(device)

    log_event("OrchestratorEnd", "AutoRL Agent Orchestrator finished.")

if __name__ == "__main__":
    # Ensure Tesseract is installed and configured for pytesseract
    # For Debian/Ubuntu: sudo apt-get install tesseract-ocr
    # For macOS: brew install tesseract
    # For Windows: Install from https://tesseract-ocr.github.io/tessdoc/Installation.html
    # Then set the TESSERACT_CMD environment variable if not in PATH
    # os.environ["TESSERACT_CMD"] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    asyncio.run(main())

