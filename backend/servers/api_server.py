
from flask import Flask, jsonify, request
from flask_cors import CORS
import asyncio
from datetime import datetime
import threading
import subprocess
import os

from runtime.device_manager import Device, DeviceManager
from production_readiness.logging_utils import log_event, store_secure_log, decrypt_secure_log
from production_readiness.metrics_server import start_metrics_server, task_success_counter, task_failure_counter, current_active_tasks_gauge, avg_runtime_gauge
# PolicyManager not available - using mock
try:
    from rl.policy_manager import PolicyManager
except ImportError:
    # Mock PolicyManager for now
    class PolicyManager:
        def __init__(self):
            self.policies = {}
        def register_policy(self, name, policy):
            self.policies[name] = policy
from llm.llm_planner import LLMPlanner
from perception.visual_perception import VisualPerception

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize components
device_manager = DeviceManager()
policy_manager = PolicyManager()
llm_planner = LLMPlanner()
visual_perception = VisualPerception()

# Add some dummy devices (these should ideally be managed by the orchestrator)
device_manager.add_device(Device("emulator-5554", "android", False))
device_manager.add_device(Device("iPhone 15", "ios", False))
device_manager.add_device(Device("emulator-5556", "android", False))

# In-memory store for activity log (for API exposure)
api_activity_log = []

# Function to run the main agent orchestrator in a separate thread/process
def run_orchestrator_in_background():
    # This assumes main.py is runnable and uses the same environment
    # In a production setup, this would be a more robust process management
    log_event("API_Server", "Starting main.py orchestrator in background.")
    # Use Popen to run main.py as a separate process
    subprocess.Popen(["python3", "main.py"])

# Start Prometheus metrics server in a separate thread
def start_metrics_thread():
    start_metrics_server(port=9000)

metrics_thread = threading.Thread(target=start_metrics_thread)
metrics_thread.daemon = True
metrics_thread.start()

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    log_event("API_Health", "Health check requested.")
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route("/api/devices", methods=["GET"])
def get_devices():
    """Get all connected devices"""
    devices_list = [
        {
            "id": device.device_id,
            "platform": device.platform,
            "status": "idle" if device.session is None else "running", # Simplified status
            "currentTask": None # This would be populated by actual agent state
        }
        for device in device_manager.devices
    ]
    return jsonify(devices_list)

@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    """Get all tasks (mocked for now) """
    # In a real system, this would query a task queue/database
    mock_tasks = [
        {"id": 1, "name": "Login and Browse", "status": "completed", "device": "emulator-5554", "duration": "12s"},
        {"id": 2, "name": "Update Profile", "status": "running", "device": "iPhone 15", "duration": "8s"},
        {"id": 3, "name": "Search Products", "status": "queued", "device": None, "duration": "-"},
        {"id": 4, "name": "Checkout Flow", "status": "queued", "device": None, "duration": "-"}
    ]
    return jsonify(mock_tasks)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Get system metrics from Prometheus client"""
    metrics_data = {
        "totalTasksSuccess": task_success_counter._value,
        "totalTasksFailure": task_failure_counter._value,
        "activeTasks": current_active_tasks_gauge._value,
        "avgRuntime": avg_runtime_gauge._value,
        "successRate": (task_success_counter._value / (task_success_counter._value + task_failure_counter._value) * 100)
                       if (task_success_counter._value + task_failure_counter._value) > 0 else 0
    }
    return jsonify(metrics_data)

@app.route("/api/activity", methods=["GET"])
def get_activity():
    """Get recent activity log (mocked for now, would read from log file) """
    # For a real implementation, this would parse the events.log file
    mock_activity = [
        {"timestamp": datetime.now().strftime("%H:%M:%S"), "message": "API Server started", "level": "info"},
        {"timestamp": datetime.now().strftime("%H:%M:%S"), "message": "Metrics server running on :9000", "level": "info"},
        {"timestamp": datetime.now().strftime("%H:%M:%S"), "message": "Agent orchestrator started successfully", "level": "success"},
    ]
    return jsonify(mock_activity)

@app.route("/api/agent/start", methods=["POST"])
def start_agent():
    """Start the AI agent orchestrator"""
    log_event("API_AgentControl", "Agent start requested from API.")
    # In a real system, this would signal the main.py process to start/resume
    # For this example, we will just log and return success.
    # A more robust solution would use a message queue or shared state.
    run_orchestrator_in_background()
    return jsonify({"status": "starting", "message": "Agent orchestrator start initiated."})

@app.route("/api/agent/stop", methods=["POST"])
def stop_agent():
    """Stop the AI agent orchestrator"""
    log_event("API_AgentControl", "Agent stop requested from API.")
    # In a real system, this would signal the main.py process to stop
    return jsonify({"status": "stopping", "message": "Agent orchestrator stop initiated."})

@app.route("/api/policies", methods=["GET"])
def get_policies():
    """Get all registered policies"""
    # This would integrate with PolicyManager
    return jsonify([
        {"name": "initial_policy", "version": "2025-01-15T10:30:00", "strategy": "explore"},
        {"name": "optimized_policy", "version": "2025-01-16T14:20:00", "strategy": "exploit"}
    ])

@app.route("/api/plan", methods=["POST"])
def create_plan():
    """Create a new plan using the LLM planner"""
    data = request.get_json()
    user_intent = data.get("user_intent")
    current_ui_state = data.get("current_ui_state")
    if not user_intent or not current_ui_state:
        return jsonify({"error": "user_intent and current_ui_state are required"}), 400
    
    log_event("API_LLM", f"Planning for intent: {user_intent}")
    try:
        plan = llm_planner.generate_plan(user_intent, current_ui_state)
        return jsonify({"status": "success", "plan": plan})
    except Exception as e:
        log_event("API_LLM", f"Error generating plan: {e}", level="error")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/perception/analyze", methods=["POST"])
def analyze_ui():
    """Analyze UI screenshot using visual perception module"""
    data = request.get_json()
    image_path = data.get("image_path") # Expecting a path to a local image or base64 encoded image
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400
    
    log_event("API_Perception", f"Analyzing UI from: {image_path}")
    try:
        ui_elements = visual_perception.analyze_screenshot(image_path)
        return jsonify({"status": "success", "ui_elements": ui_elements})
    except Exception as e:
        log_event("API_Perception", f"Error analyzing UI: {e}", level="error")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    log_event("API_Server", "Starting AutoRL API Server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False) # use_reloader=False to prevent running twice

