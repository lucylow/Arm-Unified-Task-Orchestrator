"""
Master Backend Server - AutoRL
Consolidates ALL backend functionality into one unified server
- REST API endpoints
- WebSocket real-time updates
- AI Agent orchestration
- Plugin management
- Device management
- Authentication & Authorization
- OMH integration
- Mock data generation
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
import os

# Import master agent system
from .master_agent_system import MasterAgentOrchestrator, AgentContext, AgentState

# Conditional imports based on availability
PRODUCTION_MODE = os.getenv("AUTORL_MODE", "demo") == "production"

try:
    if PRODUCTION_MODE:
        from runtime.device_manager import Device, DeviceManager
        from perception.visual_perception import VisualPerception
        from llm.llm_planner import LLMPlanner
        from tools.action_execution import ActionExecutor
        from error_handling.recovery import RecoveryManager
        try:
            from rl.policy_manager import PolicyManager
        except ImportError:
            # Mock PolicyManager for now
            class PolicyManager:
                def __init__(self):
                    self.policies = {}
                    self.active_policy_name = None
                def register_policy(self, name, policy, is_active=False):
                    self.policies[name] = {"policy_data": policy, "version": "1.0"}
                    if is_active:
                        self.active_policy_name = name
        from agents.registry import PluginRegistry
        from production_readiness.metrics_server import start_metrics_server, record_task_metrics
        print("✅ Production modules loaded")
    else:
        # Mock imports for demo mode
        Device = None
        DeviceManager = None
        print("ℹ️  Running in DEMO mode")
except ImportError as e:
    PRODUCTION_MODE = False
    print(f"⚠️  Some production modules not available: {e}")
    print("ℹ️  Running in DEMO mode")

# Initialize FastAPI
app = FastAPI(
    title="AutoRL Master Backend",
    description="Unified backend for AutoRL platform",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MasterBackend")

# ============================================================================
# GLOBAL STATE
# ============================================================================

class GlobalState:
    """Global application state"""
    
    def __init__(self):
        self.mode = "production" if PRODUCTION_MODE else "demo"
        self.orchestrator: Optional[MasterAgentOrchestrator] = None
        self.device_manager = None
        self.policy_manager = None
        self.plugin_registry = None
        self.websocket_manager = None
        self.active_tasks: Dict[str, Dict] = {}
        self.task_history: List[Dict] = []
        self.activity_log: List[Dict] = []
        self.metrics = {
            'total_tasks_success': 0,
            'total_tasks_failure': 0,
            'tasks_in_progress': 0,
            'avg_task_runtime': 0.0
        }
    
    def add_activity(self, message: str, level: str = "info"):
        """Add activity log entry"""
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "level": level
        }
        self.activity_log.insert(0, entry)
        if len(self.activity_log) > 200:
            self.activity_log.pop()
        logger.info(f"[{level.upper()}] {message}")

state = GlobalState()

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class WebSocketConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        state.add_activity(f"WebSocket client connected (total: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        state.add_activity(f"WebSocket disconnected (remaining: {len(self.active_connections)})")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                dead_connections.append(connection)
        
        for dead in dead_connections:
            self.disconnect(dead)
    
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self.disconnect(websocket)

ws_manager = WebSocketConnectionManager()
state.websocket_manager = ws_manager

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TaskRequest(BaseModel):
    instruction: str
    device_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class DeviceInfo(BaseModel):
    id: str
    platform: str
    status: str
    is_real: bool = False
    battery: Optional[int] = None
    current_task: Optional[str] = None

class MetricsResponse(BaseModel):
    total_tasks_success: int
    total_tasks_failure: int
    tasks_in_progress: int
    avg_task_runtime_seconds: float
    success_rate: float

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    logger.info("🚀 Starting AutoRL Master Backend")
    logger.info(f"📊 Mode: {state.mode.upper()}")
    
    # Initialize master agent orchestrator
    if PRODUCTION_MODE:
        try:
            # Initialize real components
            visual_perception = VisualPerception()
            llm_planner = LLMPlanner()
            # action_executor and recovery_manager initialized per task
            
            state.orchestrator = MasterAgentOrchestrator(
                visual_perception=visual_perception,
                llm_planner=llm_planner
            )
            
            # Initialize device manager
            state.device_manager = DeviceManager()
            await state.device_manager.add_device(Device("emulator-5554", "android", False))
            await state.device_manager.add_device(Device("iPhone_15", "ios", False))
            await state.device_manager.add_device(Device("Pixel_7", "android", False))
            
            # Initialize policy manager
            state.policy_manager = PolicyManager()
            state.policy_manager.register_policy("initial_policy", {
                "strategy": "explore",
                "params": {"epsilon": 0.1}
            }, is_active=True)
            
            # Initialize plugin registry
            state.plugin_registry = PluginRegistry()
            state.plugin_registry.discover_plugins()
            state.plugin_registry.initialize_plugins()
            
            state.add_activity(f"Initialized {len(state.device_manager.devices)} devices", "success")
            state.add_activity(f"Loaded {len(state.plugin_registry.plugins)} plugins", "success")
            
        except Exception as e:
            logger.error(f"Error initializing production components: {e}")
            state.add_activity(f"Error in initialization: {str(e)}", "error")
    else:
        # Demo mode - no external dependencies
        state.orchestrator = MasterAgentOrchestrator()
        state.add_activity("Running in DEMO MODE with mock data", "warning")
    
    logger.info("✅ AutoRL Master Backend ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔴 Shutting down AutoRL Master Backend")
    
    if state.device_manager:
        await state.device_manager.shutdown()
    
    if state.plugin_registry:
        state.plugin_registry.shutdown_all()
    
    logger.info("✅ Shutdown complete")

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": state.mode,
        "timestamp": datetime.now().isoformat(),
        "devices_connected": len(state.device_manager.devices) if state.device_manager else 0,
        "active_tasks": len(state.active_tasks),
        "orchestrator_status": "initialized" if state.orchestrator else "not_initialized"
    }

@app.get("/api/status")
async def get_status():
    """Get detailed system status"""
    return {
        "mode": state.mode,
        "orchestrator": state.orchestrator.get_agent_status() if state.orchestrator else {"status": "not_initialized"},
        "devices": len(state.device_manager.devices) if state.device_manager else 0,
        "plugins": len(state.plugin_registry.plugins) if state.plugin_registry else 0,
        "websocket_clients": len(ws_manager.active_connections),
        "active_tasks": len(state.active_tasks),
        "total_tasks_executed": state.metrics['total_tasks_success'] + state.metrics['total_tasks_failure']
    }

# ============================================================================
# DEVICE ENDPOINTS
# ============================================================================

@app.get("/api/devices", response_model=List[DeviceInfo])
async def get_devices():
    """Get all connected devices"""
    if PRODUCTION_MODE and state.device_manager:
        devices = []
        for device in state.device_manager.devices:
            devices.append(DeviceInfo(
                id=device.device_id,
                platform=device.platform,
                status="active" if device.session else "idle",
                is_real=device.is_real,
                current_task=None
            ))
        return devices
    else:
        # Mock devices for demo mode
        return [
            DeviceInfo(id="android_pixel_7", platform="Android", status="connected", is_real=False, battery=85),
            DeviceInfo(id="iphone_14", platform="iOS", status="connected", is_real=False, battery=92),
            DeviceInfo(id="emulator-5554", platform="Android", status="idle", is_real=False, battery=100)
        ]

# ============================================================================
# TASK ENDPOINTS
# ============================================================================

@app.get("/api/tasks")
async def get_tasks():
    """Get task history"""
    return state.task_history[-100:]  # Last 100 tasks

@app.post("/api/plan/detailed")
async def create_detailed_plan(task_req: TaskRequest):
    """Generate a detailed plan without executing it - for preview and interaction"""
    try:
        # Import LLM planner
        if PRODUCTION_MODE:
            from llm.llm_planner import LLMPlanner
            planner = LLMPlanner()
        else:
            # Demo mode - create mock planner
            from llm.llm_planner import LLMPlanner
            planner = LLMPlanner()
        
        # Generate detailed plan
        detailed_plan = planner.generate_detailed_plan(
            instruction=task_req.instruction,
            ui_state=task_req.parameters.get("ui_state") if task_req.parameters else None
        )
        
        state.add_activity(f"Detailed plan generated: {task_req.instruction[:50]}...", "info")
        
        return JSONResponse({
            "status": "success",
            "plan": detailed_plan
        })
    except Exception as e:
        logger.error(f"Error generating detailed plan: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(task_req: TaskRequest):
    """Create and execute a new task"""
    task_id = str(uuid.uuid4())
    
    task_data = {
        "id": task_id,
        "instruction": task_req.instruction,
        "device_id": task_req.device_id or "auto",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "parameters": task_req.parameters
    }
    
    state.active_tasks[task_id] = task_data
    state.task_history.append(task_data)
    state.metrics['tasks_in_progress'] += 1
    state.add_activity(f"Task created: {task_req.instruction[:50]}...", "info")
    
    # Execute task asynchronously
    asyncio.create_task(execute_task_workflow(task_id, task_req))
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        message=f"Task '{task_req.instruction}' queued for execution"
    )

async def execute_task_workflow(task_id: str, task_req: TaskRequest):
    """Execute task using master agent orchestrator"""
    try:
        task_data = state.active_tasks[task_id]
        task_data["status"] = "running"
        start_time = time.time()
        
        # Broadcast task start
        await ws_manager.broadcast({
            "event": "task_started",
            "task_id": task_id,
            "instruction": task_req.instruction,
            "device_id": task_req.device_id or "auto"
        })
        
        # Execute through master orchestrator
        context = await execute_with_orchestrator(task_id, task_req)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update task status
        task_data["status"] = context.state.value
        task_data["completed_at"] = datetime.now().isoformat()
        task_data["duration"] = duration
        
        # Update metrics
        if context.state == AgentState.COMPLETED:
            state.metrics['total_tasks_success'] += 1
            await ws_manager.broadcast({
                "event": "task_completed",
                "task_id": task_id,
                "status": "completed",
                "duration": duration
            })
            state.add_activity(f"Task completed: {task_req.instruction[:50]}...", "success")
        else:
            state.metrics['total_tasks_failure'] += 1
            await ws_manager.broadcast({
                "event": "task_failed",
                "task_id": task_id,
                "error": context.error or "Unknown error"
            })
            state.add_activity(f"Task failed: {task_req.instruction[:50]}...", "error")
        
        # Remove from active tasks
        if task_id in state.active_tasks:
            del state.active_tasks[task_id]
        state.metrics['tasks_in_progress'] -= 1
        
    except Exception as e:
        logger.error(f"Error executing task {task_id}: {e}")
        state.add_activity(f"Task error: {str(e)}", "error")
        
        if task_id in state.active_tasks:
            state.active_tasks[task_id]["status"] = "failed"
            state.active_tasks[task_id]["error"] = str(e)
            del state.active_tasks[task_id]
        
        state.metrics['total_tasks_failure'] += 1
        state.metrics['tasks_in_progress'] -= 1
        
        await ws_manager.broadcast({
            "event": "task_failed",
            "task_id": task_id,
            "error": str(e)
        })

async def execute_with_orchestrator(task_id: str, task_req: TaskRequest) -> AgentContext:
    """Execute task with agent orchestrator and broadcast updates"""
    
    # Get driver if in production mode
    driver = None
    if PRODUCTION_MODE and state.device_manager:
        device = await state.device_manager.acquire_device()
        driver = device.session if device else None
    
    # Create hooks to broadcast agent state changes
    original_execute = {}
    for agent_name, agent in state.orchestrator.agents.items():
        original_execute[agent_name] = agent.execute
        
                async def create_broadcast_wrapper(name, orig_func):
            async def wrapper(context):
                # Broadcast agent state
                event_map = {
                    'perception': 'perception',
                    'planning': 'planning',
                    'execution': 'execution_start',
                    'recovery': 'recovery_execute',
                    'reflection': 'recovery_analyze'
                }
                
                event = event_map.get(name, name)
                
                # Enhanced broadcast with step information
                broadcast_data = {
                    "event": event,
                    "task_id": task_id,
                    "text": f"{name.capitalize()} agent active...",
                    "agent": name,
                    "timestamp": time.time()
                }
                
                # Add step information if available in context
                if hasattr(context, 'current_step'):
                    broadcast_data["step"] = context.current_step
                    broadcast_data["total_steps"] = getattr(context, 'total_steps', 0)
                
                await ws_manager.broadcast(broadcast_data)
                
                # Execute original function
                result = await orig_func(context)
                
                # Broadcast step completion for execution agent
                if name == 'execution' and hasattr(context, 'current_step'):
                    await ws_manager.broadcast({
                        "event": "step_completed",
                        "task_id": task_id,
                        "step": context.current_step,
                        "total_steps": getattr(context, 'total_steps', 0),
                        "timestamp": time.time()
                    })
                
                # Broadcast completion if appropriate
                if result.error:
                    await ws_manager.broadcast({
                        "event": "error",
                        "task_id": task_id,
                        "text": result.error,
                        "severity": "warning",
                        "timestamp": time.time()
                    })
                
                return result
            return wrapper
        
        agent.execute = await create_broadcast_wrapper(agent_name, original_execute[agent_name])
    
    # Execute the task
    context = await state.orchestrator.execute_task(
        task_id=task_id,
        instruction=task_req.instruction,
        device_id=task_req.device_id or "auto",
        driver=driver
    )
    
    # Restore original execute methods
    for agent_name, orig_func in original_execute.items():
        state.orchestrator.agents[agent_name].execute = orig_func
    
    # Release device if in production mode
    if PRODUCTION_MODE and state.device_manager and driver:
        # Release device back to pool
        pass
    
    return context

# ============================================================================
# METRICS ENDPOINT
# ============================================================================

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics"""
    total = state.metrics['total_tasks_success'] + state.metrics['total_tasks_failure']
    success_rate = (state.metrics['total_tasks_success'] / total * 100) if total > 0 else 0
    
    return MetricsResponse(
        total_tasks_success=state.metrics['total_tasks_success'],
        total_tasks_failure=state.metrics['total_tasks_failure'],
        tasks_in_progress=state.metrics['tasks_in_progress'],
        avg_task_runtime_seconds=state.metrics.get('avg_task_runtime', 0.0),
        success_rate=round(success_rate, 1)
    )

# ============================================================================
# ACTIVITY LOG ENDPOINT
# ============================================================================

@app.get("/api/activity")
async def get_activity():
    """Get recent activity log"""
    return state.activity_log

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive client messages
            data = await websocket.receive_text()
            # Echo back or handle client messages
            await ws_manager.send_to_client(websocket, {
                "event": "pong",
                "timestamp": time.time()
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# ============================================================================
# PLUGIN ENDPOINTS (if available)
# ============================================================================

@app.get("/api/plugins")
async def get_plugins():
    """Get all loaded plugins"""
    if state.plugin_registry:
        plugins = []
        for name, meta in state.plugin_registry.plugin_meta.items():
            plugins.append({
                "name": name,
                "status": meta.get("status"),
                "path": meta.get("path")
            })
        return plugins
    else:
        return [
            {"name": "vision_boost", "status": "initialized", "path": "plugins/vision_boost.py"},
            {"name": "error_recovery", "status": "initialized", "path": "plugins/error_recovery.py"}
        ]

# ============================================================================
# POLICY ENDPOINTS (if available)
# ============================================================================

@app.get("/api/policies")
async def get_policies():
    """Get all RL policies"""
    if state.policy_manager:
        policies = []
        for name, data in state.policy_manager.policies.items():
            policies.append({
                "name": name,
                "version": data.get("version"),
                "is_active": name == state.policy_manager.active_policy_name,
                "strategy": data.get("policy_data", {}).get("strategy", "unknown")
            })
        return policies
    else:
        return [
            {"name": "initial_policy", "is_active": True, "strategy": "explore"},
            {"name": "optimized_policy", "is_active": False, "strategy": "exploit"}
        ]

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 AutoRL Master Backend Server")
    print("="*60)
    print(f"📊 Mode: {state.mode.upper()}")
    print(f"🌐 API: http://localhost:5000")
    print(f"📚 Docs: http://localhost:5000/docs")
    print(f"🔌 WebSocket: ws://localhost:5000/ws")
    print(f"📈 Metrics: http://localhost:8000/metrics (if Prometheus enabled)")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )

