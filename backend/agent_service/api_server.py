from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
from .auth import get_user
from .omh_auth import get_omh_user, get_omh_user_with_location, require_role, omh_client
from .plugin_manager import PluginManager
from .llm_adapter import LLMSandbox
from . import mock_api
from .device_manager import AsyncDeviceManager
from .store import record_user_task
from prometheus_client import make_asgi_app, start_http_server
from starlette.middleware.wsgi import WSGIMiddleware

app = FastAPI(title="Agent Service")
app.include_router(mock_api.router)

# Initialize a shared AsyncDeviceManager instance for scheduling
device_manager = AsyncDeviceManager(max_concurrent=4)

# Mount Prometheus ASGI app at /metrics
try:
    metrics_app = make_asgi_app()
    app.mount("/metrics", WSGIMiddleware(metrics_app))
except Exception:
    # make_asgi_app may not be available in older prometheus-client versions
    pass

# Optionally start a standalone Prometheus metrics server on 9000 (useful in Docker)
try:
    start_http_server(9000)
except Exception:
    pass

# Enable CORS for frontend dev servers (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_PRICE = 0.10


class UserModel(BaseModel):
    username: str


class TaskRequest(BaseModel):
    instruction: str
    device_id: str
    parameters: Optional[Dict[str, Any]] = {}


@app.post("/api/v1/execute")
async def execute_task(req: TaskRequest, request: Request, user=Depends(get_user)):
    # get_user now returns (username, user_record)
    username, user_record = user
    if user_record["tasks_used"] >= user_record["quota"]:
        raise HTTPException(status_code=402, detail="Quota exceeded. Upgrade required.")
    # schedule async task on a device
    schedule_res = await device_manager.schedule_task(req.instruction, device_id=req.device_id, timeout=60, retries=2)
    # billing/usage
    user_record["tasks_used"] += 1
    task_id = schedule_res.get("result", {}).get("task_id") or str(uuid.uuid4())
    record = {"task_id": task_id, "instruction": req.instruction, "schedule_res": schedule_res}
    record_user_task(username, record)
    return {"task_id": task_id, "status": "scheduled", "schedule_res": schedule_res, "msg": f"1 task charged (${TASK_PRICE:.2f} if pay-per-use)"}


@app.post("/api/v1/subscription/upgrade")
def upgrade(request: Request, user=Depends(get_user)):
    username, user_record = user
    user_record["subscription"] = "premium"
    user_record["quota"] = 100
    return {"status": "upgraded", "quota": user_record["quota"]}


@app.get("/api/v1/user/history")
def history(request: Request, user=Depends(get_user)):
    username, user_record = user
    return {"history": user_record.get("history", [])}


# Natural language endpoint using a pluggable LLM adapter
@app.post("/api/v1/nl/execute")
def nl_execute(payload: Dict[str, Any], request: Request, user=Depends(get_user)):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in body")
    llm = LLMSandbox()
    structured = llm.parse_to_task(query)
    # validate structured has the fields for TaskRequest
    try:
        task = TaskRequest(**structured)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid task: {e}")
    return execute_task(task, request, user)


@app.get("/api/v1/plugins")
def list_plugins():
    pm = PluginManager()
    return {"plugins": [p.metadata for p in pm.discover_plugins()]}


@app.post("/api/v1/plugins/run/{plugin_name}")
def run_plugin(plugin_name: str, payload: Dict[str, Any], request: Request, user=Depends(get_user)):
    pm = PluginManager()
    result = pm.run_plugin(plugin_name, payload)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("msg"))
    return result


@app.post("/api/v1/plugins/purchase/{plugin_name}")
def purchase_plugin(plugin_name: str, request: Request, user=Depends(get_user)):
    username, user_record = user
    # mock purchase flow: deduct quota or mark purchase -- replace with a real payment flow
    return {"success": True, "plugin": plugin_name, "owner": username}


# ============================================================================
# OMH Integration Endpoints
# ============================================================================

@app.post("/api/v1/auth/omh/login")
async def omh_login(username: str, password: str):
    """
    Login with OMH OAuth credentials
    Returns access token for subsequent requests
    """
    try:
        auth_data = omh_client.authenticate_user(username, password)
        return {
            "success": True,
            "access_token": auth_data["access_token"],
            "token_type": auth_data["token_type"],
            "expires_in": auth_data["expires_in"],
            "scope": auth_data["scope"]
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/api/v1/user/profile/omh")
async def get_omh_profile(user=Depends(get_omh_user)):
    """
    Get authenticated user profile via OMH
    Use: Authorization: Bearer <omh_token>
    """
    username, user_record = user
    return {
        "username": username,
        "profile": user_record,
        "auth_method": user_record.get("auth_method", "unknown")
    }


@app.post("/api/v1/execute/location-aware")
async def execute_location_aware_task(
    req: TaskRequest,
    request: Request,
    user_with_location=Depends(get_omh_user_with_location)
):
    """
    Execute a task with location context from OMH
    Automatically includes user location in task execution
    """
    username, user_record, location = user_with_location
    
    # Check quota
    if user_record["tasks_used"] >= user_record["quota"]:
        raise HTTPException(status_code=402, detail="Quota exceeded. Upgrade required.")
    
    # Add location context to task parameters
    task_params = req.parameters or {}
    if location:
        task_params["location_context"] = location.get("location")
        task_params["location_address"] = location.get("location", {}).get("address")
    
    # Schedule task
    schedule_res = await device_manager.schedule_task(
        req.instruction,
        device_id=req.device_id,
        timeout=60,
        retries=2
    )
    
    # Record task
    user_record["tasks_used"] += 1
    task_id = schedule_res.get("result", {}).get("task_id") or str(uuid.uuid4())
    record = {
        "task_id": task_id,
        "instruction": req.instruction,
        "schedule_res": schedule_res,
        "location_context": location
    }
    record_user_task(username, record)
    
    return {
        "task_id": task_id,
        "status": "scheduled",
        "schedule_res": schedule_res,
        "location_context": location,
        "msg": f"1 task charged (${TASK_PRICE:.2f} if pay-per-use)"
    }


@app.get("/api/v1/location/current")
async def get_current_location(user=Depends(get_omh_user_with_location)):
    """Get user's current location from OMH"""
    username, user_record, location = user
    
    if not location:
        raise HTTPException(
            status_code=404,
            detail="Location not available. Ensure OMH authentication is used."
        )
    
    return {
        "username": username,
        "location": location.get("location"),
        "timestamp": location.get("timestamp")
    }


@app.get("/api/v1/location/nearby-tasks")
async def get_nearby_tasks(user=Depends(get_omh_user_with_location)):
    """
    Get tasks available near user's current location
    Example use case: Find nearby restaurants, stores, etc.
    """
    username, user_record, location = user
    
    if not location:
        return {"tasks": [], "message": "Location not available"}
    
    loc_data = location.get("location", {})
    address = loc_data.get("address", "Unknown")
    
    # Mock nearby tasks based on location
    nearby_tasks = [
        {
            "id": "task_001",
            "name": f"Find restaurants near {address}",
            "type": "search",
            "distance": "0.5 km"
        },
        {
            "id": "task_002",
            "name": f"Book appointment at {address}",
            "type": "booking",
            "distance": "1.2 km"
        },
        {
            "id": "task_003",
            "name": f"Check store hours near {address}",
            "type": "information",
            "distance": "0.8 km"
        }
    ]
    
    return {
        "location": loc_data,
        "tasks": nearby_tasks,
        "count": len(nearby_tasks)
    }


@app.post("/api/v1/admin/reset-quota", dependencies=[Depends(require_role("admin"))])
async def admin_reset_quota(username: str):
    """
    Admin-only endpoint to reset user quota
    Requires admin role in OMH authentication
    """
    # This would update your user database
    return {
        "success": True,
        "message": f"Quota reset for user: {username}",
        "admin_action": True
    }
