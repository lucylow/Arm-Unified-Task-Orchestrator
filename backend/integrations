"""
OMH Integration API Endpoints for AutoRL

FastAPI routes for Open Mobile Hub authentication and maps services.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import sys
from pathlib import Path

# Add plugins to path
sys.path.insert(0, str(Path(__file__).parent.parent / "plugins"))

from omh_auth_plugin import omh_auth_plugin
from omh_maps_plugin import omh_maps_plugin

logger = logging.getLogger("autorl.omh_api")

# Initialize plugins
omh_auth_plugin.initialize({})
omh_maps_plugin.initialize({})

router = APIRouter(prefix="/api/v1/omh", tags=["OMH Integration"])


# ==================== Request/Response Models ====================

class AuthURLRequest(BaseModel):
    redirect_uri: str
    state: Optional[str] = None


class TokenValidationRequest(BaseModel):
    token: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class GeocodeRequest(BaseModel):
    address: str


class ReverseGeocodeRequest(BaseModel):
    latitude: float
    longitude: float


class DistanceRequest(BaseModel):
    origin: Dict[str, float]  # {"lat": float, "lng": float}
    destination: Dict[str, float]


class GeofenceRequest(BaseModel):
    location: Dict[str, float]
    geofence: Dict[str, Any]


class NearbyPlacesRequest(BaseModel):
    latitude: float
    longitude: float
    radius: int = 1000
    place_type: Optional[str] = None


class LocationTriggerRequest(BaseModel):
    task_id: str
    geofence: Dict[str, Any]
    trigger_type: str = "enter"


# ==================== Authentication Dependency ====================

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from OMH token.
    Use in routes like: user=Depends(get_current_user)
    """
    if not authorization:
        # Allow unauthenticated access in demo mode
        return {
            "user_id": "demo_user",
            "username": "demo",
            "authenticated": False,
            "mock_mode": True
        }
    
    token = authorization.replace("Bearer ", "")
    result = omh_auth_plugin.process({
        "action": "validate_token",
        "token": token
    })
    
    if not result.get("valid"):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return {
        **result,
        "authenticated": True
    }


# ==================== OMH Auth Endpoints ====================

@router.get("/auth/url")
async def get_auth_url(redirect_uri: str, state: Optional[str] = None):
    """
    Get OMH OAuth2 authorization URL for user login.
    
    Example:
        GET /api/v1/omh/auth/url?redirect_uri=http://localhost:3000/callback
    """
    result = omh_auth_plugin.process({
        "action": "get_auth_url",
        "redirect_uri": redirect_uri,
        "state": state
    })
    return result


@router.post("/auth/validate")
async def validate_token(request: TokenValidationRequest):
    """
    Validate an OMH access token.
    
    Example:
        POST /api/v1/omh/auth/validate
        {"token": "your_access_token"}
    """
    result = omh_auth_plugin.process({
        "action": "validate_token",
        "token": request.token
    })
    return result


@router.post("/auth/refresh")
async def refresh_token(request: TokenRefreshRequest):
    """
    Refresh an OMH access token.
    
    Example:
        POST /api/v1/omh/auth/refresh
        {"refresh_token": "your_refresh_token"}
    """
    result = omh_auth_plugin.process({
        "action": "refresh_token",
        "refresh_token": request.refresh_token
    })
    return result


@router.get("/auth/user")
async def get_user_info(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current authenticated user information.
    Requires Authorization header with Bearer token.
    
    Example:
        GET /api/v1/omh/auth/user
        Headers: Authorization: Bearer your_token
    """
    if not user.get("authenticated"):
        return {
            **user,
            "message": "Demo mode - no authentication required"
        }
    return user


# ==================== OMH Maps Endpoints ====================

@router.post("/maps/geocode")
async def geocode_address(
    request: GeocodeRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Convert address to coordinates.
    
    Example:
        POST /api/v1/omh/maps/geocode
        {"address": "San Francisco, CA"}
    """
    result = omh_maps_plugin.process({
        "action": "geocode",
        "address": request.address
    })
    return result


@router.post("/maps/reverse-geocode")
async def reverse_geocode(
    request: ReverseGeocodeRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Convert coordinates to address.
    
    Example:
        POST /api/v1/omh/maps/reverse-geocode
        {"latitude": 37.7749, "longitude": -122.4194}
    """
    result = omh_maps_plugin.process({
        "action": "reverse_geocode",
        "latitude": request.latitude,
        "longitude": request.longitude
    })
    return result


@router.post("/maps/distance")
async def calculate_distance(
    request: DistanceRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Calculate distance between two locations.
    
    Example:
        POST /api/v1/omh/maps/distance
        {
            "origin": {"lat": 37.7749, "lng": -122.4194},
            "destination": {"lat": 40.7128, "lng": -74.0060}
        }
    """
    result = omh_maps_plugin.process({
        "action": "calculate_distance",
        "origin": request.origin,
        "destination": request.destination
    })
    return result


@router.post("/maps/geofence")
async def check_geofence(
    request: GeofenceRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Check if location is within a geofence.
    
    Example:
        POST /api/v1/omh/maps/geofence
        {
            "location": {"lat": 37.7749, "lng": -122.4194},
            "geofence": {
                "center": {"lat": 37.7749, "lng": -122.4194},
                "radius_meters": 500
            }
        }
    """
    result = omh_maps_plugin.process({
        "action": "check_geofence",
        "location": request.location,
        "geofence": request.geofence
    })
    return result


@router.post("/maps/nearby")
async def get_nearby_places(
    request: NearbyPlacesRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Find nearby places of interest.
    
    Example:
        POST /api/v1/omh/maps/nearby
        {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "radius": 1000,
            "place_type": "restaurant"
        }
    """
    result = omh_maps_plugin.process({
        "action": "get_nearby_places",
        "latitude": request.latitude,
        "longitude": request.longitude,
        "radius": request.radius,
        "type": request.place_type
    })
    return result


@router.post("/maps/location-trigger")
async def create_location_trigger(
    request: LocationTriggerRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a location-based task trigger.
    
    Example:
        POST /api/v1/omh/maps/location-trigger
        {
            "task_id": "task_123",
            "geofence": {
                "center": {"lat": 37.7749, "lng": -122.4194},
                "radius_meters": 500
            },
            "trigger_type": "enter"
        }
    """
    result = omh_maps_plugin.create_location_trigger(
        request.task_id,
        request.geofence,
        request.trigger_type
    )
    return result


# ==================== Status/Health Endpoints ====================

@router.get("/status")
async def omh_status():
    """
    Get OMH integration status.
    """
    return {
        "auth": {
            "enabled": not omh_auth_plugin.mock_mode,
            "mock_mode": omh_auth_plugin.mock_mode
        },
        "maps": {
            "enabled": not omh_maps_plugin.mock_mode,
            "mock_mode": omh_maps_plugin.mock_mode
        },
        "message": "OMH integration active" if not (omh_auth_plugin.mock_mode and omh_maps_plugin.mock_mode) else "Running in demo/mock mode"
    }

