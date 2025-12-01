"""
OMH (Open Mobile Hub) Mock Server
==================================
FastAPI server providing mock endpoints for OAuth2 authentication,
user profiles, and location services for testing AutoRL integration.
"""

from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
from datetime import datetime, timedelta

app = FastAPI(
    title="OMH Mock Server",
    description="Mock server for Open Mobile Hub OAuth and API testing",
    version="1.0.0"
)

# CORS middleware for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mock data
MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_data", "omh_mock_data.json")

def load_mock_data():
    with open(MOCK_DATA_PATH, 'r') as f:
        return json.load(f)

MOCK_DATA = load_mock_data()


# ============================================================================
# Request/Response Models
# ============================================================================

class TokenRequest(BaseModel):
    grant_type: str
    username: Optional[str] = None
    password: Optional[str] = None
    refresh_token: Optional[str] = None
    assertion: Optional[str] = None
    client_assertion: Optional[str] = None
    client_assertion_type: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str
    scope: str
    id_token: Optional[str] = None


class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[str]
    profile_picture_url: str
    account_status: str
    last_login: str


class Location(BaseModel):
    latitude: float
    longitude: float
    address: str
    place_id: str


class LocationResponse(BaseModel):
    location: Location


# ============================================================================
# Authentication Helpers
# ============================================================================

def verify_token(authorization: Optional[str] = Header(None)) -> str:
    """Extract and verify bearer token from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    token = authorization.replace("Bearer ", "")
    
    # Check if token exists in our mappings
    if token not in MOCK_DATA.get("token_mappings", {}):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return token


def get_user_from_token(token: str) -> dict:
    """Get user profile from token"""
    user_id = MOCK_DATA["token_mappings"].get(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    for user in MOCK_DATA["user_profiles"]:
        if user["user_id"] == user_id:
            return user
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found"
    )


# ============================================================================
# OAuth2 Endpoints
# ============================================================================

@app.post("/auth/token", response_model=TokenResponse, tags=["Authentication"])
async def get_token(request: TokenRequest):
    """
    OAuth2 token endpoint - supports multiple grant types:
    - password (username/password)
    - refresh_token
    - urn:ietf:params:oauth:grant-type:jwt-bearer (JWT assertion)
    """
    if request.grant_type == "password":
        # Simple username/password authentication
        # In mock, any username returns token for first matching user
        for user in MOCK_DATA["user_profiles"]:
            if user["username"] == request.username:
                return TokenResponse(**MOCK_DATA["auth_tokens"]["basic"])
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    elif request.grant_type == "refresh_token":
        # Refresh token flow - return new tokens
        if not request.refresh_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="refresh_token required"
            )
        return TokenResponse(**MOCK_DATA["auth_tokens"]["basic"])
    
    elif request.grant_type == "urn:ietf:params:oauth:grant-type:jwt-bearer":
        # JWT bearer assertion (OMH token exchange)
        if not request.assertion:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="assertion required for JWT bearer grant"
            )
        return TokenResponse(**MOCK_DATA["auth_tokens"]["jwt_format"])
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant_type: {request.grant_type}"
        )


@app.post("/mobile/platform/sso/exchange-token", tags=["Authentication"])
async def exchange_omh_token(
    authorization: str = Header(...),
    oracle_mobile_backend_id: Optional[str] = Header(None)
):
    """
    OMH-specific token exchange endpoint
    Exchanges external JWT for OMH access token
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    # Return OMH-formatted token response
    return JSONResponse({
        "access_token": MOCK_DATA["auth_tokens"]["jwt_format"]["access_token"],
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": MOCK_DATA["auth_tokens"]["refresh_token"]["refresh_token"],
        "scope": "openid profile email",
        "id_token": MOCK_DATA["auth_tokens"]["id_token"]["id_token"]
    })


@app.post("/auth/refresh", tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token required"
        )
    
    # Return new tokens
    return JSONResponse(MOCK_DATA["auth_tokens"]["basic"])


# ============================================================================
# User Profile Endpoints
# ============================================================================

@app.get("/api/v1/user/profile", response_model=UserProfile, tags=["User"])
async def get_user_profile(token: str = Depends(verify_token)):
    """Get authenticated user's profile"""
    user = get_user_from_token(token)
    return UserProfile(**user)


@app.get("/api/v1/users/{user_id}", response_model=UserProfile, tags=["User"])
async def get_user_by_id(user_id: str, token: str = Depends(verify_token)):
    """Get user profile by ID (requires authentication)"""
    for user in MOCK_DATA["user_profiles"]:
        if user["user_id"] == user_id:
            return UserProfile(**user)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found"
    )


@app.get("/api/v1/users", response_model=List[UserProfile], tags=["User"])
async def list_users(
    token: str = Depends(verify_token),
    status: Optional[str] = None
):
    """List all users (with optional status filter)"""
    users = MOCK_DATA["user_profiles"]
    
    if status:
        users = [u for u in users if u["account_status"] == status]
    
    return [UserProfile(**u) for u in users]


# ============================================================================
# OMH Maps / Location Endpoints
# ============================================================================

@app.get("/api/v1/maps/location", response_model=LocationResponse, tags=["Maps"])
async def get_location(
    token: str = Depends(verify_token),
    place_id: Optional[str] = None
):
    """Get location data (default or by place_id)"""
    if place_id:
        for loc in MOCK_DATA["locations"]:
            if loc["location"]["place_id"] == place_id:
                return LocationResponse(**loc)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Location not found"
        )
    
    # Return default location (San Francisco)
    return LocationResponse(**MOCK_DATA["locations"][0])


@app.get("/api/v1/maps/locations", tags=["Maps"])
async def list_locations(token: str = Depends(verify_token)):
    """List all available mock locations"""
    return {
        "locations": [loc["location"] for loc in MOCK_DATA["locations"]],
        "count": len(MOCK_DATA["locations"])
    }


@app.post("/api/v1/maps/select-location", tags=["Maps"])
async def select_location(
    location: LocationResponse,
    token: str = Depends(verify_token)
):
    """Simulate location selection (stores in session/mock)"""
    return {
        "success": True,
        "message": "Location selected",
        "selected_location": location.dict()
    }


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info"""
    return {
        "service": "OMH Mock Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "authentication": [
                "/auth/token",
                "/auth/refresh",
                "/mobile/platform/sso/exchange-token"
            ],
            "user": [
                "/api/v1/user/profile",
                "/api/v1/users",
                "/api/v1/users/{user_id}"
            ],
            "maps": [
                "/api/v1/maps/location",
                "/api/v1/maps/locations",
                "/api/v1/maps/select-location"
            ]
        },
        "docs": "/docs"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/mock/tokens", tags=["Development"])
async def get_mock_tokens():
    """Development endpoint to retrieve mock tokens for testing"""
    return {
        "info": "Use these tokens in your Authorization header as 'Bearer <token>'",
        "tokens": {
            "alice": "mock_access_token_abc123xyz456",
            "alice_jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJVc2VyMTIzIiwibmFtZSI6IkFsaWNlIFNtaXRoIiwiYWRtaW4iOnRydWUsImlhdCI6MTY5NjQ2ODAwMCwiZXhwIjoxNjk2NDcxNjAwfQ._sVTeECHBI_sdMjXSQ7Vow_index53jZnZ1049k",
            "testuser": "mock_access_token_123456789"
        },
        "example_usage": {
            "curl": "curl -H 'Authorization: Bearer mock_access_token_abc123xyz456' http://localhost:8001/api/v1/user/profile"
        }
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

