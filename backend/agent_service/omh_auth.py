"""
OMH Authentication Integration for AutoRL
==========================================
Provides OAuth2 authentication using OMH Mock Server or real OMH provider.
"""

import requests
from fastapi import HTTPException, Header, Depends
from typing import Optional, Tuple, Dict, Any
import os
from datetime import datetime, timedelta


class OMHAuthClient:
    """Client for authenticating with OMH OAuth provider"""
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("OMH_BASE_URL", "http://localhost:8001")
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify token with OMH server and return user profile"""
        
        # Check cache first (simple caching, no expiry check)
        if token in self._token_cache:
            return self._token_cache[token]
        
        try:
            # Get user profile from OMH server
            response = requests.get(
                f"{self.base_url}/api/v1/user/profile",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            
            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired token"
                )
            
            response.raise_for_status()
            user_profile = response.json()
            
            # Cache the user profile
            self._token_cache[token] = user_profile
            
            return user_profile
            
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="OMH authentication service unavailable"
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail="OMH authentication service timeout"
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Authentication error: {str(e)}"
            )
    
    def get_user_location(self, token: str, place_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user location from OMH Maps service"""
        try:
            params = {"place_id": place_id} if place_id else {}
            response = requests.get(
                f"{self.base_url}/api/v1/maps/location",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Return None instead of raising to make location optional
            return None
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and get token"""
        try:
            response = requests.post(
                f"{self.base_url}/auth/token",
                json={
                    "grant_type": "password",
                    "username": username,
                    "password": password
                },
                timeout=5
            )
            
            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid credentials"
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Authentication error: {str(e)}"
            )


# Global OMH client instance
omh_client = OMHAuthClient()


# Dependency for FastAPI routes
async def get_omh_user(
    authorization: Optional[str] = Header(None),
    x_username: Optional[str] = Header(None)
) -> Tuple[str, Dict[str, Any]]:
    """
    FastAPI dependency to get authenticated OMH user.
    Supports both:
    - OMH OAuth token (Authorization: Bearer <token>)
    - Legacy x-username header (for backward compatibility)
    """
    
    # Try OMH OAuth first
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        user_profile = omh_client.verify_token(token)
        
        # Return username and enhanced user record
        username = user_profile["username"]
        user_record = {
            "username": username,
            "user_id": user_profile["user_id"],
            "email": user_profile["email"],
            "full_name": user_profile["full_name"],
            "roles": user_profile["roles"],
            "account_status": user_profile["account_status"],
            "auth_method": "omh_oauth",
            "subscription": "premium" if "admin" in user_profile["roles"] else "basic",
            "quota": 100 if "admin" in user_profile["roles"] else 10,
            "tasks_used": 0  # This would come from your database
        }
        
        return (username, user_record)
    
    # Fallback to legacy x-username header
    elif x_username:
        # Legacy authentication - kept for backward compatibility
        from .store import users_db
        if x_username not in users_db:
            raise HTTPException(status_code=401, detail="Invalid user")
        user_record = users_db[x_username]
        user_record["auth_method"] = "legacy"
        return (x_username, user_record)
    
    else:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Provide Authorization header or x-username."
        )


async def get_omh_user_with_location(
    authorization: Optional[str] = Header(None),
    x_username: Optional[str] = Header(None)
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    FastAPI dependency to get authenticated OMH user with location context.
    Returns: (username, user_record, location_data)
    """
    username, user_record = await get_omh_user(authorization, x_username)
    
    # Try to get location if OMH auth was used
    location = None
    if user_record.get("auth_method") == "omh_oauth" and authorization:
        token = authorization.replace("Bearer ", "")
        location = omh_client.get_user_location(token)
    
    return (username, user_record, location)


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.
    Usage: @app.get("/admin", dependencies=[Depends(require_role("admin"))])
    """
    async def _check_role(
        authorization: Optional[str] = Header(None),
        x_username: Optional[str] = Header(None)
    ):
        username, user_record = await get_omh_user(authorization, x_username)
        
        if required_role not in user_record.get("roles", []):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return (username, user_record)
    
    return _check_role


# Helper functions for non-FastAPI usage
def verify_token_sync(token: str) -> Dict[str, Any]:
    """Synchronous version of token verification"""
    return omh_client.verify_token(token)


def get_location_context(token: str, place_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get location context for a task"""
    return omh_client.get_user_location(token, place_id)

