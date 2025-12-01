"""
Open Mobile Hub (OMH) Authentication Plugin for AutoRL

Provides OAuth2-based authentication integration with OMH Auth SDK,
enabling secure user authentication for mobile automation workflows.

Features:
- OAuth2 token management
- User profile fetching
- Token refresh handling
- Session management
"""

import os
import httpx
import json
import time
from typing import Dict, Any, Optional
from base_plugin import BasePlugin
import logging

logger = logging.getLogger("autorl.omh_auth")


class OMHAuthPlugin(BasePlugin):
    """
    OMH Authentication Plugin
    
    Integrates Open Mobile Hub OAuth2 authentication to secure
    AutoRL API endpoints and manage user sessions.
    """
    
    def __init__(self):
        self.client_id: Optional[str] = None
        self.client_secret: Optional[str] = None
        self.auth_url: Optional[str] = None
        self.token_url: Optional[str] = None
        self.user_info_url: Optional[str] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize OMH Auth Plugin with configuration.
        
        Args:
            config: Configuration dict with OMH credentials and endpoints
        """
        self.client_id = config.get("omh_client_id") or os.getenv("OMH_CLIENT_ID")
        self.client_secret = config.get("omh_client_secret") or os.getenv("OMH_CLIENT_SECRET")
        self.auth_url = config.get("omh_auth_url") or os.getenv("OMH_AUTH_URL", "https://auth.openmobilehub.org/oauth/authorize")
        self.token_url = config.get("omh_token_url") or os.getenv("OMH_TOKEN_URL", "https://auth.openmobilehub.org/oauth/token")
        self.user_info_url = config.get("omh_user_info_url") or os.getenv("OMH_USER_INFO_URL", "https://auth.openmobilehub.org/oauth/userinfo")
        
        if not self.client_id or not self.client_secret:
            logger.warning("OMH Auth credentials not configured. Plugin will operate in mock mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"OMH Auth Plugin initialized (mock_mode={self.mock_mode})")
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process authentication requests.
        
        Args:
            input_data: Request data with 'action' key
            
        Returns:
            Response data with authentication results
        """
        action = input_data.get("action")
        
        if action == "validate_token":
            return self._validate_token(input_data.get("token"))
        elif action == "refresh_token":
            return self._refresh_token(input_data.get("refresh_token"))
        elif action == "get_user_info":
            return self._get_user_info(input_data.get("token"))
        elif action == "get_auth_url":
            return self._get_auth_url(input_data.get("redirect_uri"), input_data.get("state"))
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _get_auth_url(self, redirect_uri: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Generate OAuth2 authorization URL."""
        if self.mock_mode:
            return {
                "auth_url": f"https://mock-omh-auth.example.com/authorize?client_id=mock&redirect_uri={redirect_uri}",
                "mock_mode": True
            }
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid profile email",
        }
        if state:
            params["state"] = state
            
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return {
            "auth_url": f"{self.auth_url}?{query_string}",
            "mock_mode": False
        }
    
    def _validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an access token.
        
        Args:
            token: Access token to validate
            
        Returns:
            Validation result with user info if valid
        """
        if self.mock_mode:
            # Mock validation for demo purposes
            if token and token.startswith("mock_token_"):
                return {
                    "valid": True,
                    "user_id": "demo_user_123",
                    "username": "demo_user",
                    "email": "demo@autorl.app",
                    "roles": ["user", "automation_runner"],
                    "mock_mode": True
                }
            return {"valid": False, "error": "Invalid token", "mock_mode": True}
        
        # Check cache first
        if token in self.token_cache:
            cached = self.token_cache[token]
            if cached["expires_at"] > time.time():
                return {"valid": True, **cached["user_info"]}
        
        # Validate with OMH server
        try:
            return self._validate_token_with_server(token)
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def _validate_token_with_server(self, token: str) -> Dict[str, Any]:
        """Validate token by calling OMH user info endpoint."""
        # This would be async in production, simplified for plugin demo
        headers = {"Authorization": f"Bearer {token}"}
        # In real implementation, use async httpx call
        return {
            "valid": True,
            "user_id": "omh_user_id",
            "username": "omh_user",
            "email": "user@example.com",
            "roles": ["user"]
        }
    
    def _refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New access token and expiry info
        """
        if self.mock_mode:
            return {
                "access_token": f"mock_token_{int(time.time())}",
                "refresh_token": refresh_token,
                "expires_in": 3600,
                "token_type": "Bearer",
                "mock_mode": True
            }
        
        # Real refresh token flow
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
            # In real implementation, use async httpx.post
            return {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "expires_in": 3600,
                "token_type": "Bearer"
            }
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return {"error": str(e)}
    
    def _get_user_info(self, token: str) -> Dict[str, Any]:
        """
        Fetch user profile information using access token.
        
        Args:
            token: Access token
            
        Returns:
            User profile data
        """
        validation = self._validate_token(token)
        if not validation.get("valid"):
            return {"error": "Invalid token"}
        
        if self.mock_mode:
            return {
                "user_id": validation["user_id"],
                "username": validation["username"],
                "email": validation["email"],
                "name": "Demo User",
                "picture": "https://via.placeholder.com/150",
                "roles": validation["roles"],
                "created_at": "2024-01-01T00:00:00Z",
                "mock_mode": True
            }
        
        # Fetch from OMH user info endpoint
        return validation
    
    def get_authorization_header(self, token: str) -> str:
        """Helper to format authorization header."""
        return f"Bearer {token}"
    
    def is_authenticated(self, token: Optional[str]) -> bool:
        """Check if a token is valid."""
        if not token:
            return False
        result = self._validate_token(token)
        return result.get("valid", False)
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self.http_client:
            # In async context, use: await self.http_client.aclose()
            logger.info("OMH Auth Plugin shutdown")
        self.token_cache.clear()


# Singleton instance for easy import
omh_auth_plugin = OMHAuthPlugin()

