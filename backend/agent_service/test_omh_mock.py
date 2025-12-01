"""
Test script for OMH Mock Server
Validates all endpoints and mock data functionality
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8001"


def test_health_check():
    """Test health endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✅ Health check passed")
    return data


def test_get_mock_tokens():
    """Test mock tokens endpoint"""
    print("\nTesting mock tokens endpoint...")
    response = requests.get(f"{BASE_URL}/mock/tokens")
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    print("✅ Mock tokens retrieved")
    print(f"   Available tokens: {list(data['tokens'].keys())}")
    return data["tokens"]


def test_password_authentication():
    """Test password grant authentication"""
    print("\nTesting password authentication...")
    response = requests.post(
        f"{BASE_URL}/auth/token",
        json={
            "grant_type": "password",
            "username": "alice.smith",
            "password": "test123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "Bearer"
    assert data["expires_in"] == 3600
    print("✅ Password authentication successful")
    print(f"   Token: {data['access_token'][:50]}...")
    return data["access_token"]


def test_refresh_token():
    """Test refresh token flow"""
    print("\nTesting refresh token...")
    response = requests.post(
        f"{BASE_URL}/auth/refresh",
        params={"refresh_token": "mock_refresh_token_xyz789abc123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    print("✅ Refresh token successful")
    return data["access_token"]


def test_omh_token_exchange():
    """Test OMH SSO token exchange"""
    print("\nTesting OMH token exchange...")
    response = requests.post(
        f"{BASE_URL}/mobile/platform/sso/exchange-token",
        headers={
            "Authorization": "Bearer external-jwt-token",
            "oracle-mobile-backend-id": "TEST_BACKEND_ID"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "id_token" in data
    print("✅ OMH token exchange successful")
    return data


def test_user_profile(token: str):
    """Test getting user profile"""
    print("\nTesting user profile endpoint...")
    response = requests.get(
        f"{BASE_URL}/api/v1/user/profile",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "username" in data
    assert "email" in data
    print("✅ User profile retrieved")
    print(f"   User: {data['username']} ({data['email']})")
    return data


def test_list_users(token: str):
    """Test listing all users"""
    print("\nTesting list users endpoint...")
    response = requests.get(
        f"{BASE_URL}/api/v1/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    print(f"✅ Users listed: {len(data)} users found")
    return data


def test_get_user_by_id(token: str, user_id: str):
    """Test getting specific user by ID"""
    print(f"\nTesting get user by ID ({user_id})...")
    response = requests.get(
        f"{BASE_URL}/api/v1/users/{user_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user_id
    print(f"✅ User retrieved: {data['username']}")
    return data


def test_get_location(token: str):
    """Test getting location data"""
    print("\nTesting get location endpoint...")
    response = requests.get(
        f"{BASE_URL}/api/v1/maps/location",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "location" in data
    location = data["location"]
    assert "latitude" in location
    assert "longitude" in location
    assert "address" in location
    print(f"✅ Location retrieved: {location['address']}")
    print(f"   Coordinates: ({location['latitude']}, {location['longitude']})")
    return data


def test_list_locations(token: str):
    """Test listing all locations"""
    print("\nTesting list locations endpoint...")
    response = requests.get(
        f"{BASE_URL}/api/v1/maps/locations",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert data["count"] > 0
    print(f"✅ Locations listed: {data['count']} locations found")
    for loc in data["locations"]:
        print(f"   - {loc['address']}")
    return data


def test_select_location(token: str):
    """Test location selection"""
    print("\nTesting select location endpoint...")
    response = requests.post(
        f"{BASE_URL}/api/v1/maps/select-location",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "address": "San Francisco, CA, USA",
                "place_id": "mock_place_12345"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print("✅ Location selection successful")
    return data


def test_unauthorized_access():
    """Test that endpoints require authentication"""
    print("\nTesting unauthorized access protection...")
    response = requests.get(f"{BASE_URL}/api/v1/user/profile")
    assert response.status_code == 401
    print("✅ Unauthorized access properly blocked")


def test_invalid_token():
    """Test invalid token handling"""
    print("\nTesting invalid token handling...")
    response = requests.get(
        f"{BASE_URL}/api/v1/user/profile",
        headers={"Authorization": "Bearer invalid_token_xyz"}
    )
    assert response.status_code == 401
    print("✅ Invalid token properly rejected")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("OMH Mock Server Test Suite")
    print("=" * 60)
    
    try:
        # System tests
        test_health_check()
        mock_tokens = test_get_mock_tokens()
        
        # Authentication tests
        password_token = test_password_authentication()
        refresh_token = test_refresh_token()
        test_omh_token_exchange()
        
        # Use a known token for subsequent tests
        test_token = mock_tokens.get("alice", password_token)
        
        # User profile tests
        user_profile = test_user_profile(test_token)
        users = test_list_users(test_token)
        test_get_user_by_id(test_token, user_profile["user_id"])
        
        # Location tests
        test_get_location(test_token)
        test_list_locations(test_token)
        test_select_location(test_token)
        
        # Security tests
        test_unauthorized_access()
        test_invalid_token()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR: Is the server running on http://localhost:8001?")
        print("   Start it with: python omh_mock_server.py")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

