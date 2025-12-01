"""
Example: AutoRL Integration with OMH Mock Server
=================================================
Demonstrates how to integrate OMH authentication and location services
with AutoRL workflows using the mock server.
"""

import requests
from typing import Dict, Optional, Any
import json


class OMHMockClient:
    """Client for interacting with OMH Mock Server"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        
    def login(self, username: str, password: str = "test") -> Dict[str, Any]:
        """Login with username and password"""
        response = requests.post(
            f"{self.base_url}/auth/token",
            json={
                "grant_type": "password",
                "username": username,
                "password": password
            }
        )
        response.raise_for_status()
        data = response.json()
        
        self.access_token = data["access_token"]
        self.refresh_token = data.get("refresh_token")
        
        return data
    
    def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh the access token"""
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        response = requests.post(
            f"{self.base_url}/auth/refresh",
            params={"refresh_token": self.refresh_token}
        )
        response.raise_for_status()
        data = response.json()
        
        self.access_token = data["access_token"]
        return data
    
    def get_headers(self) -> Dict[str, str]:
        """Get authorization headers"""
        if not self.access_token:
            raise ValueError("Not authenticated. Call login() first.")
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get current user's profile"""
        response = requests.get(
            f"{self.base_url}/api/v1/user/profile",
            headers=self.get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_location(self, place_id: Optional[str] = None) -> Dict[str, Any]:
        """Get location data"""
        params = {"place_id": place_id} if place_id else {}
        response = requests.get(
            f"{self.base_url}/api/v1/maps/location",
            headers=self.get_headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def list_locations(self) -> Dict[str, Any]:
        """List all available locations"""
        response = requests.get(
            f"{self.base_url}/api/v1/maps/locations",
            headers=self.get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def select_location(self, latitude: float, longitude: float, 
                       address: str, place_id: str) -> Dict[str, Any]:
        """Select a location"""
        response = requests.post(
            f"{self.base_url}/api/v1/maps/select-location",
            headers=self.get_headers(),
            json={
                "location": {
                    "latitude": latitude,
                    "longitude": longitude,
                    "address": address,
                    "place_id": place_id
                }
            }
        )
        response.raise_for_status()
        return response.json()


def example_basic_auth_flow():
    """Example: Basic authentication flow"""
    print("=" * 60)
    print("Example 1: Basic Authentication Flow")
    print("=" * 60)
    
    client = OMHMockClient()
    
    # Login
    print("\n1. Logging in as alice.smith...")
    auth_data = client.login("alice.smith")
    print(f"   ✓ Access token: {auth_data['access_token'][:50]}...")
    print(f"   ✓ Expires in: {auth_data['expires_in']} seconds")
    
    # Get profile
    print("\n2. Fetching user profile...")
    profile = client.get_user_profile()
    print(f"   ✓ User: {profile['full_name']} ({profile['email']})")
    print(f"   ✓ Roles: {', '.join(profile['roles'])}")
    print(f"   ✓ Status: {profile['account_status']}")


def example_location_workflow():
    """Example: Location-based workflow"""
    print("\n" + "=" * 60)
    print("Example 2: Location-Based Workflow")
    print("=" * 60)
    
    client = OMHMockClient()
    client.login("alice.smith")
    
    # List available locations
    print("\n1. Listing available locations...")
    locations_data = client.list_locations()
    print(f"   ✓ Found {locations_data['count']} locations:")
    for loc in locations_data['locations']:
        print(f"     - {loc['address']} ({loc['place_id']})")
    
    # Get default location
    print("\n2. Getting default location...")
    location = client.get_location()
    loc = location['location']
    print(f"   ✓ Location: {loc['address']}")
    print(f"   ✓ Coordinates: ({loc['latitude']}, {loc['longitude']})")
    
    # Select a location
    print("\n3. Selecting a location...")
    result = client.select_location(
        latitude=loc['latitude'],
        longitude=loc['longitude'],
        address=loc['address'],
        place_id=loc['place_id']
    )
    print(f"   ✓ {result['message']}")


def example_autorl_task_with_location():
    """Example: AutoRL task with location context"""
    print("\n" + "=" * 60)
    print("Example 3: AutoRL Task with Location Context")
    print("=" * 60)
    
    client = OMHMockClient()
    
    # Authenticate
    print("\n1. Authenticating user...")
    client.login("bob.jones")
    profile = client.get_user_profile()
    print(f"   ✓ Authenticated as: {profile['full_name']}")
    
    # Get location context
    print("\n2. Getting location context...")
    location = client.get_location()
    loc = location['location']
    
    # Simulate AutoRL task with location context
    print("\n3. Creating AutoRL task with context...")
    task_context = {
        "user": {
            "id": profile['user_id'],
            "name": profile['full_name'],
            "email": profile['email'],
            "roles": profile['roles']
        },
        "location": {
            "address": loc['address'],
            "coordinates": {
                "lat": loc['latitude'],
                "lng": loc['longitude']
            },
            "place_id": loc['place_id']
        },
        "task": {
            "type": "location_based_search",
            "description": f"Find restaurants near {loc['address']}"
        }
    }
    
    print("   ✓ Task context created:")
    print(json.dumps(task_context, indent=4))
    
    return task_context


def example_multi_user_scenario():
    """Example: Multi-user scenario with different roles"""
    print("\n" + "=" * 60)
    print("Example 4: Multi-User Scenario")
    print("=" * 60)
    
    users = ["alice.smith", "bob.jones", "carol.wu"]
    
    for username in users:
        print(f"\n--- User: {username} ---")
        client = OMHMockClient()
        
        try:
            # Login
            client.login(username)
            profile = client.get_user_profile()
            
            print(f"  Name: {profile['full_name']}")
            print(f"  Email: {profile['email']}")
            print(f"  Roles: {', '.join(profile['roles'])}")
            print(f"  Status: {profile['account_status']}")
            
            # Try to get location
            if profile['account_status'] == 'active':
                location = client.get_location()
                print(f"  Location: {location['location']['address']}")
            else:
                print("  ⚠ User inactive, skipping location fetch")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")


def example_token_refresh():
    """Example: Token refresh workflow"""
    print("\n" + "=" * 60)
    print("Example 5: Token Refresh Workflow")
    print("=" * 60)
    
    client = OMHMockClient()
    
    # Initial login
    print("\n1. Initial login...")
    auth_data = client.login("alice.smith")
    print(f"   ✓ Access token obtained")
    print(f"   ✓ Refresh token: {auth_data['refresh_token'][:50]}...")
    
    # Use the token
    print("\n2. Using access token to fetch profile...")
    profile = client.get_user_profile()
    print(f"   ✓ Profile retrieved: {profile['username']}")
    
    # Simulate token expiry and refresh
    print("\n3. Simulating token expiry and refresh...")
    new_auth = client.refresh_access_token()
    print(f"   ✓ New access token obtained")
    print(f"   ✓ New token: {new_auth['access_token'][:50]}...")
    
    # Use refreshed token
    print("\n4. Using refreshed token...")
    profile = client.get_user_profile()
    print(f"   ✓ Profile retrieved with new token: {profile['username']}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "OMH Mock Server Integration Examples" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        # Check if server is running
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code != 200:
            raise Exception("Server not healthy")
    except Exception as e:
        print("\n❌ Error: OMH Mock Server is not running!")
        print("   Start it with: python src/agent_service/omh_mock_server.py")
        print("   Or use: cd src/agent_service && ./start_omh_mock.sh")
        return
    
    try:
        # Run examples
        example_basic_auth_flow()
        example_location_workflow()
        example_autorl_task_with_location()
        example_multi_user_scenario()
        example_token_refresh()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - Explore the API docs at http://localhost:8001/docs")
        print("  - Modify mock data in src/agent_service/mock_data/omh_mock_data.json")
        print("  - Integrate with your AutoRL workflows")
        print()
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

