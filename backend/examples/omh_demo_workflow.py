"""
OMH Integration Demo Workflow

Demonstrates how to use Open Mobile Hub (OMH) authentication and maps
features within AutoRL automation workflows.

This example shows:
1. Authenticating users with OMH OAuth2
2. Using OMH Maps for location-based task triggers
3. Geocoding addresses for mobile automation
4. Creating geofenced automation workflows
"""

import asyncio
import httpx
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"


async def demo_omh_authentication():
    """Demo: OMH Authentication Flow"""
    print("\n" + "="*60)
    print("OMH AUTHENTICATION DEMO")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # 1. Get OMH auth URL
        print("1. Getting OMH authorization URL...")
        response = await client.get(
            f"{API_BASE_URL}/api/v1/omh/auth/url",
            params={"redirect_uri": "http://localhost:3000/callback"}
        )
        auth_data = response.json()
        print(f"   Auth URL: {auth_data['auth_url']}")
        print(f"   Mock Mode: {auth_data.get('mock_mode', False)}")
        
        # 2. Simulate token (in real app, get from OAuth callback)
        mock_token = "mock_token_demo_12345"
        
        # 3. Validate token
        print("\n2. Validating access token...")
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/auth/validate",
            json={"token": mock_token}
        )
        validation = response.json()
        
        if validation.get("valid"):
            print(f"   ✓ Token valid")
            print(f"   User: {validation.get('username')}")
            print(f"   Email: {validation.get('email')}")
            print(f"   Roles: {validation.get('roles')}")
        else:
            print(f"   ✗ Token invalid")
        
        # 4. Get user info
        print("\n3. Fetching user profile...")
        response = await client.get(
            f"{API_BASE_URL}/api/v1/omh/auth/user",
            headers={"Authorization": f"Bearer {mock_token}"}
        )
        user_info = response.json()
        print(f"   User ID: {user_info.get('user_id')}")
        print(f"   Name: {user_info.get('username')}")
        print(f"   Created: {user_info.get('created_at', 'N/A')}")


async def demo_omh_maps_geocoding():
    """Demo: OMH Maps Geocoding"""
    print("\n" + "="*60)
    print("OMH MAPS GEOCODING DEMO")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # 1. Geocode an address
        addresses = [
            "San Francisco, CA",
            "New York, NY",
            "London, UK"
        ]
        
        for address in addresses:
            print(f"\nGeocoding: {address}")
            response = await client.post(
                f"{API_BASE_URL}/api/v1/omh/maps/geocode",
                json={"address": address}
            )
            result = response.json()
            
            if "error" not in result:
                print(f"   Coordinates: {result['latitude']:.4f}, {result['longitude']:.4f}")
                print(f"   Formatted: {result['formatted_address']}")
            else:
                print(f"   Error: {result['error']}")


async def demo_omh_maps_nearby_places():
    """Demo: OMH Maps Nearby Places"""
    print("\n" + "="*60)
    print("OMH MAPS NEARBY PLACES DEMO")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # Find places near San Francisco coordinates
        lat, lng = 37.7749, -122.4194
        
        print(f"Finding places near ({lat}, {lng})...\n")
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/maps/nearby",
            json={
                "latitude": lat,
                "longitude": lng,
                "radius": 1000,
                "place_type": None
            }
        )
        result = response.json()
        
        if result.get("places"):
            print(f"Found {result['count']} places:\n")
            for place in result["places"]:
                print(f"   • {place['name']}")
                print(f"     Type: {place['type']}")
                print(f"     Distance: {place['distance_meters']}m")
                print(f"     Rating: ⭐ {place['rating']}")
                print()
        else:
            print("   No places found")


async def demo_location_based_automation():
    """Demo: Location-Based Task Automation"""
    print("\n" + "="*60)
    print("LOCATION-BASED AUTOMATION DEMO")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # Scenario: Trigger mobile automation when user enters office
        
        print("Creating geofenced automation workflow...\n")
        
        # 1. Define office location
        office_coords = {"lat": 37.7879, "lng": -122.4074}  # Example: SF Tech Office
        
        # 2. Create geofence trigger
        print("1. Setting up geofence trigger:")
        print(f"   Location: Office ({office_coords['lat']}, {office_coords['lng']})")
        print(f"   Radius: 100 meters")
        print(f"   Trigger: on_enter")
        
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/maps/location-trigger",
            json={
                "task_id": "office_arrival_automation",
                "geofence": {
                    "center": office_coords,
                    "radius_meters": 100
                },
                "trigger_type": "enter"
            }
        )
        trigger = response.json()
        print(f"   ✓ Trigger created: {trigger['trigger_id']}")
        
        # 3. Simulate user approaching office
        print("\n2. Simulating user location...")
        user_locations = [
            {"lat": 37.7900, "lng": -122.4100, "desc": "Far from office"},
            {"lat": 37.7885, "lng": -122.4080, "desc": "Approaching office"},
            {"lat": 37.7879, "lng": -122.4074, "desc": "At office entrance"}
        ]
        
        for loc in user_locations:
            print(f"\n   User location: {loc['desc']}")
            print(f"   Coordinates: ({loc['lat']}, {loc['lng']})")
            
            # Check if inside geofence
            response = await client.post(
                f"{API_BASE_URL}/api/v1/omh/maps/geofence",
                json={
                    "location": {"lat": loc["lat"], "lng": loc["lng"]},
                    "geofence": {
                        "center": office_coords,
                        "radius_meters": 100
                    }
                }
            )
            check = response.json()
            
            if check["inside_geofence"]:
                print(f"   🔔 TRIGGER ACTIVATED!")
                print(f"   → Executing automation task: office_arrival_automation")
                print(f"   → Tasks: Disable phone ringer, open work apps, send arrival notification")
                break
            else:
                distance = check["distance_from_center"]
                print(f"   Distance from office: {distance:.0f}m (not yet triggered)")
            
            await asyncio.sleep(1)


async def demo_distance_calculation():
    """Demo: Calculate distance between locations"""
    print("\n" + "="*60)
    print("DISTANCE CALCULATION DEMO")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # Calculate distance SF to NY
        sf = {"lat": 37.7749, "lng": -122.4194}
        ny = {"lat": 40.7128, "lng": -74.0060}
        
        print("Calculating distance: San Francisco to New York\n")
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/maps/distance",
            json={
                "origin": sf,
                "destination": ny
            }
        )
        result = response.json()
        
        print(f"   Distance: {result['distance_miles']:.2f} miles")
        print(f"   Distance: {result['distance_km']:.2f} kilometers")
        print(f"   Distance: {result['distance_meters']:.0f} meters")


async def demo_complete_workflow():
    """Demo: Complete workflow combining auth + maps"""
    print("\n" + "="*60)
    print("COMPLETE OMH WORKFLOW DEMO")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        # 1. Authenticate
        print("Step 1: Authenticate user with OMH")
        token = "mock_token_complete_demo"
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/auth/validate",
            json={"token": token}
        )
        auth = response.json()
        
        if not auth.get("valid"):
            print("   ✗ Authentication failed")
            return
        
        print(f"   ✓ Authenticated as {auth['username']}")
        
        # 2. Get user's current location (simulated)
        print("\nStep 2: Get user's current location")
        user_location = {"lat": 37.7749, "lng": -122.4194}
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/maps/reverse-geocode",
            json={
                "latitude": user_location["lat"],
                "longitude": user_location["lng"]
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        location_info = response.json()
        print(f"   Location: {location_info['formatted_address']}")
        print(f"   City: {location_info['city']}")
        
        # 3. Find nearby automation-relevant places
        print("\nStep 3: Find nearby places for context-aware automation")
        response = await client.post(
            f"{API_BASE_URL}/api/v1/omh/maps/nearby",
            json={
                "latitude": user_location["lat"],
                "longitude": user_location["lng"],
                "radius": 500,
                "place_type": "bank"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        nearby = response.json()
        
        if nearby.get("places"):
            nearest = nearby["places"][0]
            print(f"   Nearest bank: {nearest['name']} ({nearest['distance_meters']}m away)")
            print(f"   → Suggested automation: Open banking app when nearby")
        
        # 4. Create smart automation based on location
        print("\nStep 4: Create location-aware mobile automation")
        print("   ✓ Automation created: Auto-open banking app when near branch")
        print("   ✓ Geofence: 200m radius around bank")
        print("   ✓ User: " + auth['username'])
        
        print("\n✅ Complete workflow executed successfully!")


async def check_omh_status():
    """Check OMH integration status"""
    print("\n" + "="*60)
    print("OMH INTEGRATION STATUS")
    print("="*60 + "\n")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/api/v1/omh/status")
            status = response.json()
            
            print(f"Overall Status: {status['message']}\n")
            print(f"Authentication:")
            print(f"   Enabled: {status['auth']['enabled']}")
            print(f"   Mock Mode: {status['auth']['mock_mode']}")
            print(f"\nMaps:")
            print(f"   Enabled: {status['maps']['enabled']}")
            print(f"   Mock Mode: {status['maps']['mock_mode']}")
            
            return status
        except Exception as e:
            print(f"❌ Failed to connect to API: {e}")
            print(f"   Make sure the AutoRL backend is running at {API_BASE_URL}")
            return None


async def main():
    """Run all OMH integration demos"""
    print("\n" + "="*70)
    print(" "*15 + "AUTORL OMH INTEGRATION DEMO")
    print("="*70)
    print("\nDemonstrating Open Mobile Hub integration capabilities\n")
    
    # Check if backend is running
    status = await check_omh_status()
    if not status:
        print("\n⚠️  Please start the AutoRL backend first:")
        print("   uvicorn src.main:app --reload")
        return
    
    try:
        # Run all demos
        await demo_omh_authentication()
        await demo_omh_maps_geocoding()
        await demo_omh_maps_nearby_places()
        await demo_distance_calculation()
        await demo_location_based_automation()
        await demo_complete_workflow()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

