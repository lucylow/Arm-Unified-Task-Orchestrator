"""
AutoRL Workflows with OMH Integration
======================================
Pre-built workflows that leverage OMH authentication and location services
for mobile automation tasks.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import requests


@dataclass
class WorkflowContext:
    """Context for workflow execution"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    location: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = None
    access_token: Optional[str] = None


class OMHWorkflowEngine:
    """Engine for executing location-aware mobile automation workflows"""
    
    def __init__(self, autorl_api_base: str = "http://localhost:8000", 
                 omh_base: str = "http://localhost:8001"):
        self.autorl_api_base = autorl_api_base
        self.omh_base = omh_base
    
    def create_context_from_token(self, access_token: str) -> WorkflowContext:
        """Create workflow context from OMH access token"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Get user profile
        response = requests.get(f"{self.omh_base}/api/v1/user/profile", headers=headers)
        response.raise_for_status()
        profile = response.json()
        
        # Get location
        location = None
        try:
            loc_response = requests.get(f"{self.omh_base}/api/v1/maps/location", headers=headers)
            if loc_response.status_code == 200:
                location = loc_response.json().get("location")
        except:
            pass
        
        return WorkflowContext(
            user_id=profile["user_id"],
            username=profile["username"],
            email=profile["email"],
            roles=profile["roles"],
            location=location,
            access_token=access_token
        )
    
    def execute_location_based_search(self, context: WorkflowContext, 
                                     search_query: str, 
                                     device_id: str) -> Dict[str, Any]:
        """
        Execute a location-based search task
        Example: "Find restaurants near my location"
        """
        if not context.location:
            return {
                "success": False,
                "error": "Location not available"
            }
        
        address = context.location.get("address", "your location")
        instruction = f"{search_query} near {address}"
        
        # Execute task via AutoRL API
        headers = {"Authorization": f"Bearer {context.access_token}"}
        payload = {
            "instruction": instruction,
            "device_id": device_id,
            "parameters": {
                "location_context": context.location,
                "search_type": "location_based",
                "user_id": context.user_id
            }
        }
        
        response = requests.post(
            f"{self.autorl_api_base}/api/v1/execute/location-aware",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return {"success": True, "result": response.json()}
        else:
            return {"success": False, "error": response.text}
    
    def execute_navigation_workflow(self, context: WorkflowContext,
                                   destination: str,
                                   device_id: str) -> Dict[str, Any]:
        """
        Execute navigation workflow with current location context
        Example: Navigate to a destination from current location
        """
        if not context.location:
            return {"success": False, "error": "Location not available"}
        
        current_address = context.location.get("address", "current location")
        instruction = f"Open maps and navigate from {current_address} to {destination}"
        
        headers = {"Authorization": f"Bearer {context.access_token}"}
        payload = {
            "instruction": instruction,
            "device_id": device_id,
            "parameters": {
                "origin": context.location,
                "destination": destination,
                "navigation_type": "driving"
            }
        }
        
        response = requests.post(
            f"{self.autorl_api_base}/api/v1/execute/location-aware",
            headers=headers,
            json=payload
        )
        
        return {"success": response.status_code == 200, "result": response.json() if response.ok else response.text}
    
    def execute_nearby_services_workflow(self, context: WorkflowContext,
                                        service_type: str,
                                        device_id: str) -> Dict[str, Any]:
        """
        Find and interact with nearby services
        Example: "Book appointment at nearby clinic"
        """
        if not context.location:
            return {"success": False, "error": "Location not available"}
        
        coordinates = f"({context.location['latitude']}, {context.location['longitude']})"
        instruction = f"Find {service_type} near {coordinates} and show top 5 results"
        
        headers = {"Authorization": f"Bearer {context.access_token}"}
        payload = {
            "instruction": instruction,
            "device_id": device_id,
            "parameters": {
                "service_type": service_type,
                "location": context.location,
                "max_results": 5,
                "sort_by": "distance"
            }
        }
        
        response = requests.post(
            f"{self.autorl_api_base}/api/v1/execute/location-aware",
            headers=headers,
            json=payload
        )
        
        return {"success": response.status_code == 200, "result": response.json() if response.ok else response.text}
    
    def execute_context_aware_reminder(self, context: WorkflowContext,
                                      reminder_text: str,
                                      device_id: str) -> Dict[str, Any]:
        """
        Create location-aware reminder
        Example: "Remind me to buy groceries when I'm near the supermarket"
        """
        if not context.location:
            return {"success": False, "error": "Location not available"}
        
        instruction = f"Set location-based reminder: '{reminder_text}' for area near {context.location['address']}"
        
        headers = {"Authorization": f"Bearer {context.access_token}"}
        payload = {
            "instruction": instruction,
            "device_id": device_id,
            "parameters": {
                "reminder_text": reminder_text,
                "trigger_location": context.location,
                "trigger_radius_meters": 500,
                "user_email": context.email
            }
        }
        
        response = requests.post(
            f"{self.autorl_api_base}/api/v1/execute/location-aware",
            headers=headers,
            json=payload
        )
        
        return {"success": response.status_code == 200, "result": response.json() if response.ok else response.text}
    
    def execute_multi_location_workflow(self, context: WorkflowContext,
                                       locations: List[str],
                                       task_per_location: str,
                                       device_id: str) -> Dict[str, Any]:
        """
        Execute tasks across multiple locations
        Example: "Check store hours at 3 different locations"
        """
        results = []
        
        for location in locations:
            instruction = f"{task_per_location} at {location}"
            
            headers = {"Authorization": f"Bearer {context.access_token}"}
            payload = {
                "instruction": instruction,
                "device_id": device_id,
                "parameters": {
                    "base_location": context.location,
                    "target_location": location,
                    "task_type": "multi_location"
                }
            }
            
            response = requests.post(
                f"{self.autorl_api_base}/api/v1/execute/location-aware",
                headers=headers,
                json=payload
            )
            
            results.append({
                "location": location,
                "success": response.status_code == 200,
                "result": response.json() if response.ok else response.text
            })
        
        return {
            "success": all(r["success"] for r in results),
            "results": results,
            "total_locations": len(locations)
        }


# Pre-built workflow templates
class WorkflowTemplates:
    """Common workflow templates for AutoRL + OMH"""
    
    @staticmethod
    def restaurant_finder(engine: OMHWorkflowEngine, context: WorkflowContext, 
                         cuisine_type: str, device_id: str) -> Dict[str, Any]:
        """Find restaurants of specific cuisine near user"""
        return engine.execute_location_based_search(
            context,
            f"Find {cuisine_type} restaurants",
            device_id
        )
    
    @staticmethod
    def gas_station_locator(engine: OMHWorkflowEngine, context: WorkflowContext,
                           device_id: str) -> Dict[str, Any]:
        """Find nearest gas stations with prices"""
        return engine.execute_nearby_services_workflow(
            context,
            "gas stations with current prices",
            device_id
        )
    
    @staticmethod
    def pharmacy_finder(engine: OMHWorkflowEngine, context: WorkflowContext,
                       device_id: str, open_now: bool = True) -> Dict[str, Any]:
        """Find nearby pharmacies"""
        query = "open pharmacies" if open_now else "pharmacies"
        return engine.execute_nearby_services_workflow(
            context,
            query,
            device_id
        )
    
    @staticmethod
    def commute_optimizer(engine: OMHWorkflowEngine, context: WorkflowContext,
                         destination: str, device_id: str) -> Dict[str, Any]:
        """Get optimal commute route with traffic"""
        if not context.location:
            return {"success": False, "error": "Location required"}
        
        instruction = f"Find fastest route from {context.location['address']} to {destination} considering current traffic"
        
        headers = {"Authorization": f"Bearer {context.access_token}"}
        payload = {
            "instruction": instruction,
            "device_id": device_id,
            "parameters": {
                "origin": context.location,
                "destination": destination,
                "optimize_for": "time",
                "include_traffic": True,
                "alternatives": 3
            }
        }
        
        response = requests.post(
            f"{engine.autorl_api_base}/api/v1/execute/location-aware",
            headers=headers,
            json=payload
        )
        
        return {"success": response.status_code == 200, "result": response.json() if response.ok else response.text}
    
    @staticmethod
    def local_events_finder(engine: OMHWorkflowEngine, context: WorkflowContext,
                           event_type: str, device_id: str) -> Dict[str, Any]:
        """Find local events near user"""
        return engine.execute_location_based_search(
            context,
            f"Find {event_type} events happening nearby",
            device_id
        )
    
    @staticmethod
    def price_comparison_workflow(engine: OMHWorkflowEngine, context: WorkflowContext,
                                  product_name: str, device_id: str) -> Dict[str, Any]:
        """Compare prices for a product at nearby stores"""
        if not context.location:
            return {"success": False, "error": "Location required"}
        
        instruction = f"Compare prices for '{product_name}' at stores near {context.location['address']}"
        
        headers = {"Authorization": f"Bearer {context.access_token}"}
        payload = {
            "instruction": instruction,
            "device_id": device_id,
            "parameters": {
                "product": product_name,
                "location": context.location,
                "max_distance_km": 5,
                "sort_by": "price"
            }
        }
        
        response = requests.post(
            f"{engine.autorl_api_base}/api/v1/execute/location-aware",
            headers=headers,
            json=payload
        )
        
        return {"success": response.status_code == 200, "result": response.json() if response.ok else response.text}


# Utility functions
def get_authenticated_engine(username: str, password: str,
                            autorl_api: str = "http://localhost:8000",
                            omh_api: str = "http://localhost:8001") -> tuple:
    """Get authenticated workflow engine and context"""
    
    # Authenticate with OMH
    auth_response = requests.post(
        f"{omh_api}/auth/token",
        json={
            "grant_type": "password",
            "username": username,
            "password": password
        }
    )
    auth_response.raise_for_status()
    auth_data = auth_response.json()
    access_token = auth_data["access_token"]
    
    # Create engine and context
    engine = OMHWorkflowEngine(autorl_api, omh_api)
    context = engine.create_context_from_token(access_token)
    
    return engine, context

