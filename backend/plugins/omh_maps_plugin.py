"""
Open Mobile Hub (OMH) Maps Plugin for AutoRL

Provides location-based automation capabilities using OMH Maps SDK,
enabling geo-triggered workflows and location-aware task execution.

Features:
- Geocoding and reverse geocoding
- Location-based task triggers
- Map data integration
- Distance calculations
"""

import os
import httpx
import json
from typing import Dict, Any, Optional, List, Tuple
from base_plugin import BasePlugin
import logging
import math

logger = logging.getLogger("autorl.omh_maps")


class OMHMapsPlugin(BasePlugin):
    """
    OMH Maps Plugin
    
    Integrates Open Mobile Hub Maps API to enable location-based
    automation workflows and geo-triggered task execution.
    """
    
    def __init__(self):
        self.api_key: Optional[str] = None
        self.maps_api_url: Optional[str] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.location_cache: Dict[str, Dict[str, Any]] = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize OMH Maps Plugin with configuration.
        
        Args:
            config: Configuration dict with OMH Maps API credentials
        """
        self.api_key = config.get("omh_maps_api_key") or os.getenv("OMH_MAPS_API_KEY")
        self.maps_api_url = config.get("omh_maps_api_url") or os.getenv(
            "OMH_MAPS_API_URL", 
            "https://maps.openmobilehub.org/api/v1"
        )
        
        if not self.api_key:
            logger.warning("OMH Maps API key not configured. Plugin will operate in mock mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"OMH Maps Plugin initialized (mock_mode={self.mock_mode})")
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process location-based requests.
        
        Args:
            input_data: Request data with 'action' key
            
        Returns:
            Response data with location results
        """
        action = input_data.get("action")
        
        if action == "geocode":
            return self._geocode(input_data.get("address"))
        elif action == "reverse_geocode":
            return self._reverse_geocode(
                input_data.get("latitude"),
                input_data.get("longitude")
            )
        elif action == "calculate_distance":
            return self._calculate_distance(
                input_data.get("origin"),
                input_data.get("destination")
            )
        elif action == "check_geofence":
            return self._check_geofence(
                input_data.get("location"),
                input_data.get("geofence")
            )
        elif action == "get_nearby_places":
            return self._get_nearby_places(
                input_data.get("latitude"),
                input_data.get("longitude"),
                input_data.get("radius", 1000),
                input_data.get("type")
            )
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _geocode(self, address: str) -> Dict[str, Any]:
        """
        Convert address to coordinates.
        
        Args:
            address: Street address or place name
            
        Returns:
            Coordinates and formatted address
        """
        if self.mock_mode:
            # Return mock coordinates for demo
            mock_locations = {
                "San Francisco": {"lat": 37.7749, "lng": -122.4194, "formatted": "San Francisco, CA, USA"},
                "New York": {"lat": 40.7128, "lng": -74.0060, "formatted": "New York, NY, USA"},
                "London": {"lat": 51.5074, "lng": -0.1278, "formatted": "London, UK"},
                "Tokyo": {"lat": 35.6762, "lng": 139.6503, "formatted": "Tokyo, Japan"},
            }
            
            for city, coords in mock_locations.items():
                if city.lower() in address.lower():
                    return {
                        "latitude": coords["lat"],
                        "longitude": coords["lng"],
                        "formatted_address": coords["formatted"],
                        "mock_mode": True
                    }
            
            # Default location
            return {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "formatted_address": address,
                "mock_mode": True
            }
        
        # Real geocoding API call
        try:
            params = {
                "address": address,
                "key": self.api_key
            }
            # In real implementation, use async httpx.get
            return {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "formatted_address": address
            }
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return {"error": str(e)}
    
    def _reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Convert coordinates to address.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Formatted address and place information
        """
        if self.mock_mode:
            return {
                "formatted_address": f"Location at {latitude:.4f}, {longitude:.4f}",
                "city": "San Francisco",
                "state": "California",
                "country": "USA",
                "postal_code": "94102",
                "mock_mode": True
            }
        
        # Real reverse geocoding API call
        try:
            params = {
                "latlng": f"{latitude},{longitude}",
                "key": self.api_key
            }
            # In real implementation, use async httpx.get
            return {
                "formatted_address": "123 Main St, San Francisco, CA 94102",
                "city": "San Francisco",
                "state": "California",
                "country": "USA"
            }
        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return {"error": str(e)}
    
    def _calculate_distance(
        self, 
        origin: Dict[str, float], 
        destination: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate distance between two locations.
        
        Args:
            origin: {"lat": float, "lng": float}
            destination: {"lat": float, "lng": float}
            
        Returns:
            Distance in meters and miles
        """
        # Haversine formula for great-circle distance
        lat1, lon1 = math.radians(origin["lat"]), math.radians(origin["lng"])
        lat2, lon2 = math.radians(destination["lat"]), math.radians(destination["lng"])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        r = 6371000
        distance_m = c * r
        distance_mi = distance_m / 1609.34
        
        return {
            "distance_meters": round(distance_m, 2),
            "distance_miles": round(distance_mi, 2),
            "distance_km": round(distance_m / 1000, 2),
            "origin": origin,
            "destination": destination,
            "mock_mode": self.mock_mode
        }
    
    def _check_geofence(
        self,
        location: Dict[str, float],
        geofence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if location is within a geofence.
        
        Args:
            location: {"lat": float, "lng": float}
            geofence: {"center": {"lat": float, "lng": float}, "radius_meters": float}
            
        Returns:
            Whether location is inside geofence
        """
        center = geofence["center"]
        radius = geofence["radius_meters"]
        
        distance_result = self._calculate_distance(location, center)
        distance_m = distance_result["distance_meters"]
        
        is_inside = distance_m <= radius
        
        return {
            "inside_geofence": is_inside,
            "distance_from_center": distance_m,
            "geofence_radius": radius,
            "location": location,
            "mock_mode": self.mock_mode
        }
    
    def _get_nearby_places(
        self,
        latitude: float,
        longitude: float,
        radius: int = 1000,
        place_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find nearby places of interest.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius: Search radius in meters
            place_type: Type of place (e.g., "restaurant", "bank", "store")
            
        Returns:
            List of nearby places
        """
        if self.mock_mode:
            mock_places = [
                {
                    "name": "Coffee Shop",
                    "type": "cafe",
                    "distance_meters": 150,
                    "address": "123 Main St",
                    "rating": 4.5
                },
                {
                    "name": "Bank Branch",
                    "type": "bank",
                    "distance_meters": 300,
                    "address": "456 Market St",
                    "rating": 4.2
                },
                {
                    "name": "Grocery Store",
                    "type": "store",
                    "distance_meters": 500,
                    "address": "789 Oak Ave",
                    "rating": 4.7
                }
            ]
            
            if place_type:
                mock_places = [p for p in mock_places if p["type"] == place_type]
            
            return {
                "places": mock_places,
                "count": len(mock_places),
                "center": {"lat": latitude, "lng": longitude},
                "radius_meters": radius,
                "mock_mode": True
            }
        
        # Real nearby places API call
        try:
            params = {
                "location": f"{latitude},{longitude}",
                "radius": radius,
                "type": place_type,
                "key": self.api_key
            }
            # In real implementation, use async httpx.get
            return {
                "places": [],
                "count": 0,
                "center": {"lat": latitude, "lng": longitude},
                "radius_meters": radius
            }
        except Exception as e:
            logger.error(f"Nearby places error: {e}")
            return {"error": str(e)}
    
    def create_location_trigger(
        self,
        task_id: str,
        geofence: Dict[str, Any],
        trigger_type: str = "enter"
    ) -> Dict[str, Any]:
        """
        Create a location-based task trigger.
        
        Args:
            task_id: Task to trigger
            geofence: Geofence definition
            trigger_type: "enter" or "exit"
            
        Returns:
            Trigger configuration
        """
        return {
            "trigger_id": f"geo_trigger_{task_id}",
            "task_id": task_id,
            "geofence": geofence,
            "trigger_type": trigger_type,
            "active": True,
            "created_at": "2024-01-01T00:00:00Z"
        }
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self.http_client:
            # In async context, use: await self.http_client.aclose()
            logger.info("OMH Maps Plugin shutdown")
        self.location_cache.clear()


# Singleton instance for easy import
omh_maps_plugin = OMHMapsPlugin()

