"""
Complete AutoRL + OMH Integration Demo
=======================================
End-to-end demonstration of AutoRL mobile automation with OMH authentication
and location services.
"""

import sys
import time
from agent_service.autorl_omh_workflows import (
    get_authenticated_engine,
    WorkflowTemplates,
    OMHWorkflowEngine
)


def print_header(text: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(result: dict):
    """Pretty print workflow result"""
    if result.get("success"):
        print("✅ Success!")
        if "result" in result:
            print(f"   Result: {result['result']}")
    else:
        print("❌ Failed!")
        print(f"   Error: {result.get('error', 'Unknown error')}")


def demo_authentication():
    """Demo 1: OMH Authentication"""
    print_header("Demo 1: OMH Authentication")
    
    print("\n1. Authenticating user 'alice.smith' with OMH...")
    try:
        engine, context = get_authenticated_engine("alice.smith", "test")
        print(f"   ✅ Authenticated as: {context.username}")
        print(f"   📧 Email: {context.email}")
        print(f"   👥 Roles: {', '.join(context.roles)}")
        if context.location:
            print(f"   📍 Location: {context.location['address']}")
        return engine, context
    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print("\n⚠️  Make sure OMH Mock Server is running on http://localhost:8001")
        print("   Start it with: cd src/agent_service && python omh_mock_server.py")
        return None, None


def demo_location_context(engine: OMHWorkflowEngine, context):
    """Demo 2: Location Context Retrieval"""
    print_header("Demo 2: Location Context")
    
    if not context.location:
        print("❌ Location not available")
        return
    
    print("\nCurrent location context:")
    print(f"   📍 Address: {context.location['address']}")
    print(f"   🌍 Coordinates: ({context.location['latitude']}, {context.location['longitude']})")
    print(f"   🆔 Place ID: {context.location['place_id']}")


def demo_location_based_search(engine: OMHWorkflowEngine, context):
    """Demo 3: Location-Based Search"""
    print_header("Demo 3: Location-Based Search Workflows")
    
    device_id = "emulator-5554"
    
    print(f"\nUsing device: {device_id}")
    
    # Demo 3a: Restaurant finder
    print("\n3a. Finding Italian restaurants nearby...")
    result = WorkflowTemplates.restaurant_finder(engine, context, "Italian", device_id)
    print_result(result)
    
    # Demo 3b: Gas station locator
    print("\n3b. Finding nearest gas stations...")
    result = WorkflowTemplates.gas_station_locator(engine, context, device_id)
    print_result(result)
    
    # Demo 3c: Pharmacy finder
    print("\n3c. Finding open pharmacies...")
    result = WorkflowTemplates.pharmacy_finder(engine, context, device_id, open_now=True)
    print_result(result)


def demo_navigation_workflows(engine: OMHWorkflowEngine, context):
    """Demo 4: Navigation Workflows"""
    print_header("Demo 4: Navigation & Route Planning")
    
    device_id = "emulator-5554"
    
    # Demo 4a: Navigation to destination
    print("\n4a. Planning route to downtown...")
    result = engine.execute_navigation_workflow(
        context,
        "Downtown City Center",
        device_id
    )
    print_result(result)
    
    # Demo 4b: Commute optimization
    print("\n4b. Optimizing commute with traffic...")
    result = WorkflowTemplates.commute_optimizer(
        engine, context,
        "123 Main Street, Office Building",
        device_id
    )
    print_result(result)


def demo_advanced_workflows(engine: OMHWorkflowEngine, context):
    """Demo 5: Advanced Location-Aware Workflows"""
    print_header("Demo 5: Advanced Location-Aware Workflows")
    
    device_id = "emulator-5554"
    
    # Demo 5a: Context-aware reminder
    print("\n5a. Creating location-based reminder...")
    result = engine.execute_context_aware_reminder(
        context,
        "Buy groceries",
        device_id
    )
    print_result(result)
    
    # Demo 5b: Local events
    print("\n5b. Finding local events...")
    result = WorkflowTemplates.local_events_finder(
        engine, context,
        "music concerts",
        device_id
    )
    print_result(result)
    
    # Demo 5c: Price comparison
    print("\n5c. Comparing prices at nearby stores...")
    result = WorkflowTemplates.price_comparison_workflow(
        engine, context,
        "iPhone 15 Pro",
        device_id
    )
    print_result(result)


def demo_multi_location_workflow(engine: OMHWorkflowEngine, context):
    """Demo 6: Multi-Location Workflow"""
    print_header("Demo 6: Multi-Location Workflow")
    
    device_id = "emulator-5554"
    
    print("\n6. Checking store hours at multiple locations...")
    locations = [
        "Walmart on 5th Avenue",
        "Target Downtown",
        "Best Buy Tech Center"
    ]
    
    result = engine.execute_multi_location_workflow(
        context,
        locations,
        "Check store hours",
        device_id
    )
    
    if result.get("success"):
        print(f"✅ Completed {result['total_locations']} location checks")
        for loc_result in result["results"]:
            status = "✅" if loc_result["success"] else "❌"
            print(f"   {status} {loc_result['location']}")
    else:
        print("❌ Multi-location workflow failed")


def demo_role_based_access(engine: OMHWorkflowEngine, context):
    """Demo 7: Role-Based Access Control"""
    print_header("Demo 7: Role-Based Access Control")
    
    print(f"\nUser roles: {', '.join(context.roles)}")
    
    if "admin" in context.roles:
        print("✅ User has admin privileges")
        print("   Can access admin-only endpoints")
        print("   Can reset quotas for other users")
    else:
        print("ℹ️  User has standard privileges")
        print("   Cannot access admin endpoints")


def demo_task_history_and_metrics():
    """Demo 8: Task History and Metrics"""
    print_header("Demo 8: Task History & Metrics")
    
    print("\nThis demo would show:")
    print("   📊 Tasks executed with location context")
    print("   ⏱️  Average task execution time by location")
    print("   📈 Success rates for location-aware tasks")
    print("   🗺️  Heat map of task execution locations")
    print("\n(Connect to AutoRL metrics dashboard for live data)")


def main():
    """Run complete integration demo"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "AutoRL + OMH Integration Demo" + " " * 24 + "║")
    print("║" + " " * 15 + "Location-Aware Mobile Automation" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Check if services are running
    print("\n🔍 Checking services...")
    
    try:
        import requests
        
        # Check OMH Mock Server
        try:
            requests.get("http://localhost:8001/health", timeout=2)
            print("   ✅ OMH Mock Server running (port 8001)")
        except:
            print("   ❌ OMH Mock Server not running")
            print("      Start with: cd src/agent_service && python omh_mock_server.py")
            sys.exit(1)
        
        # Check AutoRL API (optional)
        try:
            requests.get("http://localhost:8000/docs", timeout=2)
            print("   ✅ AutoRL API Server running (port 8000)")
        except:
            print("   ⚠️  AutoRL API Server not detected (port 8000)")
            print("      Some features may be limited")
    
    except ImportError:
        print("   ⚠️  requests library not found, skipping service checks")
    
    # Run demos
    try:
        # Demo 1: Authentication
        engine, context = demo_authentication()
        if not engine or not context:
            return
        
        time.sleep(1)
        
        # Demo 2: Location Context
        demo_location_context(engine, context)
        time.sleep(1)
        
        # Demo 3: Location-Based Search
        demo_location_based_search(engine, context)
        time.sleep(1)
        
        # Demo 4: Navigation
        demo_navigation_workflows(engine, context)
        time.sleep(1)
        
        # Demo 5: Advanced Workflows
        demo_advanced_workflows(engine, context)
        time.sleep(1)
        
        # Demo 6: Multi-Location
        demo_multi_location_workflow(engine, context)
        time.sleep(1)
        
        # Demo 7: Role-Based Access
        demo_role_based_access(engine, context)
        time.sleep(1)
        
        # Demo 8: Metrics
        demo_task_history_and_metrics()
        
        # Summary
        print_header("Demo Complete!")
        print("\n✅ All demos completed successfully!")
        print("\nNext steps:")
        print("   1. Explore the AutoRL API docs at http://localhost:8000/docs")
        print("   2. Try OMH Mock Server docs at http://localhost:8001/docs")
        print("   3. View workflow templates in: src/agent_service/autorl_omh_workflows.py")
        print("   4. Check integration examples in: examples/")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

