#!/usr/bin/env python3
"""
AutoRL Critical Fixes Validation Script
Tests all the implemented critical fixes
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
from typing import Dict, Any

class AutoRLTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000/ws/metrics"
        self.test_results = {}
        
    async def test_fastapi_startup(self) -> bool:
        """Test 1: FastAPI startup and basic endpoints"""
        print("🚀 Testing FastAPI startup...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test root endpoint
                async with session.get(f"{self.base_url}/") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Root endpoint: {data}")
                    else:
                        print(f"❌ Root endpoint failed: {response.status}")
                        return False
                
                # Test health endpoint
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Health endpoint: {data}")
                    else:
                        print(f"❌ Health endpoint failed: {response.status}")
                        return False
                
                # Test metrics endpoint
                async with session.get(f"{self.base_url}/metrics") as response:
                    if response.status == 200:
                        print("✅ Prometheus metrics endpoint accessible")
                    else:
                        print(f"❌ Metrics endpoint failed: {response.status}")
                        return False
                
                return True
                
        except Exception as e:
            print(f"❌ FastAPI startup test failed: {e}")
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test 2: WebSocket connection and messaging"""
        print("🔌 Testing WebSocket connection...")
        
        try:
            import websockets
            
            async with websockets.connect(self.ws_url) as websocket:
                # Wait for connection confirmation
                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(message)
                
                if data.get("type") == "connection_established":
                    print("✅ WebSocket connection established")
                    
                    # Send ping
                    ping_msg = {"type": "ping", "timestamp": time.time()}
                    await websocket.send(json.dumps(ping_msg))
                    
                    # Wait for pong
                    pong_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                    pong_data = json.loads(pong_msg)
                    
                    if pong_data.get("type") == "pong":
                        print("✅ WebSocket ping/pong working")
                        return True
                    else:
                        print(f"❌ Unexpected pong message: {pong_data}")
                        return False
                else:
                    print(f"❌ Unexpected connection message: {data}")
                    return False
                    
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            return False
    
    async def test_task_execution(self) -> bool:
        """Test 3: Task execution with error handling"""
        print("📱 Testing task execution...")
        
        try:
            async with aiohttp.ClientSession() as session:
                task_request = {
                    "task_description": "Test login automation",
                    "device_id": "emulator-5554",
                    "timeout_seconds": 30
                }
                
                async with session.post(
                    f"{self.base_url}/api/v1/execute",
                    json=task_request
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Task execution initiated: {data}")
                        return True
                    else:
                        error_data = await response.json()
                        print(f"❌ Task execution failed: {error_data}")
                        return False
                        
        except Exception as e:
            print(f"❌ Task execution test failed: {e}")
            return False
    
    async def test_pii_masking(self) -> bool:
        """Test 4: PII masking functionality"""
        print("🔒 Testing PII masking...")
        
        try:
            # Import PII manager
            sys.path.append('src')
            from security.pii_manager import PIIMaskingManager
            
            pii_manager = PIIMaskingManager()
            
            # Test data with PII
            test_data = {
                "user_email": "test@example.com",
                "phone": "1234567890",
                "api_key": "sk-1234567890abcdef1234567890abcdef1234567890abcdef12",
                "password": "secretpassword123"
            }
            
            masked_data = pii_manager.mask_log_data(test_data)
            
            # Check if PII is masked
            if "@" in str(masked_data["user_email"]):
                print("❌ Email not properly masked")
                return False
            
            if "sk-" in str(masked_data["api_key"]):
                print("❌ API key not properly masked")
                return False
                
            print("✅ PII masking working correctly")
            return True
            
        except Exception as e:
            print(f"❌ PII masking test failed: {e}")
            return False
    
    def test_prerequisites(self) -> bool:
        """Test 5: System prerequisites"""
        print("🔧 Testing system prerequisites...")
        
        try:
            # Check if curl is available
            result = subprocess.run(["curl", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ curl not available")
                return False
            
            # Check if adb is available
            result = subprocess.run(["adb", "version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ adb not available")
                return False
            
            print("✅ System prerequisites met")
            return True
            
        except Exception as e:
            print(f"❌ Prerequisites test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all critical tests"""
        print("🧪 Starting AutoRL Critical Fixes Validation")
        print("=" * 50)
        
        tests = [
            ("FastAPI Startup", self.test_fastapi_startup),
            ("WebSocket Connection", self.test_websocket_connection),
            ("Task Execution", self.test_task_execution),
            ("PII Masking", self.test_pii_masking),
            ("System Prerequisites", self.test_prerequisites)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                self.test_results[test_name] = result
                if result:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                self.test_results[test_name] = False
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All critical fixes are working correctly!")
            print("\n✅ Your AutoRL system is ready for production!")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
            print("\n🔧 Fix the failing tests before proceeding to production.")
        
        return passed == total

async def main():
    """Main test runner"""
    tester = AutoRLTester()
    
    print("AutoRL Critical Fixes Validation")
    print("This script tests all the implemented critical fixes")
    print()
    print("Make sure the AutoRL API is running:")
    print("  uvicorn src.main:app --host 0.0.0.0 --port 8000")
    print()
    
    input("Press Enter to start testing...")
    
    success = await tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready to deploy!")
        sys.exit(0)
    else:
        print("\n🛠️  Please fix the failing tests before deployment.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())




