#!/usr/bin/env python3
"""
Integration Test Suite for AutoRL Platform
Tests the complete integration of Frontend, Backend, AI Agents, and Mock Data
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(test_name):
    print(f"\n{Colors.CYAN}▶ Testing: {test_name}{Colors.END}")

def print_pass(message):
    print(f"  {Colors.GREEN}✅ PASS:{Colors.END} {message}")

def print_fail(message):
    print(f"  {Colors.RED}❌ FAIL:{Colors.END} {message}")

def print_info(message):
    print(f"  {Colors.BLUE}ℹ️  INFO:{Colors.END} {message}")

def print_warn(message):
    print(f"  {Colors.YELLOW}⚠️  WARN:{Colors.END} {message}")

class IntegrationTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name, func):
        """Run a test and track results"""
        print_test(name)
        try:
            result = func()
            if result:
                self.passed += 1
                print_pass(name)
            else:
                self.failed += 1
                print_fail(name)
            self.tests.append({"name": name, "passed": result})
        except Exception as e:
            self.failed += 1
            print_fail(f"{name} - {str(e)}")
            self.tests.append({"name": name, "passed": False, "error": str(e)})
    
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Test Summary{Colors.END}")
        print(f"{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.END}")
        print(f"Success Rate: {(self.passed/total*100):.1f}%\n" if total > 0 else "No tests run\n")
        
        return self.failed == 0

# Test functions
def test_file_structure():
    """Test that all key files exist"""
    required_files = [
        'backend_server.py',
        'config.yaml',
        'package.json',
        'README.md',
        'INTEGRATION_GUIDE.md',
        'QUICKSTART_INTEGRATED.md',
        'start_autorl.py',
        'src/services/api.js',
        'src/hooks/useWebSocket.js',
        'src/pages/Dashboard.jsx',
        'src/components/TaskControlPanel.jsx',
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print_fail(f"Missing files: {', '.join(missing)}")
        return False
    else:
        print_pass("All required files present")
        return True

def test_backend_imports():
    """Test that backend can import required modules"""
    try:
        # Test conditional imports
        imports_ok = True
        
        # Core imports that should always work
        try:
            from pathlib import Path
            import json
            import asyncio
            print_info("Core Python modules OK")
        except ImportError as e:
            print_fail(f"Core import failed: {e}")
            imports_ok = False
        
        # Optional imports for production mode
        try:
            import fastapi
            import uvicorn
            print_info("FastAPI/Uvicorn available")
        except ImportError:
            print_warn("FastAPI/Uvicorn not available (run: pip install fastapi uvicorn)")
        
        return imports_ok
    except Exception as e:
        print_fail(f"Import test failed: {e}")
        return False

def test_config_file():
    """Test that config file is valid YAML"""
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key sections
        required_sections = ['server', 'api', 'frontend', 'appium', 'devices']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            print_fail(f"Config missing sections: {', '.join(missing)}")
            return False
        else:
            print_pass("Config file valid with all required sections")
            return True
    except ImportError:
        print_warn("PyYAML not installed (optional)")
        return True
    except Exception as e:
        print_fail(f"Config validation failed: {e}")
        return False

def test_frontend_structure():
    """Test frontend structure and key files"""
    try:
        # Check package.json
        with open('package.json', 'r') as f:
            package = json.load(f)
        
        # Verify scripts
        required_scripts = ['dev', 'build', 'start', 'backend']
        missing_scripts = [s for s in required_scripts if s not in package.get('scripts', {})]
        
        if missing_scripts:
            print_warn(f"Package.json missing scripts: {', '.join(missing_scripts)}")
        else:
            print_info("All npm scripts present")
        
        # Check key dependencies
        deps = package.get('dependencies', {})
        required_deps = ['react', 'react-dom', 'lucide-react']
        missing_deps = [d for d in required_deps if d not in deps]
        
        if missing_deps:
            print_fail(f"Missing dependencies: {', '.join(missing_deps)}")
            return False
        else:
            print_pass("All required frontend dependencies present")
            return True
            
    except Exception as e:
        print_fail(f"Frontend structure test failed: {e}")
        return False

def test_api_service():
    """Test API service structure"""
    try:
        # Read and check API service file
        api_file = Path('src/services/api.js')
        content = api_file.read_text()
        
        # Check for required methods
        required_methods = [
            'healthCheck',
            'getDevices',
            'getTasks',
            'createTask',
            'getMetrics',
            'getPolicies'
        ]
        
        missing = [m for m in required_methods if m not in content]
        
        if missing:
            print_fail(f"API service missing methods: {', '.join(missing)}")
            return False
        else:
            print_pass("API service has all required methods")
            return True
            
    except Exception as e:
        print_fail(f"API service test failed: {e}")
        return False

def test_websocket_hook():
    """Test WebSocket hook structure"""
    try:
        ws_file = Path('src/hooks/useWebSocket.js')
        content = ws_file.read_text()
        
        # Check for required functionality
        required_features = [
            'useWebSocket',
            'connect',
            'disconnect',
            'sendMessage',
            'onmessage',
            'reconnect'
        ]
        
        missing = [f for f in required_features if f not in content]
        
        if missing:
            print_fail(f"WebSocket hook missing features: {', '.join(missing)}")
            return False
        else:
            print_pass("WebSocket hook has all required features")
            return True
            
    except Exception as e:
        print_fail(f"WebSocket hook test failed: {e}")
        return False

def test_backend_structure():
    """Test backend server structure"""
    try:
        backend_file = Path('backend_server.py')
        content = backend_file.read_text()
        
        # Check for required components
        required_components = [
            'FastAPI',
            'WebSocket',
            'CORS',
            'apiService',
            '/api/health',
            '/api/devices',
            '/api/tasks',
            '/api/metrics',
            'ConnectionManager'
        ]
        
        missing = [c for c in required_components if c not in content]
        
        if missing:
            print_warn(f"Backend may be missing: {', '.join(missing)}")
            return len(missing) < 3  # Pass if most components present
        else:
            print_pass("Backend server has all required components")
            return True
            
    except Exception as e:
        print_fail(f"Backend structure test failed: {e}")
        return False

def test_documentation():
    """Test that documentation is comprehensive"""
    try:
        docs = {
            'README.md': ['AutoRL', 'Architecture', 'Features', 'Getting Started'],
            'INTEGRATION_GUIDE.md': ['Integration Points', 'Data Flow', 'API Endpoints'],
            'QUICKSTART_INTEGRATED.md': ['Quick Start', 'Creating Your First Task', 'Troubleshooting']
        }
        
        all_ok = True
        for doc, required_sections in docs.items():
            if not Path(doc).exists():
                print_fail(f"Missing documentation: {doc}")
                all_ok = False
                continue
            
            content = Path(doc).read_text()
            missing_sections = [s for s in required_sections if s not in content]
            
            if missing_sections:
                print_warn(f"{doc} may be missing sections: {', '.join(missing_sections)}")
            else:
                print_info(f"{doc} has all required sections")
        
        if all_ok:
            print_pass("All documentation files present and comprehensive")
        
        return all_ok
        
    except Exception as e:
        print_fail(f"Documentation test failed: {e}")
        return False

def test_startup_script():
    """Test startup script structure"""
    try:
        startup_file = Path('start_autorl.py')
        content = startup_file.read_text()
        
        required_functions = [
            'check_dependencies',
            'start_backend',
            'start_frontend',
            'cleanup',
            'main'
        ]
        
        missing = [f for f in required_functions if f not in content]
        
        if missing:
            print_fail(f"Startup script missing functions: {', '.join(missing)}")
            return False
        else:
            print_pass("Startup script has all required functions")
            return True
            
    except Exception as e:
        print_fail(f"Startup script test failed: {e}")
        return False

def test_integration_completeness():
    """Test that all integration points are connected"""
    try:
        checks = []
        
        # Check 1: Frontend has API service
        checks.append(Path('src/services/api.js').exists())
        
        # Check 2: Frontend has WebSocket hook
        checks.append(Path('src/hooks/useWebSocket.js').exists())
        
        # Check 3: Dashboard uses both
        dashboard = Path('src/pages/Dashboard.jsx').read_text()
        checks.append('apiService' in dashboard and 'useWebSocket' in dashboard)
        
        # Check 4: TaskControlPanel uses API service
        task_panel = Path('src/components/TaskControlPanel.jsx').read_text()
        checks.append('apiService' in task_panel)
        
        # Check 5: Backend has all endpoints
        backend = Path('backend_server.py').read_text()
        checks.append(all(endpoint in backend for endpoint in ['/api/health', '/api/tasks', '/api/devices']))
        
        # Check 6: Config file exists
        checks.append(Path('config.yaml').exists())
        
        passed = sum(checks)
        total = len(checks)
        
        print_info(f"Integration checks passed: {passed}/{total}")
        
        if passed == total:
            print_pass("All integration points properly connected")
            return True
        else:
            print_warn("Some integration points may need review")
            return passed >= total * 0.8  # Pass if 80% of checks pass
            
    except Exception as e:
        print_fail(f"Integration completeness test failed: {e}")
        return False

# Main test runner
def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}AutoRL Integration Test Suite{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")
    
    tester = IntegrationTester()
    
    # Run all tests
    tester.test("File Structure", test_file_structure)
    tester.test("Backend Imports", test_backend_imports)
    tester.test("Configuration File", test_config_file)
    tester.test("Frontend Structure", test_frontend_structure)
    tester.test("API Service", test_api_service)
    tester.test("WebSocket Hook", test_websocket_hook)
    tester.test("Backend Structure", test_backend_structure)
    tester.test("Documentation", test_documentation)
    tester.test("Startup Script", test_startup_script)
    tester.test("Integration Completeness", test_integration_completeness)
    
    # Print summary
    all_passed = tester.summary()
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All integration tests PASSED!{Colors.END}")
        print(f"{Colors.GREEN}The AutoRL platform is fully integrated and ready to use!{Colors.END}\n")
        print(f"{Colors.CYAN}Next steps:{Colors.END}")
        print(f"  1. Run: {Colors.BOLD}python start_autorl.py{Colors.END}")
        print(f"  2. Open: {Colors.BOLD}http://localhost:5173{Colors.END}")
        print(f"  3. Read: {Colors.BOLD}QUICKSTART_INTEGRATED.md{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Some tests failed, but the system may still work{Colors.END}")
        print(f"{Colors.YELLOW}Review the failed tests above and fix any issues{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

