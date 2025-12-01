#!/usr/bin/env python3
"""
AutoRL Startup Script
Starts all necessary services for AutoRL platform
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

processes = []

def cleanup():
    """Kill all spawned processes"""
    print_warning("Shutting down all services...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print_success("All services stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\n")
    cleanup()
    sys.exit(0)

def check_dependencies():
    """Check if required dependencies are installed"""
    print_info("Checking dependencies...")
    
    # Check Python packages
    try:
        import fastapi
        import uvicorn
        print_success("Backend dependencies OK")
    except ImportError:
        print_error("Missing Python dependencies. Run: pip install -r requirements.txt")
        return False
    
    # Check if node_modules exists
    if not Path("node_modules").exists():
        print_error("Frontend dependencies missing. Run: npm install")
        return False
    
    print_success("All dependencies OK")
    return True

def start_backend():
    """Start the master backend server"""
    print_info("Starting Master Backend Server...")
    
    env = os.environ.copy()
    env['AUTORL_MODE'] = env.get('AUTORL_MODE', 'demo')  # Default to demo mode
    
    proc = subprocess.Popen(
        [sys.executable, "master_backend.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    processes.append(proc)
    
    # Wait for server to start
    time.sleep(3)
    
    if proc.poll() is None:
        print_success("Master Backend Server started on http://localhost:5000")
        return True
    else:
        print_error("Failed to start backend server")
        return False

def start_frontend():
    """Start the frontend development server"""
    print_info("Starting Frontend Development Server...")
    
    # Determine the command based on OS
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    
    proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    processes.append(proc)
    
    # Wait for server to start
    time.sleep(3)
    
    if proc.poll() is None:
        print_success("Frontend Development Server started on http://localhost:5173")
        return True
    else:
        print_error("Failed to start frontend server")
        return False

def main():
    """Main startup sequence"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print_header("AutoRL Platform Startup")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start backend
    if not start_backend():
        cleanup()
        sys.exit(1)
    
    # Start frontend
    if not start_frontend():
        cleanup()
        sys.exit(1)
    
    # Print status
    print_header("AutoRL Platform Running")
    print_success("All services started successfully!\n")
    
    print(f"{Colors.BOLD}Service URLs:{Colors.ENDC}")
    print(f"  📊 Dashboard:      {Colors.OKCYAN}http://localhost:5173{Colors.ENDC}")
    print(f"  🔌 Backend API:    {Colors.OKCYAN}http://localhost:5000{Colors.ENDC}")
    print(f"  📈 Metrics:        {Colors.OKCYAN}http://localhost:8000/metrics{Colors.ENDC}")
    print(f"  🔗 WebSocket:      {Colors.OKCYAN}ws://localhost:5000/ws{Colors.ENDC}")
    print(f"\n{Colors.BOLD}API Documentation:{Colors.ENDC}")
    print(f"  📚 Swagger UI:     {Colors.OKCYAN}http://localhost:5000/docs{Colors.ENDC}")
    print(f"  📖 ReDoc:          {Colors.OKCYAN}http://localhost:5000/redoc{Colors.ENDC}")
    
    print(f"\n{Colors.WARNING}Press Ctrl+C to stop all services{Colors.ENDC}\n")
    
    # Keep script running and monitor processes
    try:
        while True:
            time.sleep(1)
            # Check if any process died
            for proc in processes:
                if proc.poll() is not None:
                    print_error("A service has stopped unexpectedly!")
                    cleanup()
                    sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()

