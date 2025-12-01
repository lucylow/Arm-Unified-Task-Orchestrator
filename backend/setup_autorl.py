#!/usr/bin/env python3
"""
AutoRL Setup and Verification Script
Checks dependencies, configuration, and provides helpful diagnostics
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "success": Colors.GREEN + "✓",
        "warning": Colors.YELLOW + "⚠",
        "error": Colors.RED + "✗",
        "info": Colors.BLUE + "ℹ"
    }
    icon = colors.get(status, Colors.BLUE + "ℹ")
    print(f"{icon} {message}{Colors.RESET}")

def check_python_version():
    """Check if Python version is 3.9+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} detected. Requires Python 3.9+", "error")
        return False

def check_node_version():
    """Check if Node.js is installed and version"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        version = result.stdout.strip()
        major_version = int(version.replace('v', '').split('.')[0])
        if major_version >= 16:
            print_status(f"Node.js {version} ✓", "success")
            return True
        else:
            print_status(f"Node.js {version} detected. Requires Node.js 16+", "warning")
            return False
    except FileNotFoundError:
        print_status("Node.js not found. Please install Node.js 16+", "error")
        return False

def check_appium():
    """Check if Appium is installed"""
    try:
        result = subprocess.run(['appium', '--version'], capture_output=True, text=True)
        version = result.stdout.strip()
        print_status(f"Appium {version} ✓", "success")
        return True
    except FileNotFoundError:
        print_status("Appium not found. Install with: npm install -g appium", "warning")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        version_line = result.stdout.split('\n')[0]
        print_status(f"Tesseract {version_line} ✓", "success")
        return True
    except FileNotFoundError:
        print_status("Tesseract OCR not found. OCR features will be disabled", "warning")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path('.env')
    env_example = Path('.env.example')
    
    if not env_path.exists():
        print_status(".env file not found", "warning")
        if env_example.exists():
            print_status("  → Copy .env.example to .env and update with your API keys", "info")
            print(f"     {Colors.YELLOW}cp .env.example .env{Colors.RESET}")
        elif Path('env.template').exists():
            print_status("  → Copy env.template to .env and update with your API keys", "info")
            print(f"     {Colors.YELLOW}cp env.template .env{Colors.RESET}")
        return False
    
    # Check for OPENAI_API_KEY
    with open(env_path) as f:
        env_content = f.read()
        if 'OPENAI_API_KEY=sk-' not in env_content and 'your_openai_api_key_here' in env_content:
            print_status(".env file exists but OPENAI_API_KEY not set", "warning")
            print_status("  → System will run in DEMO mode without AI features", "info")
            return True
        else:
            print_status(".env file configured ✓", "success")
            return True

def check_config_yaml():
    """Check if config.yaml exists and is valid"""
    config_path = Path('config.yaml')
    
    if not config_path.exists():
        print_status("config.yaml not found", "error")
        return False
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['server', 'api', 'devices', 'llm', 'reinforcement_learning']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            print_status(f"config.yaml missing sections: {', '.join(missing)}", "error")
            return False
        
        print_status("config.yaml is valid ✓", "success")
        
        # Show current mode
        mode = config.get('server', {}).get('mode', 'unknown')
        print_status(f"  → Running in {mode.upper()} mode", "info")
        
        return True
    except yaml.YAMLError as e:
        print_status(f"config.yaml is invalid: {e}", "error")
        return False

def check_python_packages():
    """Check if Python packages are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        print_status("Core Python packages installed ✓", "success")
        return True
    except ImportError as e:
        print_status(f"Missing Python packages", "error")
        print_status("  → Run: pip install -r requirements.txt", "info")
        return False

def check_node_packages():
    """Check if Node.js packages are installed"""
    node_modules = Path('node_modules')
    if node_modules.exists():
        print_status("Node.js packages installed ✓", "success")
        return True
    else:
        print_status("Node.js packages not installed", "warning")
        print_status("  → Run: npm install", "info")
        return False

def check_devices():
    """Check connected Android/iOS devices"""
    print_status("\nChecking connected devices:", "info")
    
    # Check Android devices
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = [line.split()[0] for line in result.stdout.split('\n')[1:] if '\tdevice' in line]
        if devices:
            print_status(f"  Android devices: {', '.join(devices)}", "success")
        else:
            print_status("  No Android devices connected", "warning")
    except FileNotFoundError:
        print_status("  ADB not found - Android debugging unavailable", "warning")
    
    # Check iOS simulators (macOS only)
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(['xcrun', 'simctl', 'list', 'devices'], capture_output=True, text=True)
            print_status("  iOS simulators available (check with: xcrun simctl list)", "info")
        except FileNotFoundError:
            print_status("  Xcode command line tools not found", "warning")

def print_startup_commands():
    """Print commands to start the application"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}🚀 AutoRL Startup Commands{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}1. Start Backend Server (Terminal 1):{Colors.RESET}")
    print(f"   {Colors.GREEN}python backend_server.py{Colors.RESET}")
    print(f"   Or with demo mode:")
    print(f"   {Colors.GREEN}$env:AUTORL_MODE='demo'; python backend_server.py  # Windows PowerShell{Colors.RESET}")
    print(f"   {Colors.GREEN}AUTORL_MODE=demo python backend_server.py  # Linux/Mac{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}2. Start Frontend (Terminal 2):{Colors.RESET}")
    print(f"   {Colors.GREEN}npm run dev{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}3. (Optional) Start Appium Server (Terminal 3):{Colors.RESET}")
    print(f"   {Colors.GREEN}appium{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}📊 Access Points:{Colors.RESET}")
    print(f"   Frontend:  {Colors.BLUE}http://localhost:5173{Colors.RESET}")
    print(f"   API:       {Colors.BLUE}http://localhost:5000/api/health{Colors.RESET}")
    print(f"   WebSocket: {Colors.BLUE}ws://localhost:5000/ws{Colors.RESET}")
    print(f"   Metrics:   {Colors.BLUE}http://localhost:8000/metrics{Colors.RESET}")
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def main():
    """Main setup verification"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}🔧 AutoRL Setup Verification{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    checks = []
    
    print(f"{Colors.BOLD}System Dependencies:{Colors.RESET}")
    checks.append(check_python_version())
    checks.append(check_node_version())
    checks.append(check_appium())
    checks.append(check_tesseract())
    
    print(f"\n{Colors.BOLD}Configuration Files:{Colors.RESET}")
    checks.append(check_env_file())
    checks.append(check_config_yaml())
    
    print(f"\n{Colors.BOLD}Package Dependencies:{Colors.RESET}")
    checks.append(check_python_packages())
    checks.append(check_node_packages())
    
    check_devices()
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print_status(f"All checks passed! ({passed}/{total})", "success")
        print_startup_commands()
    elif passed >= total - 2:
        print_status(f"Most checks passed ({passed}/{total}). Review warnings above.", "warning")
        print_startup_commands()
    else:
        print_status(f"Setup incomplete ({passed}/{total}). Please fix errors above.", "error")
        print(f"\n{Colors.BOLD}Quick Setup Guide:{Colors.RESET}")
        print(f"  1. Install dependencies: {Colors.GREEN}pip install -r requirements.txt && npm install{Colors.RESET}")
        print(f"  2. Configure environment: {Colors.GREEN}cp .env.example .env{Colors.RESET}")
        print(f"  3. Edit .env with your API keys")
        print(f"  4. Run this script again: {Colors.GREEN}python setup_autorl.py{Colors.RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup verification cancelled{Colors.RESET}")
        sys.exit(1)

