<div align="center">

# 🤖 AutoRL - Intelligent Mobile Automation Platform

### *Autonomous AI-Powered Mobile Automation with Reinforcement Learning*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18.2-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API](#-api-reference) • [Contributing](#-contributing)

---

### *Transform mobile testing and automation with natural language commands powered by AI agents that perceive, plan, execute, and recover from errors automatically*

</div>

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Plugin System](#-plugin-system)
- [AI Agents](#-ai-agents)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Use Cases](#-use-cases)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

**AutoRL** is a next-generation mobile automation platform that combines cutting-edge AI technologies to provide intelligent, adaptive, and self-healing mobile automation. Unlike traditional automation frameworks that follow rigid scripts, AutoRL uses AI agents and reinforcement learning to understand, plan, and execute mobile tasks with human-like intelligence.

### What Makes AutoRL Different?

| Traditional Automation | AutoRL |
|------------------------|---------|
| 🔧 Fixed scripts & selectors | 🧠 Natural language instructions |
| ❌ Breaks on UI changes | ✅ Adapts to changes automatically |
| 📝 Manual error handling | 🔄 Self-healing with AI recovery |
| 📊 Static execution | 📈 Learns and improves over time |
| 🎯 Single-platform | 🌐 Android & iOS support |

### Core Technologies

- **🤖 AI Agents**: Perception, Planning, Execution, and Recovery agents working in harmony
- **🧠 Large Language Models**: GPT-4 powered natural language understanding and action planning
- **🎯 Reinforcement Learning**: Adaptive policies that continuously improve success rates
- **📱 Appium Integration**: Cross-platform mobile automation (Android & iOS)
- **⚡ Real-time Dashboard**: Modern React interface with live WebSocket updates
- **🔌 Plugin Architecture**: Extensible system for custom functionality
- **📊 Metrics & Monitoring**: Prometheus integration for production-grade observability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                           │
│  (Dashboard, Tasks, Devices, Analytics, AI Training)        │
└─────────────────┬───────────────────────────────────────────┘
                  │ WebSocket + REST API
┌─────────────────▼───────────────────────────────────────────┐
│                 Backend API Server                          │
│  (Flask/FastAPI - Task Queue, Metrics, Device Management)   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                  AI Orchestrator                            │
│  ┌──────────────┬────────────┬──────────────┬─────────┐    │
│  │  Perception  │  Planning  │  Execution   │ Recovery│    │
│  │    Agent     │   Agent    │    Agent     │  Agent  │    │
│  └──────────────┴────────────┴──────────────┴─────────┘    │
│                         │                                   │
│              ┌──────────┴──────────┐                       │
│              │  Plugin System      │                       │
│              │  (Vision, OMH, etc) │                       │
│              └─────────────────────┘                       │
└─────────────────┬───────────────────────────────────────────┘
                  │ Appium Protocol
┌─────────────────▼───────────────────────────────────────────┐
│            Mobile Devices (Android/iOS)                     │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│     │  Emulator  │  │ Real Device│  │  Emulator  │        │
│     └────────────┘  └────────────┘  └────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🤖 Intelligent Automation

<details>
<summary><b>Natural Language Task Execution</b></summary>

Simply describe what you want in plain English:
```python
"Open Instagram, navigate to the Explore page, and like the first 3 posts"
```

AutoRL's LLM-powered planning agent understands the intent and generates the action sequence automatically.

</details>

<details>
<summary><b>Visual Perception & OCR</b></summary>

- **Screenshot Analysis**: Capture and analyze UI state in real-time
- **OCR Text Extraction**: Extract text from any UI element using Tesseract
- **Element Detection**: Identify buttons, inputs, and interactive elements
- **Screen State Understanding**: Contextual awareness of app state
- **Vision Boost Plugin**: Enhanced visual recognition with confidence scoring

</details>

<details>
<summary><b>Smart Action Planning</b></summary>

- **LLM-Powered**: GPT-4 generates optimal action sequences
- **Context-Aware**: Considers current screen state and task history
- **Confidence Scoring**: Evaluates plan quality before execution
- **Multi-Step Workflows**: Handles complex, multi-step automations
- **Adaptive Replanning**: Adjusts plans based on execution results

</details>

<details>
<summary><b>Self-Healing Error Recovery</b></summary>

When errors occur, AutoRL doesn't just fail—it recovers:
- **Automatic Detection**: Identifies errors and unexpected states
- **Root Cause Analysis**: LLM analyzes what went wrong
- **Recovery Planning**: Generates intelligent recovery strategies
- **Fallback Strategies**: Multiple recovery paths with priority
- **Learning from Failures**: Stores recovery patterns for future use

Example recovery flow:
```
Error: Button not found
  ↓
Recovery Agent Analyzes
  ↓
Tries Alternative: Scroll down and retry
  ↓
Success: Button found after scroll
  ↓
Stores Recovery Pattern: "scroll_when_element_missing"
```

</details>

<details>
<summary><b>Reinforcement Learning</b></summary>

- **Adaptive Policies**: Learn optimal action strategies over time
- **Experience Buffer**: Stores successful and failed attempts
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Shadow Mode**: Test new policies safely alongside production
- **Policy Promotion**: Hot-swap policies without downtime
- **Performance Tracking**: Monitor policy success rates and metrics

</details>

### 📊 Real-time Monitoring & Analytics

<details>
<summary><b>Live Dashboard</b></summary>

Modern React dashboard with real-time updates via WebSocket:
- **Task Execution Viewer**: Watch tasks execute in real-time
- **Agent Activity Log**: See what each agent is thinking and doing
- **Thought Bubbles**: Visual representation of agent state
- **Device Viewer**: Live device screens with interaction tracking
- **Performance Charts**: Success rates, execution times, trends
- **System Health**: CPU, memory, and task queue metrics

</details>

<details>
<summary><b>Prometheus Metrics</b></summary>

Production-grade observability:
```
autorl_task_success_total          # Total successful tasks
autorl_task_failure_total          # Total failed tasks
autorl_tasks_in_progress           # Currently executing tasks
autorl_avg_task_runtime_seconds    # Average execution time
autorl_device_utilization          # Device usage metrics
autorl_agent_performance           # Per-agent success rates
```

Access at: `http://localhost:8000/metrics`

</details>

<details>
<summary><b>Activity Logging</b></summary>

- **Structured JSON Logs**: Machine-readable audit trails
- **Sensitive Data Masking**: Automatic PII protection
- **Secure Audit Logs**: Critical operations tracking
- **Log Rotation**: Automatic cleanup with configurable retention
- **Multi-level Logging**: DEBUG, INFO, WARNING, ERROR, CRITICAL

</details>

### 🔌 Extensibility & Integration

<details>
<summary><b>Plugin Architecture</b></summary>

Extend AutoRL with custom functionality:

```python
from plugins.base_plugin import BasePlugin

class MyCustomPlugin(BasePlugin):
    def initialize(self, config):
        # Setup code
        self.api_key = config.get('api_key')
    
    def process(self, input_data):
        # Custom logic
        return {"status": "success", "data": processed_data}
    
    def shutdown(self):
        # Cleanup
        pass
```

Drop plugins in `plugins/` directory for automatic discovery.

**Included Plugins:**
- `vision_boost` - Enhanced visual recognition
- `error_recovery` - Advanced error handling
- `memory_cache` - Performance optimization
- `omh_auth_plugin` - OMH authentication
- `omh_maps_plugin` - Location services integration

</details>

<details>
<summary><b>REST & WebSocket APIs</b></summary>

**REST API Endpoints:**
```bash
GET    /api/health              # Health check
GET    /api/devices             # List devices
POST   /api/devices             # Add device
GET    /api/tasks               # Task history
POST   /api/tasks               # Create task
GET    /api/tasks/{id}          # Task details
GET    /api/metrics             # System metrics
GET    /api/policies            # RL policies
POST   /api/policies/promote    # Promote policy
GET    /api/plugins                      # List plugins
POST   /api/plugins/{plugin_name}/execute   # Execute plugin
```

**WebSocket Events:**
```javascript
ws://localhost:5000/ws

Events:
- perception              // UI analysis complete
- planning               // Action plan generated
- execution_start        // Task execution begins
- execution_step         // Individual action executed
- error                  // Error occurred
- recovery_initiated     // Recovery process started
- recovery_succeeded     // Recovery successful
- recovery_failed        // Recovery failed
- completed              // Task completed
- memory_saved           // Learning data stored
```

Full API documentation at: `http://localhost:5000/docs`

</details>

### 🚀 Production-Ready Features

- ✅ **Docker Support**: Containerized deployment with docker-compose
- ✅ **Configuration Management**: YAML-based with environment variable overrides
- ✅ **Demo Mode**: Test without real devices using mock data
- ✅ **Hot Reload**: Development mode with automatic code reload
- ✅ **Error Handling**: Comprehensive try-catch with graceful degradation
- ✅ **Rate Limiting**: Protect APIs from abuse
- ✅ **Security**: API key authentication, encryption, audit logging
- ✅ **Scalability**: Parallel task execution, queue management
- ✅ **Testing**: Comprehensive test suite with pytest
- ✅ **Documentation**: Swagger/OpenAPI, detailed guides

## 📂 Project Structure

```
├── src/
│   ├── components/          # React UI components
│   ├── pages/              # Dashboard pages
│   ├── agent_service/      # AI agent implementations
│   ├── runtime/            # Device and task management
│   ├── llm/                # LLM integration
│   ├── rl/                 # Reinforcement learning
│   ├── perception/         # Visual perception
│   ├── orchestrator.py     # Agent orchestration
│   └── main.py             # Entry point
├── backend_server.py       # Unified backend server
├── master_backend.py       # Master backend with agents
├── start_autorl.py         # Startup script
├── plugins/                # Extensible plugin system
├── agents/                 # Agent registry
├── config.yaml             # Configuration file
└── tests/                  # Test suites
```

## 🚀 Quick Start

Get AutoRL running in under 5 minutes!

### Option 1: One-Command Startup (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/autorl.git
cd autorl

# Install dependencies
pip install -r requirements.txt
npm install

# Start everything with one command
python start_autorl.py
```

**That's it!** Open http://localhost:5173 to see the dashboard.

### Option 2: Docker (Zero Setup)

```bash
# Clone and start with Docker
git clone https://github.com/YOUR_USERNAME/autorl.git
cd autorl

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

Access at: http://localhost:5173

### Option 3: Manual Setup (Full Control)

<details>
<summary>Click to expand manual setup instructions</summary>

**Terminal 1 - Backend:**
```bash
python master_backend.py
# Or use backend_server.py for simpler setup
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

**Terminal 3 - Appium (optional, for real devices):**
```bash
appium
```

</details>

### First Steps After Installation

1. **Open Dashboard**: Navigate to http://localhost:5173
2. **Check Connection**: Look for green "WebSocket Connected" badge
3. **Create Your First Task**:
   ```
   Task: "Open Instagram and navigate to Explore page"
   Device: Select any (or leave empty for demo mode)
   Click: Start Task
   ```
4. **Watch the Magic**: See real-time updates as AI agents execute your task

### Demo Mode (No Devices Required)

AutoRL starts in **demo mode** by default, which simulates everything:
- ✅ No physical devices needed
- ✅ No Appium installation required
- ✅ Realistic AI execution simulation
- ✅ WebSocket real-time updates
- ✅ Perfect for development and testing

**Switch to production mode:**
```bash
# In config.yaml
server:
  mode: "production"

# Or via environment variable
export AUTORL_MODE=production
python master_backend.py
```

## 💿 Installation

### Prerequisites

| Component | Version | Required | Purpose |
|-----------|---------|----------|---------|
| Python | 3.9+ | ✅ Yes | Backend runtime |
| Node.js | 16+ | ✅ Yes | Frontend build |
| Appium | 2.0+ | ⚠️ Production only | Device automation |
| Android SDK | Latest | ⚠️ Android only | Emulator support |
| Xcode | Latest | ⚠️ iOS only | iOS simulator |
| Docker | 20+ | ❌ Optional | Container deployment |

### Detailed Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/autorl.git
cd autorl
```

#### 2. Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `appium-python-client` - Mobile automation
- `openai` - LLM integration
- `Pillow` & `pytesseract` - Image processing & OCR
- `prometheus-client` - Metrics
- `websockets` - Real-time communication

#### 3. Node.js Dependencies

```bash
npm install
```

**Core Dependencies:**
- `react` & `react-dom` - UI framework
- `vite` - Build tool
- `tailwindcss` - Styling
- `lucide-react` - Icons

#### 4. Additional Setup (Optional)

<details>
<summary><b>Tesseract OCR (for text extraction)</b></summary>

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

</details>

<details>
<summary><b>Appium (for real device automation)</b></summary>

```bash
# Install Appium
npm install -g appium

# Install drivers
appium driver install uiautomator2  # Android
appium driver install xcuitest       # iOS

# Verify installation
appium --version
```

</details>

<details>
<summary><b>Android SDK Setup</b></summary>

1. Install Android Studio
2. Open SDK Manager
3. Install:
   - Android SDK Platform Tools
   - Android Emulator
   - System Images (Android 11+)
4. Add to PATH:
   ```bash
   export ANDROID_HOME=$HOME/Android/Sdk
   export PATH=$PATH:$ANDROID_HOME/emulator
   export PATH=$PATH:$ANDROID_HOME/platform-tools
   ```

</details>

#### 5. Verify Installation

```bash
# Check Python installation
python --version
pip list | grep fastapi

# Check Node.js installation
node --version
npm list react

# Test backend
python -c "import fastapi; print('Backend OK')"

# Test frontend build
npm run build
```

## ⚙️ Configuration

AutoRL uses a powerful YAML-based configuration system with environment variable overrides.

### config.yaml

The main configuration file (`config.yaml`) controls all aspects of the system:

<details>
<summary><b>Server Configuration</b></summary>

```yaml
server:
  host: "0.0.0.0"              # Bind address
  port: 5000                    # API port
  mode: "production"            # "demo" or "production"
  debug: false                  # Debug mode
  cors_origins: ["*"]           # CORS allowed origins
```

</details>

<details>
<summary><b>API Configuration</b></summary>

```yaml
api:
  base_url: "http://localhost:5000/api"
  websocket_url: "ws://localhost:5000/ws"
  metrics_port: 8000
  timeout: 300                  # Request timeout (seconds)
```

</details>

<details>
<summary><b>Device Configuration</b></summary>

```yaml
devices:
  max_parallel: 4               # Max simultaneous devices
  default_platform: "android"   # Default platform
  auto_discover: true           # Auto-discover connected devices
  
  preregistered:
    - device_id: "emulator-5554"
      platform: "android"
      is_real: false
    - device_id: "iPhone 15"
      platform: "ios"
      is_real: false
```

Add devices programmatically:
```python
from src.runtime.device_manager import DeviceManager, Device

device_manager = DeviceManager()
device_manager.add_device(
    Device("Pixel_7", "android", is_real=True)
)
```

</details>

<details>
<summary><b>LLM Configuration</b></summary>

```yaml
llm:
  provider: "openai"            # "openai", "anthropic", "local"
  model: "gpt-4"                # Model name
  temperature: 0.7              # Creativity (0-1)
  max_tokens: 2000              # Response length
  api_key_env: "OPENAI_API_KEY" # Environment variable name
  fallback_enabled: true        # Use fallback on failure
```

</details>

<details>
<summary><b>Reinforcement Learning</b></summary>

```yaml
reinforcement_learning:
  initial_policy: "initial_policy"
  shadow_mode_enabled: true     # Test policies safely
  learning_rate: 0.001
  epsilon_start: 0.1            # Initial exploration
  epsilon_min: 0.01             # Minimum exploration
  epsilon_decay: 0.995          # Decay rate
  discount_factor: 0.99         # Future reward weight
  experience_buffer_size: 10000 # Memory size
```

</details>

<details>
<summary><b>Plugin Configuration</b></summary>

```yaml
plugins:
  directory: "./plugins"
  auto_load: true
  enabled_plugins:
    - vision_boost
    - error_recovery
    - memory_cache
    - omh_auth_plugin
  
  # Plugin-specific settings
  vision_boost:
    confidence_threshold: 0.85
    max_retries: 3
  
  error_recovery:
    max_recovery_attempts: 3
    recovery_timeout: 30
```

</details>

### Environment Variables

Create a `.env` file in the project root:

```bash
# ============================================
# API Keys & Secrets
# ============================================
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-key-here

# ============================================
# Appium Configuration
# ============================================
APPIUM_SERVER_URL=http://localhost:4723/wd/hub

# ============================================
# Application Mode
# ============================================
AUTORL_MODE=demo                # "demo" or "production"

# ============================================
# Database (Optional)
# ============================================
DB_PASSWORD=your-db-password
REDIS_URL=redis://localhost:6379

# ============================================
# Security (Production)
# ============================================
API_KEY=your-api-key
SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# ============================================
# Monitoring (Optional)
# ============================================
PROMETHEUS_PORT=8000
SENTRY_DSN=your-sentry-dsn

# ============================================
# Frontend
# ============================================
VITE_API_BASE_URL=http://localhost:5000/api
VITE_WS_URL=ws://localhost:5000/ws
```

### Configuration Priority

Configuration values are loaded in this order (later overrides earlier):

1. **Default values** in code
2. **config.yaml** file
3. **Environment variables**
4. **Command-line arguments** (if applicable)

Example:
```bash
# Override mode via environment variable
export AUTORL_MODE=production
python backend_server.py

# Or pass directly
AUTORL_MODE=production python backend_server.py
```

## 📖 Usage Examples

### Example 1: Basic Task Execution

```python
import requests

# Create a task via API
response = requests.post('http://localhost:5000/api/tasks', json={
    "instruction": "Open Twitter and like the first post",
    "device_id": "emulator-5554",
    "parameters": {
        "enable_learning": True,
        "timeout": 120
    }
})

task_id = response.json()['task_id']
print(f"Task created: {task_id}")
```

**What happens:**
1. Perception Agent captures screenshot
2. Planning Agent generates action plan
3. Execution Agent performs actions
4. Recovery Agent handles any errors
5. Results are stored for learning

### Example 2: Real-time Monitoring with WebSocket

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:5000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.event) {
    case 'perception':
      console.log('UI analyzed:', data.data);
      break;
    case 'planning':
      console.log('Action plan:', data.data.actions);
      break;
    case 'execution_step':
      console.log('Executing:', data.data.action);
      break;
    case 'completed':
      console.log('Task completed!', data.data.result);
      break;
    case 'error':
      console.log('Error occurred:', data.data.error);
      break;
  }
};
```

### Example 3: Custom Plugin Development

```python
# plugins/custom_analytics.py
from plugins.base_plugin import BasePlugin
import logging

class CustomAnalyticsPlugin(BasePlugin):
    """Custom analytics plugin for tracking user behavior"""
    
    def initialize(self, config):
        self.logger = logging.getLogger(__name__)
        self.analytics_endpoint = config.get('endpoint')
        self.logger.info("Analytics plugin initialized")
    
    def process(self, input_data):
        """Process analytics data"""
        event_type = input_data.get('type')
        event_data = input_data.get('data')
        
        # Send to analytics service
        self._send_analytics(event_type, event_data)
        
        return {
            "status": "success",
            "events_sent": 1
        }
    
    def _send_analytics(self, event_type, data):
        # Your analytics logic here
        pass
    
    def shutdown(self):
        self.logger.info("Analytics plugin shutdown")
```

**Usage:**
```python
# Execute plugin via API
response = requests.post('http://localhost:5000/api/plugins/custom_analytics/execute', json={
    "type": "task_completed",
    "data": {"task_id": "123", "duration": 45.2}
})
```

### Example 4: Advanced Task with Recovery

```python
# Complex task with custom error handling
task_config = {
    "instruction": """
        1. Open the weather app
        2. Search for 'New York'
        3. Take a screenshot of the 7-day forecast
        4. Navigate to settings and enable notifications
    """,
    "device_id": "Pixel_7",
    "parameters": {
        "enable_learning": True,
        "max_retries": 3,
        "recovery_strategy": "adaptive",  # or "aggressive", "conservative"
        "screenshot_on_error": True,
        "timeout": 180
    }
}

response = requests.post('http://localhost:5000/api/tasks', json=task_config)
```

### Example 5: Parallel Device Execution

```python
import asyncio
import httpx

async def execute_on_device(device_id, instruction):
    async with httpx.AsyncClient() as client:
        response = await client.post('http://localhost:5000/api/tasks', json={
            "instruction": instruction,
            "device_id": device_id
        })
        return response.json()

# Execute same task on multiple devices simultaneously
devices = ["emulator-5554", "Pixel_7", "iPhone_15"]
instruction = "Open Instagram and check notifications"

results = await asyncio.gather(*[
    execute_on_device(device, instruction) 
    for device in devices
])

print(f"Started {len(results)} tasks")
```

### Example 6: RL Policy Management

```python
# Get current policies
response = requests.get('http://localhost:5000/api/policies')
policies = response.json()

# Promote shadow policy to active
requests.post('http://localhost:5000/api/policies/promote', json={
    "policy_id": "shadow_policy_v2",
    "strategy": "gradual"  # or "immediate"
})

# Monitor policy performance
metrics = requests.get('http://localhost:5000/api/metrics').json()
print(f"Policy success rate: {metrics['policy_success_rate']}%")
```

### Example 7: Dashboard Integration

```jsx
// React component using AutoRL
import { useState, useEffect } from 'react';
import { apiService } from './services/api';
import { useWebSocket } from './hooks/useWebSocket';

function TaskExecutor() {
  const [taskStatus, setTaskStatus] = useState(null);
  const { lastMessage } = useWebSocket((message) => {
    if (message.event === 'execution_step') {
      setTaskStatus(message.data);
    }
  });
  
  const executeTask = async (instruction) => {
    const response = await apiService.createTask(instruction);
    console.log('Task started:', response.task_id);
  };
  
  return (
    <div>
      <button onClick={() => executeTask('Open Settings')}>
        Execute Task
      </button>
      {taskStatus && (
        <div>Status: {taskStatus.action}</div>
      )}
    </div>
  );
}
```

### Example 8: Batch Task Processing

```python
# Process multiple tasks in sequence
tasks = [
    "Open Gmail and archive first 5 emails",
    "Open Calendar and check tomorrow's events",
    "Open Notes and create new note 'Meeting Summary'",
    "Open Photos and upload latest screenshot"
]

for task in tasks:
    response = requests.post('http://localhost:5000/api/tasks', json={
        "instruction": task,
        "parameters": {"enable_learning": True}
    })
    task_id = response.json()['task_id']
    print(f"Queued: {task_id}")
    
    # Wait for completion (in real app, use WebSocket)
    # ... check status endpoint ...
```

## 📊 Key Components

### Frontend Dashboard

Modern React SPA with Tailwind CSS:

| Page | Description |
|------|-------------|
| **Dashboard** | Real-time task monitoring with WebSocket updates |
| **Tasks** | Create, manage, and view task history |
| **Devices** | Device connection and status management |
| **Analytics** | Performance charts and success rate tracking |
| **AI Training** | RL policy management and A/B testing |
| **Plugins** | Plugin marketplace and configuration |
| **Settings** | System configuration and preferences |

**Key Features:**
- 🔴 Live WebSocket connection indicator
- 📊 Real-time metrics and charts
- 🎨 Thought bubble visualization
- 📱 Device screen viewer
- 📝 Scrollable agent activity log
- 🎯 Task builder with drag-and-drop

## 🔌 API Reference

### REST API Endpoints

Full interactive documentation available at: http://localhost:5000/docs

#### Health & Status

```bash
GET /api/health
# Returns: { "status": "ok", "version": "1.0.0", "mode": "demo" }

GET /api/status
# Returns: { "tasks_running": 2, "devices_connected": 3, "uptime": 3600 }
```

#### Device Management

```bash
GET /api/devices
# List all connected devices

POST /api/devices
# Add a new device
Body: { "device_id": "emulator-5554", "platform": "android", "is_real": false }

GET /api/devices/{device_id}
# Get specific device details

DELETE /api/devices/{device_id}
# Remove a device
```

#### Task Management

```bash
GET /api/tasks
# List all tasks (with pagination)
Query params: ?limit=50&offset=0&status=completed

POST /api/tasks
# Create a new task
Body: {
  "instruction": "Open Instagram",
  "device_id": "emulator-5554",  # Optional (null for demo)
  "parameters": {
    "enable_learning": true,
    "timeout": 300,
    "max_retries": 3
  }
}
Response: { "task_id": "uuid", "status": "queued" }

GET /api/tasks/{task_id}
# Get task status and results
Response: {
  "task_id": "uuid",
  "status": "completed",
  "result": {...},
  "duration": 45.2,
  "success": true
}

DELETE /api/tasks/{task_id}
# Cancel a running task
```

#### Metrics & Analytics

```bash
GET /api/metrics
# Get system metrics
Response: {
  "tasks_total": 150,
  "tasks_success": 142,
  "tasks_failed": 8,
  "success_rate": 94.67,
  "avg_execution_time": 32.5,
  "devices_active": 3,
  "policy_performance": {...}
}

GET /api/metrics/prometheus
# Prometheus-formatted metrics
```

#### RL Policy Management

```bash
GET /api/policies
# List all RL policies
Response: {
  "active": { "id": "policy_v1", "success_rate": 94.5 },
  "shadow": { "id": "policy_v2", "success_rate": 96.2 },
  "archived": [...]
}

POST /api/policies/promote
# Promote shadow policy to active
Body: { "policy_id": "policy_v2", "strategy": "gradual" }

POST /api/policies/upload
# Upload new policy
Body: (multipart/form-data with policy file)
```

#### Plugin Management

```bash
GET /api/plugins
# List all available plugins
Response: [
  {
    "id": "vision_boost",
    "name": "Vision Boost",
    "version": "1.0.0",
    "enabled": true,
    "config": {...}
  }
]

POST /api/plugins/{plugin_name}/execute
# Execute a plugin
Body: { ...input_data... }
Response: { "status": "success", "result": {...} }
```

### WebSocket API

Connect to `ws://localhost:5000/ws` for real-time updates.

#### Event Types

| Event | Description | Data Schema |
|-------|-------------|-------------|
| `connected` | WebSocket connection established | `{ "client_id": "uuid" }` |
| `perception` | UI analysis completed | `{ "screenshot": "base64", "elements": [...], "text": "..." }` |
| `planning` | Action plan generated | `{ "actions": [...], "confidence": 0.95 }` |
| `execution_start` | Task execution begins | `{ "task_id": "uuid", "device": "..." }` |
| `execution_step` | Individual action executed | `{ "action": "tap", "target": "button", "success": true }` |
| `error` | Error occurred | `{ "error_type": "ElementNotFound", "message": "...", "screenshot": "..." }` |
| `recovery_initiated` | Recovery process started | `{ "strategy": "scroll_and_retry", "attempts": 1 }` |
| `recovery_succeeded` | Recovery successful | `{ "recovery_time": 5.2, "strategy_used": "..." }` |
| `recovery_failed` | Recovery failed | `{ "final_error": "...", "attempts": 3 }` |
| `completed` | Task completed | `{ "task_id": "uuid", "success": true, "duration": 45.2, "result": {...} }` |
| `memory_saved` | Learning data stored | `{ "experience_id": "uuid", "reward": 1.0 }` |
| `metrics_update` | Metrics updated | `{ "success_rate": 94.5, "tasks_completed": 150 }` |

#### Usage Example

```javascript
const ws = new WebSocket('ws://localhost:5000/ws');

// Connection opened
ws.onopen = () => {
  console.log('Connected to AutoRL');
};

// Receive messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log(`[${message.event}]`, message.data);
  
  // Handle specific events
  if (message.event === 'error') {
    handleError(message.data);
  }
};

// Connection closed
ws.onclose = () => {
  console.log('Disconnected, attempting reconnect...');
  // Implement reconnection logic
};

// Send messages (if needed)
ws.send(JSON.stringify({
  event: 'subscribe',
  data: { task_id: 'uuid' }
}));
```

## 🤖 AI Agents Deep Dive

AutoRL's intelligence comes from a coordinated system of specialized AI agents.

### 1. Perception Agent 👁️

**Role:** Understand the current state of the mobile device UI.

**Capabilities:**
- **Screenshot Capture**: High-quality device screen capture
- **OCR Text Extraction**: Extract all visible text using Tesseract
- **Element Detection**: Identify buttons, inputs, images, and interactive elements
- **UI Hierarchy Analysis**: Parse accessibility tree for element relationships
- **State Recognition**: Detect app state, screens, and navigation context
- **Vision Boost**: Enhanced visual recognition with confidence scoring

**Process Flow:**
```
1. Capture screenshot → 2. Run OCR → 3. Analyze elements → 
4. Build UI state → 5. Return perception data
```

**Output Example:**
```json
{
  "screen_state": "home_feed",
  "detected_text": ["Welcome", "Notifications", "Profile"],
  "elements": [
    {"type": "button", "text": "Like", "bounds": [100, 200, 180, 250]},
    {"type": "input", "text": "", "bounds": [50, 300, 300, 350]}
  ],
  "confidence": 0.92
}
```

### 2. Planning Agent 🧠

**Role:** Generate optimal action sequences to achieve task goals.

**Capabilities:**
- **Natural Language Understanding**: Parse complex task descriptions
- **LLM-Powered Planning**: Use GPT-4 to generate action plans
- **Context-Aware**: Consider current screen state and task history
- **Multi-Step Workflows**: Handle complex, multi-step automations
- **Confidence Scoring**: Evaluate plan quality before execution
- **Adaptive Replanning**: Adjust plans based on execution feedback

**Process Flow:**
```
1. Receive task instruction → 2. Analyze perception data → 
3. Query LLM for plan → 4. Validate feasibility → 
5. Return action sequence
```

**Example Plan:**
```json
{
  "task": "Open Instagram and like first post",
  "actions": [
    {"type": "tap", "target": "Instagram app icon", "timeout": 5},
    {"type": "wait", "duration": 2},
    {"type": "tap", "target": "first post like button", "timeout": 3},
    {"type": "verify", "condition": "button turned red"}
  ],
  "confidence": 0.95,
  "estimated_duration": 15
}
```

### 3. Execution Agent ⚡

**Role:** Execute planned actions on the mobile device.

**Capabilities:**
- **Action Execution**: Perform taps, swipes, inputs, etc.
- **Retry Logic**: Automatic retries with exponential backoff
- **Element Interaction**: Find and interact with UI elements
- **Timing Management**: Smart waits and timeouts
- **State Verification**: Confirm actions succeeded
- **Screenshot on Error**: Capture evidence of failures

**Supported Actions:**
- `tap` - Tap on element or coordinates
- `swipe` - Swipe in direction
- `input` - Type text into fields
- `press_back` - Press back button
- `wait` - Wait for duration or condition
- `verify` - Verify expected state
- `scroll` - Scroll to element

**Process Flow:**
```
1. Receive action → 2. Find target element → 
3. Execute action → 4. Verify result → 
5. Retry if failed → 6. Report status
```

### 4. Recovery Agent 🔄

**Role:** Detect and recover from errors automatically.

**Capabilities:**
- **Error Detection**: Identify when things go wrong
- **Root Cause Analysis**: LLM analyzes error context
- **Recovery Planning**: Generate intelligent recovery strategies
- **Multiple Strategies**: Try different approaches
- **Learning from Failures**: Store successful recoveries
- **Graceful Degradation**: Fall back to safe states

**Recovery Strategies:**
1. **Retry**: Simple retry with delay
2. **Scroll and Retry**: Scroll to find hidden elements
3. **Alternative Path**: Try different UI elements
4. **App Restart**: Close and reopen app
5. **Device Reset**: Return to home screen
6. **Abort**: Give up gracefully

**Process Flow:**
```
Error Detected → Analyze Context → Generate Recovery Plans → 
Try Strategy 1 → Failed? → Try Strategy 2 → Success? → Continue Task
```

**Example Recovery:**
```json
{
  "original_error": "ElementNotFound: 'Like button'",
  "analysis": "Element likely below fold",
  "recovery_strategy": "scroll_down_and_retry",
  "attempts": 2,
  "success": true,
  "time_to_recover": 5.3
}
```

### Agent Coordination

All agents work together in a coordinated workflow:

```
┌─────────────────────────────────────────────┐
│           Task Instruction                  │
│   "Open Instagram and like first post"     │
└─────────────────┬───────────────────────────┘
                  │
          ┌───────▼────────┐
          │  PERCEPTION    │ ← Screenshots, OCR, Elements
          │     AGENT      │
          └───────┬────────┘
                  │ UI State
          ┌───────▼────────┐
          │   PLANNING     │ ← LLM, Action Planning
          │     AGENT      │
          └───────┬────────┘
                  │ Action Plan
          ┌───────▼────────┐
          │  EXECUTION     │ ← Device Automation
          │     AGENT      │
          └───────┬────────┘
                  │
         Success? │ Error?
                  │
          ┌───────▼────────┐
          │   RECOVERY     │ ← Error Handling
          │     AGENT      │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │    RESULT      │
          └────────────────┘
```

### Orchestrator

The **Orchestrator** (`src/orchestrator.py`) coordinates all agents:

```python
# Simplified workflow
async def execute_task_with_recovery(
    task_description: str,
    device_id: str,
    timeout_seconds: int = 300
) -> TaskState:
    # 1. Perception
    ui_state = await perception_agent.analyze_screen(device_id)
    
    # 2. Planning
    action_plan = await planning_agent.create_plan(task_description, ui_state)
    
    # 3. Execution
    try:
        for action in action_plan['actions']:
            success = await execution_agent.execute_action_with_retry(action, device_id)
            if not success:
                raise Exception(f"Action failed: {action}")
        return TaskState(task_id=task_id, status="completed", progress=100.0)
    except Exception as error:
        # 4. Recovery
        await recovery_agent.handle_error(task_state, error)
        # Retry logic here
        return task_state
```

## 🔌 Plugin System

AutoRL's plugin system allows easy extension of functionality without modifying core code.

### Creating a Plugin

```python
# plugins/my_custom_plugin.py
from plugins.base_plugin import BasePlugin
import logging

class MyCustomPlugin(BasePlugin):
    """Custom plugin description"""
    
    def initialize(self, config):
        """Called once when plugin is loaded"""
        self.logger = logging.getLogger(__name__)
        self.api_key = config.get('api_key')
        self.enabled = config.get('enabled', True)
        self.logger.info(f"{self.name} initialized")
    
    def process(self, input_data):
        """Main plugin logic"""
        try:
            # Your custom logic here
            result = self._do_something(input_data)
            return {
                "status": "success",
                "data": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def shutdown(self):
        """Called when plugin is unloaded"""
        self.logger.info(f"{self.name} shutdown")
```

### Plugin Configuration

Add plugin config to `config.yaml`:

```yaml
plugins:
  enabled_plugins:
    - my_custom_plugin
  
  my_custom_plugin:
    api_key: "your-key"
    enabled: true
    custom_setting: "value"
```

### Plugin Discovery

Plugins are automatically discovered from the `plugins/` directory. AutoRL scans for classes that inherit from `BasePlugin`.

### Included Plugins

| Plugin | Purpose | Configuration |
|--------|---------|---------------|
| `vision_boost` | Enhanced visual recognition | `confidence_threshold`, `max_retries` |
| `error_recovery` | Advanced error handling | `max_attempts`, `timeout` |
| `memory_cache` | Performance optimization | `max_entries`, `ttl` |
| `omh_auth_plugin` | OMH authentication | `auth_endpoint`, `client_id` |
| `omh_maps_plugin` | Location services | `api_key`, `default_location` |

## 📈 Metrics & Monitoring

### Prometheus Metrics

Available at `http://localhost:8000/metrics`

**Task Metrics:**
- `autorl_task_success_total` - Total successful tasks
- `autorl_task_failure_total` - Total failed tasks
- `autorl_tasks_in_progress` - Currently running tasks
- `autorl_avg_task_runtime_seconds` - Average execution time
- `autorl_task_queue_size` - Tasks waiting in queue

**Device Metrics:**
- `autorl_device_utilization` - Device usage percentage
- `autorl_devices_connected` - Number of connected devices
- `autorl_device_errors_total` - Device connection errors

**Agent Metrics:**
- `autorl_agent_performance` - Per-agent success rates
- `autorl_perception_time_seconds` - Time spent in perception
- `autorl_planning_time_seconds` - Time spent in planning
- `autorl_execution_time_seconds` - Time spent executing

**RL Metrics:**
- `autorl_policy_success_rate` - Current policy success rate
- `autorl_epsilon_value` - Current exploration rate
- `autorl_experience_buffer_size` - Stored experiences

### Grafana Dashboard

Import our pre-built dashboard:

```bash
# Start Grafana with Docker
docker run -d -p 3000:3000 grafana/grafana

# Import dashboard JSON from deployment/grafana-dashboard.json
```

### Activity Logging

Logs are written to `logs/autorl.log` with rotation:

```json
{
  "timestamp": "2024-10-11T10:30:45Z",
  "level": "INFO",
  "event": "task_completed",
  "task_id": "abc123",
  "device_id": "emulator-5554",
  "duration": 45.2,
  "success": true
}
```

**Features:**
- Structured JSON format
- Automatic PII masking
- Log rotation (10MB max, 5 backups)
- Secure audit trail for critical operations

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/test_error_handling_basic.py

# Run specific test
pytest tests/test_agents.py::test_perception_agent

# Run with verbose output
pytest -v tests/
```

### Integration Tests

```bash
# Start mock server first
cd autorl-demo/backend
python app.py

# In another terminal, run integration tests
pytest tests/test_automation_suite.py
pytest tests/test_integration.py -v
```

### Test Categories

| Category | Command | Description |
|----------|---------|-------------|
| Unit | `pytest tests/test_*.py` | Test individual components |
| Integration | `pytest tests/test_integration.py` | Test component interactions |
| E2E | `pytest tests/test_e2e.py` | Test full workflows |
| Performance | `pytest tests/test_performance.py` | Test speed and resource usage |

### Writing Tests

```python
# tests/test_custom.py
import pytest
from src.orchestrator import Orchestrator

@pytest.mark.asyncio
async def test_task_execution():
    """Test basic task execution"""
    orchestrator = Orchestrator()
    
    result = await orchestrator.execute_task_with_recovery(
        instruction="Open Settings",
        device_id=None,  # Demo mode
        timeout=60
    )
    
    assert result["success"] == True
    assert "result" in result

@pytest.fixture
def mock_device():
    """Fixture for mock device"""
    return {
        "device_id": "test-device",
        "platform": "android",
        "is_real": False
    }
```

### Test Coverage

Aim for:
- ✅ 80%+ overall coverage
- ✅ 90%+ for critical paths
- ✅ 100% for error handling

View coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 🚢 Deployment

### Docker Deployment (Recommended)

**Single Command:**
```bash
docker-compose up -d
```

**Manual Steps:**
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f autorl-backend
docker-compose logs -f autorl-frontend

# Scale services
docker-compose up -d --scale autorl-backend=3

# Stop services
docker-compose down

# Clean up
docker-compose down -v  # Removes volumes too
```

### Docker Compose Services

```yaml
version: '3.8'
services:
  autorl-backend:
    build: .
    ports:
      - "5000:5000"
      - "8000:8000"
    environment:
      - AUTORL_MODE=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./screenshots:/app/screenshots
  
  autorl-frontend:
    build: ./autorl-frontend
    ports:
      - "80:80"
    depends_on:
      - autorl-backend
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

### Production Deployment

#### 1. Cloud Provider Setup

**AWS:**
```bash
# Use ECS/EKS for container orchestration
# Store secrets in AWS Secrets Manager
# Use Application Load Balancer

aws ecr create-repository --repository-name autorl
docker tag autorl:latest ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/autorl:latest
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/autorl:latest
```

**Google Cloud:**
```bash
# Use Cloud Run or GKE
gcloud builds submit --tag gcr.io/${PROJECT_ID}/autorl
gcloud run deploy autorl --image gcr.io/${PROJECT_ID}/autorl --platform managed
```

**Azure:**
```bash
# Use Azure Container Instances or AKS
az acr build --registry ${ACR_NAME} --image autorl:latest .
az container create --resource-group ${RG} --name autorl --image ${ACR_NAME}.azurecr.io/autorl:latest
```

#### 2. Environment Configuration

```bash
# Production .env file
AUTORL_MODE=production
OPENAI_API_KEY=sk-prod-key
API_KEY=your-secure-api-key
SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key

# Database
DB_HOST=prod-db.example.com
DB_PASSWORD=secure-password

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_PORT=8000

# Device Farm (optional)
BROWSERSTACK_USERNAME=your-username
BROWSERSTACK_ACCESS_KEY=your-key
```

#### 3. Security Checklist

- ✅ Enable API key authentication
- ✅ Use HTTPS/TLS for all communications
- ✅ Store secrets in environment variables or secret manager
- ✅ Enable rate limiting
- ✅ Set up WAF (Web Application Firewall)
- ✅ Enable audit logging
- ✅ Implement IP whitelisting if needed
- ✅ Regular security updates

#### 4. Monitoring & Alerts

```yaml
# Prometheus alert rules
groups:
  - name: autorl_alerts
    rules:
      - alert: HighTaskFailureRate
        expr: autorl_task_failure_total / autorl_task_success_total > 0.1
        for: 5m
        annotations:
          summary: "Task failure rate above 10%"
      
      - alert: NoDevicesConnected
        expr: autorl_devices_connected == 0
        for: 2m
        annotations:
          summary: "No devices connected"
```

#### 5. Scaling Considerations

**Horizontal Scaling:**
- Multiple backend instances behind load balancer
- Shared Redis for caching
- Shared database for state

**Vertical Scaling:**
- Increase CPU/RAM for LLM operations
- GPU instances for vision processing

**Device Farm:**
- Use cloud device providers (BrowserStack, Sauce Labs)
- Or manage your own device farm

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autorl-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autorl
  template:
    metadata:
      labels:
        app: autorl
    spec:
      containers:
      - name: autorl
        image: autorl:latest
        ports:
        - containerPort: 5000
        env:
        - name: AUTORL_MODE
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: autorl-secrets
              key: openai-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: autorl-service
spec:
  selector:
    app: autorl
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/
kubectl get pods
kubectl logs -f deployment/autorl-backend
```

### Health Checks

Configure health checks for load balancers:

```
HTTP GET /api/health
Expected: 200 OK
Response: {"status": "ok", "version": "1.0.0"}
```

### Backup Strategy

1. **Database backups**: Daily automated backups
2. **RL Policy backups**: After each promotion
3. **Logs**: Store in durable storage (S3, GCS)
4. **Configuration**: Version control in Git

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>WebSocket won't connect</b></summary>

**Symptoms:** Red "Disconnected" badge in dashboard

**Solutions:**
1. Check backend is running: `curl http://localhost:5000/api/health`
2. Verify WebSocket URL in `config.yaml` or `.env`
3. Check firewall isn't blocking port 5000
4. Look for CORS errors in browser console
5. Try different browser

```bash
# Check backend logs
python master_backend.py
# Look for "WebSocket connection opened" messages
```

</details>

<details>
<summary><b>Tasks not executing</b></summary>

**Symptoms:** Tasks stuck in "queued" status

**Solutions:**
1. Check if in demo mode: `config.yaml` → `server.mode: "demo"`
2. Verify devices are connected: `GET /api/devices`
3. Check backend logs for errors
4. Ensure Appium is running (production mode): `appium --version`
5. Verify device IDs are correct

```bash
# List connected Android devices
adb devices

# List iOS simulators
xcrun simctl list devices
```

</details>

<details>
<summary><b>LLM errors / OpenAI API issues</b></summary>

**Symptoms:** Planning agent fails, API errors

**Solutions:**
1. Verify API key: `echo $OPENAI_API_KEY`
2. Check API key is valid and has credits
3. Review rate limits
4. Enable fallback: `config.yaml` → `llm.fallback_enabled: true`
5. Try different model: `llm.model: "gpt-3.5-turbo"`

```bash
# Test OpenAI API directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

</details>

<details>
<summary><b>High memory usage</b></summary>

**Solutions:**
1. Reduce experience buffer: `config.yaml` → `reinforcement_learning.experience_buffer_size: 5000`
2. Enable memory cache cleanup: `plugins.memory_cache.ttl: 1800`
3. Limit parallel tasks: `task_execution.max_parallel_tasks: 2`
4. Increase Docker memory limit
5. Monitor with: `GET /api/metrics`

</details>

<details>
<summary><b>Appium connection issues</b></summary>

**Symptoms:** "Could not connect to Appium server"

**Solutions:**
1. Start Appium: `appium`
2. Verify URL: `config.yaml` → `appium.server_url`
3. Check Appium drivers installed:
   ```bash
   appium driver list --installed
   ```
4. Verify device is connected and authorized
5. Check Appium logs for errors

</details>

### Debug Mode

Enable detailed logging:

```bash
# In config.yaml
logging:
  level: "DEBUG"

# Or via environment
export LOG_LEVEL=DEBUG
python master_backend.py
```

### Getting Help

1. **Check logs**: `logs/autorl.log`
2. **API documentation**: http://localhost:5000/docs
3. **GitHub Issues**: Search existing issues or create new one
4. **Community**: Join our Discord/Slack (link in repo)

## 🎯 Use Cases

### 1. Automated Mobile Testing

Replace brittle test scripts with intelligent AI agents:

```python
test_scenarios = [
    "Login with valid credentials",
    "Add item to cart and checkout",
    "Apply discount code and verify price",
    "Navigate to profile and update email",
    "Test password reset flow"
]

for scenario in test_scenarios:
    result = await orchestrator.execute_task_with_recovery(scenario, device_id="auto")
    assert result.status == "completed", f"Failed: {scenario}"
```

**Benefits:**
- Self-healing tests (adapts to UI changes)
- Natural language test cases
- Automatic error recovery
- Detailed execution logs

### 2. User Journey Validation

Test complete user flows across multiple screens:

```python
user_journey = """
1. Open the banking app
2. Login with credentials
3. Navigate to transfers
4. Send $50 to John Doe
5. Verify transaction appears in history
6. Take screenshot of confirmation
"""

result = await orchestrator.execute_task_with_recovery(user_journey, device_id="Pixel_7")
```

### 3. Competitive Analysis

Monitor competitor apps automatically:

```python
competitor_tasks = [
    "Open Uber app and check pricing for JFK to Manhattan",
    "Open Lyft app and check same route pricing",
    "Compare features and take screenshots",
    "Monitor promotional offers"
]
```

### 4. Performance & Load Testing

Simulate real user behavior:

```python
# Simulate 100 concurrent users
async def simulate_user():
    await execute_task("Browse products and add to cart")
    await asyncio.sleep(random.randint(5, 15))
    await execute_task("Complete checkout")

tasks = [simulate_user() for _ in range(100)]
await asyncio.gather(*tasks)
```

### 5. Accessibility Testing

Validate screen reader compatibility:

```python
accessibility_tests = [
    "Navigate entire app using TalkBack",
    "Verify all buttons have labels",
    "Test high contrast mode",
    "Check text scaling support"
]
```

### 6. Data Collection & Analytics

Gather usage patterns:

```python
data_collection = """
Track user flow through onboarding:
1. Open app
2. Complete tutorial
3. Create account
4. Record time spent on each screen
5. Log all interactions
"""
```

## 🗺️ Roadmap

### Q1 2025
- [ ] **Multi-modal LLM Support**: Claude, Gemini, local models
- [ ] **Advanced Vision**: GPT-4 Vision integration
- [ ] **Mobile-specific RL**: Custom algorithms for mobile automation
- [ ] **Plugin Marketplace**: Share and discover plugins
- [ ] **Cloud Device Farms**: BrowserStack, Sauce Labs integration

### Q2 2025
- [ ] **On-Device Inference**: Run models locally on mobile devices
- [ ] **Cross-app Workflows**: Automate across multiple apps
- [ ] **Voice Control**: Voice commands for task creation
- [ ] **Collaborative Learning**: Share experiences across instances
- [ ] **Advanced Analytics**: ML-powered insights

### Q3 2025
- [ ] **iOS Support Enhancement**: Native XCTest integration
- [ ] **Flutter/React Native Support**: Framework-aware automation
- [ ] **Visual Regression Testing**: Automated screenshot comparison
- [ ] **A/B Testing**: Built-in experimentation framework
- [ ] **CI/CD Integration**: GitHub Actions, GitLab CI plugins

### Future
- [ ] **Natural Language Assertions**: "Verify the price is less than $100"
- [ ] **Self-Improving Agents**: Agents that learn and adapt automatically
- [ ] **Multi-device Orchestration**: Coordinate tasks across devices
- [ ] **Blockchain Integration**: Decentralized test execution
- [ ] **Quantum ML**: Explore quantum computing for RL

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **🐛 Report Bugs**: Open an issue with detailed reproduction steps
2. **💡 Suggest Features**: Share your ideas in GitHub Discussions
3. **📝 Improve Documentation**: Fix typos, add examples, clarify instructions
4. **🔧 Submit Code**: Fix bugs or implement new features
5. **🧪 Write Tests**: Increase test coverage
6. **🎨 Improve UI**: Enhance the dashboard design
7. **🌍 Translations**: Add support for other languages

### Development Setup

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/autorl.git
cd autorl

# Create a branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit code ...

# Run tests
pytest tests/

# Run linters
black src/
flake8 src/
eslint src/

# Commit your changes
git add .
git commit -m "Add: your feature description"

# Push to your fork
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
```

### Contribution Guidelines

1. **Code Style**:
   - Python: Follow PEP 8, use Black formatter
   - JavaScript: Follow Airbnb style guide, use ESLint
   - Write docstrings for all functions/classes

2. **Testing**:
   - Add tests for all new features
   - Maintain > 80% code coverage
   - Ensure all tests pass before submitting PR

3. **Documentation**:
   - Update README.md if adding features
   - Add docstrings and inline comments
   - Update API documentation

4. **Commit Messages**:
   - Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`
   - Be descriptive: "Add: support for iOS 17" not "fix"

5. **Pull Requests**:
   - One feature per PR
   - Reference related issues: "Fixes #123"
   - Provide clear description of changes
   - Include screenshots for UI changes

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on what's best for the community

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Liability and warranty not provided

## 🙏 Acknowledgments

AutoRL stands on the shoulders of giants. Special thanks to:

### Technologies
- **[React](https://reactjs.org/)** - UI framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Backend framework
- **[Appium](https://appium.io/)** - Mobile automation
- **[OpenAI](https://openai.com/)** - LLM integration
- **[Tailwind CSS](https://tailwindcss.com/)** - Styling
- **[Prometheus](https://prometheus.io/)** - Metrics
- **[Vite](https://vitejs.dev/)** - Build tool

### Inspirations
- Google's Android UI Automator
- Microsoft's Playwright
- Selenium WebDriver
- ReAct pattern for LLM agents

### Contributors
Thank you to all contributors who have helped shape AutoRL! 🎉

## 📞 Contact & Support

### Get Help
- 📖 **Documentation**: Read the [full docs](docs/)
- 💬 **GitHub Discussions**: Ask questions and share ideas
- 🐛 **GitHub Issues**: Report bugs and request features
- 💼 **Commercial Support**: enterprise@autorl.dev

### Connect
- 🌐 **Website**: [autorl.dev](https://autorl.dev) (coming soon)
- 🐦 **Twitter**: [@autorl_dev](https://twitter.com/autorl_dev)
- 📧 **Email**: hello@autorl.dev

### Stay Updated
- ⭐ Star this repository
- 👀 Watch for releases
- 🔔 Subscribe to our newsletter

---

<div align="center">

### ⭐ Star us on GitHub! ⭐

**Built with ❤️ by the AutoRL team and contributors**

[Report Bug](https://github.com/YOUR_USERNAME/autorl/issues) · [Request Feature](https://github.com/YOUR_USERNAME/autorl/issues) · [Documentation](docs/)

---

**© 2024 AutoRL. Licensed under MIT.**

*Transform your mobile automation with AI-powered intelligence*

</div>
