# 📱 Android Emulator Testing Guide

**Complete step-by-step guide for testing AutoRL ARM Edition with Android Emulator**

> **💡 Quick Reference:** For a printable command cheat sheet, see [EMULATOR_QUICK_REFERENCE.md](EMULATOR_QUICK_REFERENCE.md)

## 🎯 Demo Setup Overview

**This guide configures a 100% working demo with:**

✅ **Cloud Planner**: LLM-based planning runs on backend server (not on-device)  
✅ **ARM Integration**: On-device vision inference for UI element detection  
✅ **Mock Data**: Realistic mock responses for reliable demo  
✅ **Complete Flow**: Mobile app → Backend → Frontend dashboard  

**Architecture:**
```
┌─────────────────────────────────────────┐
│  Android Emulator (ARM)                 │
│  ┌───────────────────────────────────┐ │
│  │  Mobile App                       │ │
│  │  - ARM Vision Inference          │ │
│  │  - Sends screenshots to backend  │ │
│  └──────────────┬────────────────────┘ │
└─────────────────┼───────────────────────┘
                  │ HTTP API
┌─────────────────▼───────────────────────┐
│  Backend Server (Python)                 │
│  ┌───────────────────────────────────┐  │
│  │  Cloud Planner (LLM)             │  │
│  │  - Generates action plans        │  │
│  │  - Uses mock data for demo        │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  ARM Inference Engine             │  │
│  │  - Processes screenshots         │  │
│  │  - UI element detection          │  │
│  └───────────────────────────────────┘  │
└─────────────────┬───────────────────────┘
                  │ WebSocket
┌─────────────────▼───────────────────────┐
│  Frontend Dashboard (React)              │
│  - Real-time task monitoring           │
│  - ARM performance metrics              │
│  - Cloud planner decisions              │
└─────────────────────────────────────────┘
```

**This setup is 100% functional and ready for demo! 🚀**

## 🎯 Quick Start (5 Minutes)

### Step 1: Install Android Studio

1. **Download Android Studio**
   - Visit: https://developer.android.com/studio
   - Download and install Android Studio for your OS (Windows/Mac/Linux)

2. **Install Required Components**
   - Open Android Studio
   - Go to **Tools → SDK Manager**
   - In **SDK Platforms** tab, install:
     - ✅ Android 11.0 (API 30) or higher
     - ✅ Android 12.0 (API 31) or higher (recommended)
   - In **SDK Tools** tab, ensure these are checked:
     - ✅ Android SDK Build-Tools
     - ✅ Android SDK Platform-Tools
     - ✅ Android Emulator
     - ✅ Android SDK Command-line Tools

3. **Set Environment Variables** (Important!)

   **Windows (PowerShell):**
   ```powershell
   # Add to your user environment variables or run in PowerShell:
   $env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
   $env:PATH += ";$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator"
   ```

   **Windows (Command Prompt):**
   ```cmd
   setx ANDROID_HOME "%LOCALAPPDATA%\Android\Sdk"
   setx PATH "%PATH%;%ANDROID_HOME%\platform-tools;%ANDROID_HOME%\emulator"
   ```

   **Mac/Linux:**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc:
   export ANDROID_HOME=$HOME/Library/Android/sdk
   export PATH=$PATH:$ANDROID_HOME/platform-tools
   export PATH=$PATH:$ANDROID_HOME/emulator
   
   # Then reload:
   source ~/.bashrc  # or source ~/.zshrc
   ```

### Step 2: Create an Android Virtual Device (AVD)

1. **Open AVD Manager**
   - In Android Studio: **Tools → Device Manager**
   - Or click the device icon in the toolbar

2. **Create Virtual Device**
   - Click **"Create Device"**
   - Select a device definition (e.g., **Pixel 6** or **Pixel 7**)
   - Click **Next**

3. **Select System Image**
   - Choose **API 31 (Android 12)** or **API 30 (Android 11)**
   - **IMPORTANT**: Select an image with **arm64-v8a** architecture (not x86_64)
   - Look for "ARM 64 v8a" in the system image name
   - If you don't see ARM images, click **"Download"** next to one
   - Click **Next**

4. **Configure AVD**
   - Name it: `AutoRL_ARM_Emulator`
   - Verify settings:
     - Graphics: **Hardware - GLES 2.0** (recommended)
     - RAM: 2048 MB or more
   - Click **Finish**

### Step 3: Start the Emulator

**Option A: From Android Studio**
- Open **Device Manager**
- Click the **▶️ Play** button next to your AVD

**Option B: From Command Line**
```bash
# List available emulators
emulator -list-avds

# Start your emulator (replace with your AVD name)
emulator -avd AutoRL_ARM_Emulator

# Or start with specific options:
emulator -avd AutoRL_ARM_Emulator -gpu host -no-snapshot-load
```

**Wait for emulator to fully boot** (may take 1-2 minutes on first launch)

### Step 4: Verify Emulator Connection

Open a terminal/command prompt and run:

```bash
# Check if ADB can see the emulator
adb devices
```

**Expected output:**
```
List of devices attached
emulator-5554    device
```

If you see `emulator-5554 device`, you're ready! ✅

**If you see "unauthorized":**
- Check the emulator screen for a USB debugging authorization prompt
- Click "Allow" or "Always allow from this computer"

**If no devices appear:**
- Make sure the emulator is fully booted (home screen visible)
- Try: `adb kill-server` then `adb start-server`
- Verify ADB is in your PATH: `adb version`

### Step 5: Verify ARM Architecture

```bash
# Check the emulator's CPU architecture
adb shell getprop ro.product.cpu.abi
```

**Expected output:** `arm64-v8a` or `armeabi-v7a`

If you see `x86_64` or `x86`, you need to create a new AVD with an ARM system image!

## 🚀 Testing AutoRL on Emulator with Cloud Planner + ARM Integration

**This setup uses:**
- ✅ **Cloud Planner**: LLM-based planning runs on backend (not on-device)
- ✅ **ARM Integration**: On-device vision inference for UI detection
- ✅ **Mock Data**: Demo mode with realistic mock responses
- ✅ **100% Working**: Fully functional demo setup

### Step 1: Start Backend Server with Demo Mode

**Open Terminal 1 - Backend Server:**

```bash
# Navigate to project root
cd autorl-arm-edition-hackathon-submission

# Set demo mode environment variable
export AUTORL_MODE=demo  # On Windows: set AUTORL_MODE=demo

# Start backend server
cd backend
python -m servers.master_backend
```

**Expected output:**
```
✅ AutoRL Master Backend ready
INFO:     Uvicorn running on http://0.0.0.0:5000
Running in DEMO MODE with mock data
```

**Verify backend is running:**
```bash
# In another terminal, test the API
curl http://localhost:5000/api/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "mode": "demo",
  "orchestrator_status": "initialized"
}
```

### Step 2: Start Frontend Dashboard

**Open Terminal 2 - Frontend:**

```bash
# Navigate to frontend directory
cd frontend

# Start frontend dev server
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Open browser:** http://localhost:5173

You should see the AutoRL dashboard with:
- Device management (showing mock devices)
- Task execution interface
- Real-time metrics
- ARM performance indicators

### Step 3: Build the Mobile App

**Open Terminal 3 - Build APK:**

```bash
# Navigate to project root
cd autorl-arm-edition-hackathon-submission

# Build the APK (this will also export and quantize the model)
# On Windows:
cd mobile\android
gradlew.bat assembleDebug

# On Mac/Linux:
cd mobile/android
./gradlew assembleDebug
```

**Expected output:**
```
BUILD SUCCESSFUL in 30s
```

The APK will be at: `mobile/android/app/build/outputs/apk/debug/app-debug.apk`

### Step 4: Start Android Emulator

**Make sure emulator is running:**

```bash
# Check if emulator is running
adb devices
```

**Should show:**
```
List of devices attached
emulator-5554    device
```

If not running, start it from Android Studio or command line.

### Step 5: Install APK on Emulator

```bash
# Install the APK
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
```

**Expected output:**
```
Performing Streamed Install
Success
```

### Step 6: Launch the App

```bash
# Launch AutoRL app
adb shell am start -n com.autorl/.MainActivity
```

The app should open on your emulator screen!

### Step 7: Test Complete Workflow

**The Demo Flow:**

1. **On Emulator (Mobile App):**
   - App shows AutoRL interface
   - Tap **"Start Task"** button
   - App captures screenshot (ARM vision inference runs on-device)
   - App sends screenshot to backend via API

2. **Backend (Cloud Planner):**
   - Receives screenshot from mobile app
   - Uses ARM inference engine for UI element detection
   - Cloud planner (LLM) generates action plan using mock data
   - Returns action plan to mobile app

3. **Frontend Dashboard:**
   - Shows real-time task execution
   - Displays ARM performance metrics
   - Shows cloud planner decisions
   - Updates with mock data responses

**Test Commands:**

```bash
# View mobile app logs
adb logcat -s ModelRunner:I MainActivity:I

# View backend logs (in Terminal 1)
# You'll see:
# - "Creating plan for: [task description]"
# - "Mock planning" or "LLM planning"
# - "ARM inference completed in XX ms"

# Test API endpoint directly
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "emulator-5554",
    "instruction": "Send money to John",
    "task_type": "automation"
  }'
```

**Expected API Response:**
```json
{
  "task_id": "task_xxx",
  "status": "executing",
  "actions": [
    {"action": "tap", "target_id": "payment_app"},
    {"action": "type_text", "target_id": "recipient", "value": "John"},
    {"action": "type_text", "target_id": "amount", "value": "25"},
    {"action": "tap", "target_id": "send_button"}
  ],
  "confidence": 0.92,
  "arm_inference_ms": 45.23
}
```

### Step 8: Verify Complete Integration

**Check all components are working:**

```bash
# 1. Backend health
curl http://localhost:5000/api/health

# 2. Backend status (should show demo mode)
curl http://localhost:5000/api/status

# 3. Frontend accessible
curl http://localhost:5173

# 4. Emulator connected
adb devices

# 5. App installed
adb shell pm list packages | grep autorl

# 6. ARM architecture verified
adb shell getprop ro.product.cpu.abi
# Should show: arm64-v8a
```

**All should return success! ✅**

## 🎬 Complete Demo Script with Cloud Planner

**🚀 Automated Setup Script (Recommended)**

We provide a complete setup script that:
- ✅ Checks all prerequisites
- ✅ Verifies emulator connection and ARM architecture
- ✅ Builds the mobile app
- ✅ Installs APK on emulator
- ✅ Launches the app
- ✅ Provides next steps for starting backend/frontend

**Run the setup script:**

**Unix/Mac:**
```bash
./demo/start_demo_with_cloud_planner.sh
```

**Windows:**
```cmd
demo\start_demo_with_cloud_planner.bat
```

**After the script completes, follow these steps:**

### Step 1: Start Backend Server (Terminal 1)

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set demo mode
export AUTORL_MODE=demo  # On Windows: set AUTORL_MODE=demo

# Start backend
python -m servers.master_backend
```

**Expected output:**
```
✅ AutoRL Master Backend ready
INFO:     Uvicorn running on http://0.0.0.0:5000
Running in DEMO MODE with mock data
```

### Step 2: Start Frontend Dashboard (Terminal 2)

```bash
cd frontend
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

### Step 3: Test Complete Flow

1. **In the mobile app (on emulator):**
   - Tap "Start Task" button
   - Watch for ARM inference results
   - App sends screenshot to backend

2. **In the frontend dashboard (browser - http://localhost:5173):**
   - Navigate to "Tasks" or "Devices" page
   - You should see task execution in real-time
   - ARM performance metrics displayed
   - Cloud planner decisions shown

3. **In backend logs (Terminal 1):**
   - Cloud planner generating action plans
   - ARM inference engine processing screenshots
   - Mock data responses being used
   - Example log output:
     ```
     [PlanningAgent] Creating plan for: Send money to John
     [ARMInferenceEngine] Vision inference completed in 45.23 ms
     [MasterAgentOrchestrator] Task completed successfully
     ```

### Step 4: Verify Everything Works

**Test API directly:**
```bash
# Health check
curl http://localhost:5000/api/health

# Create a task
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "emulator-5554",
    "instruction": "Send money to John",
    "task_type": "automation"
  }'
```

**View mobile app logs:**
```bash
adb logcat -s ModelRunner:I MainActivity:I
```

**All components should be working! ✅**

## 🔍 Troubleshooting

### Problem: Emulator won't start

**Solution:**
- Check if virtualization is enabled in BIOS (Intel VT-x or AMD-V)
- On Windows, enable Hyper-V or use Intel HAXM
- Try: `emulator -avd AutoRL_ARM_Emulator -verbose` to see errors

### Problem: "x86_64" instead of "arm64-v8a"

**Solution:**
- You created an x86 AVD instead of ARM
- Delete the AVD and create a new one
- Make sure to select a system image with "ARM" in the name
- ARM images are slower but required for ARM testing

### Problem: ADB can't find emulator

**Solution:**
```bash
# Restart ADB server
adb kill-server
adb start-server

# Check if emulator is running
adb devices

# If still not found, restart emulator
```

### Problem: APK installation fails

**Solution:**
```bash
# Uninstall existing app first
adb uninstall com.autorl

# Then reinstall
adb install -r mobile/android/app/build/outputs/apk/debug/app-debug.apk
```

### Problem: App crashes on launch

**Solution:**
```bash
# Check crash logs
adb logcat -s AndroidRuntime:E

# Common issues:
# - Model file missing: Rebuild APK
# - Insufficient memory: Increase emulator RAM
# - Wrong architecture: Use ARM emulator
```

### Problem: Slow emulator performance

**Solution:**
- Increase emulator RAM (2048 MB minimum, 4096 MB recommended)
- Enable hardware acceleration (Graphics: Hardware - GLES 2.0)
- Close other applications
- Use snapshot loading: `emulator -avd AutoRL_ARM_Emulator -snapshot-load`

## 📊 Performance Testing on Emulator

### Measure Inference Latency

```bash
# Monitor inference times
adb logcat -s ModelRunner:I | grep "Inference completed"
```

### Capture Performance Trace

```bash
# Start Perfetto trace (30 seconds)
adb shell perfetto \
  -c - --txt \
  -o /data/misc/perfetto-traces/autorl_trace.pb <<EOF
buffers: {
    size_kb: 63488
    fill_policy: DISCARD
}
data_sources: {
    config {
        name: "linux.process_stats"
        target_buffer: 0
        process_stats_config {
            scan_all_processes_on_start: true
        }
    }
}
duration_ms: 30000
EOF

# Pull trace file
adb pull /data/misc/perfetto-traces/autorl_trace.pb .

# Open in Chrome: https://ui.perfetto.dev/
```

### Test Offline Operation

```bash
# Enable airplane mode on emulator
adb shell cmd connectivity airplane-mode enable

# Launch app and test
adb shell am start -n com.autorl/.MainActivity

# Inference should still work (on-device AI!)

# Disable airplane mode
adb shell cmd connectivity airplane-mode disable
```

## ✅ Verification Checklist

Before submitting or demonstrating, verify:

- [ ] Emulator is running and connected (`adb devices` shows device)
- [ ] Emulator uses ARM architecture (`adb shell getprop ro.product.cpu.abi` shows `arm64-v8a`)
- [ ] APK is installed successfully
- [ ] App launches without crashes
- [ ] "Start Task" button works
- [ ] Inference completes (check logs for "Inference completed in XX ms")
- [ ] App works in airplane mode (offline capability)
- [ ] Memory usage is reasonable (< 100 MB)

## 🎥 Demo Video Tips

When recording your demo video:

1. **Show emulator setup** (10 seconds)
   - Open Android Studio Device Manager
   - Show AVD with ARM architecture

2. **Show ADB connection** (5 seconds)
   - Run `adb devices` in terminal
   - Show `emulator-5554 device`

3. **Install and launch** (15 seconds)
   - Run `adb install` command
   - Show app launching on emulator

4. **Demonstrate inference** (30 seconds)
   - Tap "Start Task" button
   - Show inference results on screen
   - Show logcat output with timing

5. **Show offline capability** (20 seconds)
   - Enable airplane mode
   - Run inference again
   - Show it works without network

6. **Performance metrics** (20 seconds)
   - Show Perfetto trace
   - Highlight ARM architecture
   - Show memory usage

## 📚 Additional Resources

- [Android Studio Setup Guide](https://developer.android.com/studio/intro)
- [Create and Manage Virtual Devices](https://developer.android.com/studio/run/managing-avds)
- [ADB Command Reference](https://developer.android.com/studio/command-line/adb)
- [Perfetto Tracing](https://perfetto.dev/)

## 🆘 Still Having Issues?

1. **Check Android Studio logs:**
   - Help → Show Log in Finder/Explorer

2. **Verify environment:**
   ```bash
   echo $ANDROID_HOME  # Should show SDK path
   adb version         # Should show version
   emulator -version   # Should show emulator version
   ```

3. **Common fixes:**
   - Restart Android Studio
   - Restart emulator
   - Restart ADB: `adb kill-server && adb start-server`
   - Rebuild project: `./gradlew clean assembleDebug`

---

**🎉 You're now ready to test AutoRL ARM Edition on Android Emulator!**

For questions or issues, check the main README.md or open a GitHub issue.

