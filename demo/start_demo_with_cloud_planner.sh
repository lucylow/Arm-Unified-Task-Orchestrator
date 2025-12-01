#!/usr/bin/env bash
#
# Complete Demo Script for AutoRL ARM Edition
# Starts backend with cloud planner + ARM integration + mock data
# Then builds and installs mobile app on emulator
#

set -e

echo "============================================================"
echo "AutoRL ARM Edition - Complete Demo Setup"
echo "Cloud Planner + ARM Integration + Mock Data"
echo "============================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Step 1: Checking prerequisites...${NC}"
echo "------------------------------------------------------------"

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found${NC}"

# Check ADB
if ! command -v adb &> /dev/null; then
    echo -e "${RED}❌ ADB not found. Please install Android SDK Platform Tools.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ ADB found${NC}"

echo ""
echo -e "${CYAN}Step 2: Checking for Android emulator...${NC}"
echo "------------------------------------------------------------"

DEVICES=$(adb devices | grep -v "List" | grep "device$" | wc -l)

if [ "$DEVICES" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No Android device or emulator connected.${NC}"
    echo ""
    echo "Please start Android emulator first:"
    echo "  1. Open Android Studio"
    echo "  2. Go to Tools → Device Manager"
    echo "  3. Start your ARM emulator (ARM 64 v8a)"
    echo "  4. Wait for it to fully boot"
    echo "  5. Run this script again"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ Found $DEVICES connected device(s)${NC}"
adb devices

# Verify ARM architecture
DEVICE_NAME=$(adb devices | grep -v "List" | grep "device$" | head -n 1 | awk '{print $1}')
CPU_ABI=$(adb shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')

if [[ "$CPU_ABI" == "arm64-v8a" ]] || [[ "$CPU_ABI" == "armeabi-v7a" ]]; then
    echo -e "${GREEN}✅ ARM architecture confirmed: $CPU_ABI${NC}"
else
    echo -e "${YELLOW}⚠️  Architecture is $CPU_ABI (expected arm64-v8a)${NC}"
    echo "   This may still work, but ARM optimizations won't be used."
fi

echo ""
echo -e "${CYAN}Step 3: Checking backend dependencies...${NC}"
echo "------------------------------------------------------------"

cd "$PROJECT_ROOT/backend"

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install -r requirements.txt
fi

echo -e "${GREEN}✅ Backend dependencies OK${NC}"

echo ""
echo -e "${CYAN}Step 4: Checking frontend dependencies...${NC}"
echo "------------------------------------------------------------"

cd "$PROJECT_ROOT/frontend"

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

echo -e "${GREEN}✅ Frontend dependencies OK${NC}"

echo ""
echo -e "${CYAN}Step 5: Building mobile app...${NC}"
echo "------------------------------------------------------------"

cd "$PROJECT_ROOT/mobile/android"

if [ ! -f "app/build/outputs/apk/debug/app-debug.apk" ]; then
    echo "Building APK..."
    ./gradlew assembleDebug
else
    echo -e "${GREEN}✅ APK already built${NC}"
fi

APK_PATH="app/build/outputs/apk/debug/app-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    echo -e "${RED}❌ APK not found. Build failed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ APK built successfully${NC}"

echo ""
echo -e "${CYAN}Step 6: Installing APK on emulator...${NC}"
echo "------------------------------------------------------------"

adb install -r "$APK_PATH"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ APK installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  APK installation failed (may already be installed)${NC}"
fi

echo ""
echo -e "${CYAN}Step 7: Launching app...${NC}"
echo "------------------------------------------------------------"

adb shell am start -n com.autorl/.MainActivity
echo -e "${GREEN}✅ App launched${NC}"

echo ""
echo "============================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "============================================================"
echo ""
echo -e "${CYAN}📱 Next Steps:${NC}"
echo ""
echo "1. ${YELLOW}Start Backend Server${NC} (in a new terminal):"
echo "   cd backend"
echo "   source venv/bin/activate  # or: venv\\Scripts\\activate on Windows"
echo "   export AUTORL_MODE=demo  # or: set AUTORL_MODE=demo on Windows"
echo "   python -m servers.master_backend"
echo ""
echo "2. ${YELLOW}Start Frontend Dashboard${NC} (in another terminal):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. ${YELLOW}Open Browser:${NC}"
echo "   http://localhost:5173"
echo ""
echo "4. ${YELLOW}Test the Complete Flow:${NC}"
echo "   - In emulator: Tap 'Start Task' button"
echo "   - In dashboard: Watch real-time task execution"
echo "   - In backend logs: See cloud planner + ARM inference"
echo ""
echo -e "${CYAN}🧪 Verification Commands:${NC}"
echo ""
echo "  # Check backend health:"
echo "  curl http://localhost:5000/api/health"
echo ""
echo "  # View mobile app logs:"
echo "  adb logcat -s ModelRunner:I MainActivity:I"
echo ""
echo "  # Test API endpoint:"
echo "  curl -X POST http://localhost:5000/api/tasks \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"device_id\": \"emulator-5554\", \"instruction\": \"Send money to John\"}'"
echo ""
echo -e "${CYAN}📚 For complete guide:${NC}"
echo "   See docs/ANDROID_EMULATOR_TESTING.md"
echo ""

