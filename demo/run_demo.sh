#!/usr/bin/env bash
#
# Demo script for AutoRL Arm Edition
# Demonstrates the complete workflow: planner, APK installation, and app launch
#

set -e

echo "============================================================"
echo "AutoRL Arm Edition - Demo Script"
echo "============================================================"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo ""
echo "Step 1: Test Tiny Planner"
echo "------------------------------------------------------------"

python3 planner/tiny_planner.py

echo ""
echo "Step 2: Check for connected Android device"
echo "------------------------------------------------------------"

if ! command -v adb &> /dev/null; then
    echo "⚠️  ADB not found. Please install Android SDK Platform Tools."
    echo "   Download: https://developer.android.com/studio/releases/platform-tools"
    exit 1
fi

# Check for connected devices
DEVICES=$(adb devices | grep -v "List" | grep "device$" | wc -l)

if [ "$DEVICES" -eq 0 ]; then
    echo "⚠️  No Android device or emulator connected."
    echo ""
    echo "📱 TO USE ANDROID EMULATOR (Recommended for Testing):"
    echo "  1. Open Android Studio"
    echo "  2. Go to Tools → Device Manager"
    echo "  3. Create a new AVD with ARM 64 v8a architecture"
    echo "     (IMPORTANT: Must be ARM, not x86_64!)"
    echo "  4. Start the emulator (click ▶️ Play button)"
    echo "  5. Wait for emulator to fully boot"
    echo "  6. Run 'adb devices' to verify connection"
    echo ""
    echo "   See complete guide: docs/ANDROID_EMULATOR_TESTING.md"
    echo ""
    echo "📱 TO USE PHYSICAL DEVICE:"
    echo "  1. Enable USB debugging on your Android device"
    echo "  2. Connect via USB cable"
    echo "  3. Run 'adb devices' to verify connection"
    echo ""
    exit 1
fi

echo "✓ Found $DEVICES connected device(s)"
echo ""
echo "Connected devices:"
adb devices
echo ""

# Check if it's an emulator and verify ARM architecture
DEVICE_NAME=$(adb devices | grep -v "List" | grep "device$" | head -n 1 | awk '{print $1}')
if [[ "$DEVICE_NAME" == emulator-* ]]; then
    echo "📱 Detected Android Emulator: $DEVICE_NAME"
    echo "   Verifying ARM architecture..."
    CPU_ABI=$(adb shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')
    if [[ "$CPU_ABI" == "arm64-v8a" ]] || [[ "$CPU_ABI" == "armeabi-v7a" ]]; then
        echo "   ✅ ARM architecture confirmed: $CPU_ABI"
    else
        echo "   ⚠️  WARNING: Architecture is $CPU_ABI (expected arm64-v8a or armeabi-v7a)"
        echo "   This emulator may not work correctly. Please create an ARM-based AVD."
    fi
else
    echo "📱 Detected physical device: $DEVICE_NAME"
    CPU_ABI=$(adb shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')
    echo "   Architecture: $CPU_ABI"
fi

echo ""
echo "Step 3: Install APK"
echo "------------------------------------------------------------"

APK_PATH="mobile/android/app/build/outputs/apk/debug/app-debug.apk"

if [ ! -f "$APK_PATH" ]; then
    echo "⚠️  APK not found at $APK_PATH"
    echo "   Please run './scripts/build_mobile.sh' first"
    exit 1
fi

echo "Installing APK..."
adb install -r "$APK_PATH"

if [ $? -eq 0 ]; then
    echo "✓ APK installed successfully"
else
    echo "⚠️  APK installation failed (may already be installed)"
fi

echo ""
echo "Step 4: Launch AutoRL app"
echo "------------------------------------------------------------"

adb shell am start -n com.autorl/.MainActivity

echo "✓ App launched"

echo ""
echo "Step 5: Collect device information"
echo "------------------------------------------------------------"

echo ""
echo "Device Info:"
adb shell getprop ro.product.model
adb shell getprop ro.product.cpu.abi

echo ""
echo "Memory Info:"
adb shell dumpsys meminfo com.autorl | grep TOTAL | head -n 1

echo ""
echo "============================================================"
echo "✅ Demo completed!"
echo "============================================================"
echo ""
echo "📱 Next steps:"
echo "  1. Look at your emulator/device screen"
echo "  2. Tap 'Start Task' button in the AutoRL app"
echo "  3. View inference results on screen"
echo "  4. Check logcat for detailed timing:"
echo "     adb logcat -s ModelRunner:I MainActivity:I"
echo ""
echo "🧪 Testing commands:"
echo "  # Simulate tap action (example):"
echo "  adb shell input tap 540 960"
echo ""
echo "  # View real-time logs:"
echo "  adb logcat -s ModelRunner:I MainActivity:I | grep 'Inference'"
echo ""
echo "  # Test offline capability:"
echo "  adb shell cmd connectivity airplane-mode enable"
echo "  # Then tap 'Start Task' again - should still work!"
echo "  adb shell cmd connectivity airplane-mode disable"
echo ""
echo "📚 For complete testing guide:"
echo "  See docs/ANDROID_EMULATOR_TESTING.md"
echo ""
