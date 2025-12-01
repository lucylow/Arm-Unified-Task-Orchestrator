#!/usr/bin/env bash
# Perfetto trace capture script for AutoRL mobile AI profiling
# Captures performance traces from Android devices via ADB

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PERFETTO_CONFIG="$PROJECT_ROOT/perfetto/trace_config.pbtx"
OUTPUT_DIR="$PROJECT_ROOT/perfetto"
OUTPUT_FILE="$OUTPUT_DIR/trace_$(date +%Y%m%d_%H%M%S).pb"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}AutoRL Perfetto Trace Capture${NC}"
echo "=================================="

# Check if ADB is available
if ! command -v adb &> /dev/null; then
    echo -e "${RED}Error: ADB not found. Please install Android SDK platform-tools.${NC}"
    exit 1
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo -e "${RED}Error: No Android device connected.${NC}"
    echo "Please connect a device via USB and enable USB debugging."
    exit 1
fi

echo -e "${GREEN}✓ Device connected${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Push config to device
echo "Pushing Perfetto config to device..."
adb push "$PERFETTO_CONFIG" /data/local/tmp/autorl_perfetto_config.pbtx

# Check if we need root
if ! adb shell su -c "test -d /data/misc/perfetto-traces" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Root access not available. Using user mode.${NC}"
    OUTPUT_PATH="/sdcard/perfetto_trace.pb"
    USE_ROOT=false
else
    OUTPUT_PATH="/data/misc/perfetto-traces/autorl_trace.pb"
    USE_ROOT=true
fi

# Start Perfetto in background
echo "Starting Perfetto trace capture..."
if [ "$USE_ROOT" = true ]; then
    adb shell su -c "rm -f $OUTPUT_PATH" 2>/dev/null || true
    adb shell su -c "perfetto -c /data/local/tmp/autorl_perfetto_config.pbtx -o $OUTPUT_PATH" &
else
    adb shell "rm -f $OUTPUT_PATH" 2>/dev/null || true
    adb shell "perfetto -c /data/local/tmp/autorl_perfetto_config.pbtx -o $OUTPUT_PATH" &
fi

PERFETTO_PID=$!
sleep 2

echo -e "${GREEN}✓ Perfetto started (PID: $PERFETTO_PID)${NC}"
echo ""
echo "=================================="
echo "Now run your AutoRL demo/task..."
echo "The trace is capturing (30 seconds default)"
echo ""
echo "Press Ctrl+C to stop early, or wait for completion..."
echo "=================================="

# Wait for user interrupt or completion
trap "echo -e '\n${YELLOW}Stopping trace capture...${NC}'; kill $PERFETTO_PID 2>/dev/null || true; sleep 2" INT TERM

wait $PERFETTO_PID || true

sleep 2

# Pull trace file
echo ""
echo "Pulling trace file from device..."
if [ "$USE_ROOT" = true ]; then
    adb shell su -c "chmod 644 $OUTPUT_PATH" 2>/dev/null || true
    adb pull "$OUTPUT_PATH" "$OUTPUT_FILE"
    adb shell su -c "rm -f $OUTPUT_PATH" 2>/dev/null || true
else
    adb pull "$OUTPUT_PATH" "$OUTPUT_FILE" || {
        echo -e "${RED}Error: Failed to pull trace file${NC}"
        exit 1
    }
    adb shell "rm -f $OUTPUT_PATH" 2>/dev/null || true
fi

if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo -e "${GREEN}✓ Trace captured successfully${NC}"
    echo "  File: $OUTPUT_FILE"
    echo "  Size: $FILE_SIZE"
    echo ""
    echo "To view the trace:"
    echo "  1. Go to https://ui.perfetto.dev/"
    echo "  2. Click 'Open trace file'"
    echo "  3. Select: $OUTPUT_FILE"
else
    echo -e "${RED}Error: Trace file not found${NC}"
    exit 1
fi

