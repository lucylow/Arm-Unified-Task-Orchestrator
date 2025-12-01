#!/usr/bin/env bash
# Android build helper script for AutoRL mobile app
# Builds the Android APK with Arm-optimized models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MOBILE_DIR="$PROJECT_ROOT/mobile/android"
MODELS_DIR="$PROJECT_ROOT/models/model"
ASSETS_DIR="$MOBILE_DIR/app/src/main/assets"

echo "=================================="
echo "AutoRL Android Build Script"
echo "=================================="

# Check if model files exist
if [ ! -f "$MODELS_DIR/model_mobile_quant.pt" ]; then
    echo "Warning: Quantized model not found at $MODELS_DIR/model_mobile_quant.pt"
    echo "Attempting to export and quantize model..."
    
    cd "$PROJECT_ROOT"
    
    # Check if Python is available
    if command -v python3 &> /dev/null; then
        PYTHON=python3
    elif command -v python &> /dev/null; then
        PYTHON=python
    else
        echo "Error: Python not found. Please install Python 3."
        exit 1
    fi
    
    # Export model
    if [ -f "$PROJECT_ROOT/models/model/export_model.py" ]; then
        echo "Exporting model..."
        $PYTHON "$PROJECT_ROOT/models/model/export_model.py" || {
            echo "Warning: Model export failed, continuing anyway..."
        }
    fi
    
    # Quantize model
    if [ -f "$PROJECT_ROOT/models/model/quantize_model.py" ]; then
        echo "Quantizing model..."
        $PYTHON "$PROJECT_ROOT/models/model/quantize_model.py" || {
            echo "Warning: Model quantization failed, continuing anyway..."
        }
    fi
fi

# Ensure assets directory exists
mkdir -p "$ASSETS_DIR"
mkdir -p "$ASSETS_DIR/models"

# Copy models to assets
if [ -f "$MODELS_DIR/model_mobile_quant.pt" ]; then
    echo "Copying quantized model to assets..."
    cp "$MODELS_DIR/model_mobile_quant.pt" "$ASSETS_DIR/models/" || {
        echo "Warning: Failed to copy model file"
    }
fi

if [ -f "$MODELS_DIR/model_mobile.onnx" ]; then
    echo "Copying ONNX model to assets..."
    cp "$MODELS_DIR/model_mobile.onnx" "$ASSETS_DIR/models/" || {
        echo "Warning: Failed to copy ONNX model file"
    }
fi

# Copy test screenshot if available
if [ -f "$PROJECT_ROOT/demo/test_screen.png" ]; then
    echo "Copying test screenshot to assets..."
    cp "$PROJECT_ROOT/demo/test_screen.png" "$ASSETS_DIR/" || true
fi

# Build APK
cd "$MOBILE_DIR"

if [ ! -f "gradlew" ]; then
    echo "Error: gradlew not found in $MOBILE_DIR"
    exit 1
fi

# Make gradlew executable
chmod +x gradlew

echo ""
echo "Building Android APK..."
echo "=================================="

# Build debug APK
./gradlew assembleDebug --stacktrace

APK_PATH="$MOBILE_DIR/app/build/outputs/apk/debug/app-debug.apk"

if [ -f "$APK_PATH" ]; then
    APK_SIZE=$(du -h "$APK_PATH" | cut -f1)
    echo ""
    echo "=================================="
    echo "✅ Build successful!"
    echo "=================================="
    echo "APK: $APK_PATH"
    echo "Size: $APK_SIZE"
    echo ""
    echo "To install on device:"
    echo "  adb install -r $APK_PATH"
    echo ""
else
    echo "Error: APK not found at expected location"
    exit 1
fi

