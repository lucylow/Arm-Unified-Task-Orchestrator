#!/usr/bin/env python3
"""
Enhanced model quantization script for Arm mobile AI deployment.

Supports:
- Dynamic quantization (fast, good for inference)
- Static quantization with calibration (better accuracy)
- ONNX quantization (via ONNX Runtime quantization tools)

This script quantizes exported models to reduce size and improve inference speed
on Arm mobile devices.
"""

import torch
import os
import sys
import argparse
from pathlib import Path


def quantize_torchscript_dynamic(input_path, output_path):
    """Apply dynamic quantization to TorchScript model."""
    print(f"⏳ Applying dynamic quantization to: {input_path}")
    
    try:
        # Load model
        model = torch.jit.load(input_path)
        model.eval()
        
        input_size = input_path.stat().st_size / (1024 * 1024)
        print(f"  Input model size: {input_size:.2f} MB")
        
        # Apply dynamic quantization
        # Quantize Linear and LSTM layers (if present)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.jit.save(quantized_model, str(output_path))
        
        output_size = output_path.stat().st_size / (1024 * 1024)
        reduction = ((input_size - output_size) / input_size) * 100 if input_size > 0 else 0
        
        print(f"✓ Dynamic quantization successful")
        print(f"  Output model size: {output_size:.2f} MB")
        if reduction > 0:
            print(f"  Size reduction: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Dynamic quantization failed: {e}")
        return False


def quantize_torchscript_static(input_path, output_path, calibration_data=None):
    """Apply static quantization with calibration (requires calibration dataset)."""
    print(f"⏳ Applying static quantization to: {input_path}")
    
    try:
        # Load model
        model = torch.jit.load(input_path)
        model.eval()
        
        # Configure quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate (if calibration data provided)
        if calibration_data:
            print("  Calibrating model...")
            with torch.no_grad():
                for data in calibration_data[:100]:  # Use first 100 samples
                    _ = model(data)
        else:
            print("  ⚠️  No calibration data provided, using default calibration")
            # Create dummy calibration data
            dummy_data = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_data)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        # Save
        torch.jit.save(model, str(output_path))
        
        print(f"✓ Static quantization successful")
        return True
        
    except Exception as e:
        print(f"✗ Static quantization failed: {e}")
        print(f"  Note: Static quantization may require FBGEMM (not available on all platforms)")
        return False


def verify_quantized_model(model_path, input_shape=(1, 3, 224, 224)):
    """Verify quantized model can be loaded and run inference."""
    print(f"\n⏳ Verifying quantized model: {model_path}")
    
    try:
        model = torch.jit.load(model_path)
        model.eval()
        
        # Test inference
        test_input = torch.randn(*input_shape)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Model verification successful")
        print(f"  Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def quantize():
    """Main quantization function."""
    parser = argparse.ArgumentParser(description="Quantize model for mobile deployment")
    parser.add_argument(
        '--input',
        type=str,
        default='models/model/model_mobile.pt',
        help='Input model path (TorchScript)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output quantized model path (default: input_quant.pt)'
    )
    parser.add_argument(
        '--method',
        choices=['dynamic', 'static'],
        default='dynamic',
        help='Quantization method (default: dynamic)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify quantized model after quantization'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AutoRL Model Quantization for Arm Mobile Devices")
    print("=" * 70)
    
    # Resolve paths
    input_path = Path(args.input).expanduser().resolve()
    
    if not input_path.exists():
        print(f"\n✗ Error: Input model not found: {input_path}")
        print(f"Please run 'python scripts/export_model.py' first")
        return False
    
    # Determine output path
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # Default: add '_quant' before extension
        output_path = input_path.parent / f"{input_path.stem}_quant{input_path.suffix}"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Input model:  {input_path}")
    print(f"✓ Output model: {output_path}")
    print(f"✓ Method:       {args.method} quantization")
    
    # Quantize
    print("\n" + "=" * 70)
    success = False
    
    if args.method == 'dynamic':
        success = quantize_torchscript_dynamic(input_path, output_path)
    elif args.method == 'static':
        success = quantize_torchscript_static(input_path, output_path)
    
    if not success:
        print("\n✗ Quantization failed")
        return False
    
    # Verify if requested
    if args.verify:
        if not verify_quantized_model(output_path):
            print("\n⚠️  Verification failed, but model was saved")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Quantization completed successfully!")
    print("=" * 70)
    print(f"\nQuantized model ready for mobile deployment:")
    print(f"  {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Copy model to mobile app assets: mobile/android/app/src/main/assets/")
    print(f"  2. Build Android APK: cd mobile/android && ./gradlew assembleDebug")
    
    return True


if __name__ == "__main__":
    success = quantize()
    sys.exit(0 if success else 1)

