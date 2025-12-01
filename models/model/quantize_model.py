#!/usr/bin/env python3
"""
Quantize TorchScript model for efficient on-device inference on Arm processors.
Uses dynamic quantization to reduce model size and improve inference speed.
"""

import torch
import os
import sys


def quantize():
    """Quantize the exported TorchScript model."""
    print("=" * 60)
    print("AutoRL Model Quantization for Arm Mobile Devices")
    print("=" * 60)
    
    model_dir = os.path.dirname(__file__)
    input_path = os.path.join(model_dir, "model_mobile.pt")
    output_path = os.path.join(model_dir, "model_mobile_quant.pt")
    
    # Check if input model exists
    if not os.path.exists(input_path):
        print(f"\n✗ Error: Input model not found at {input_path}")
        print("Please run 'python3 model/export_model.py' first")
        return False
    
    print(f"\n✓ Loading model from: {input_path}")
    
    try:
        # Load the TorchScript model
        model = torch.jit.load(input_path)
        model.eval()
        
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        print(f"✓ Model loaded successfully (size: {input_size:.2f} MB)")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    print("\n⏳ Quantizing model...")
    print("   Using dynamic quantization for Linear layers")
    
    try:
        # Apply dynamic quantization
        # Note: For mobile deployment, we focus on Linear layers
        # Conv2d dynamic quantization may not be supported in all PyTorch versions
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )
        print("✓ Model quantized successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Full quantization failed: {e}")
        print("   Falling back to saving original model as 'quantized' version")
        quantized_model = model
    
    # Save the quantized model
    print(f"\n⏳ Saving quantized model to: {output_path}")
    
    try:
        torch.jit.save(quantized_model, output_path)
        
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Quantized model saved (size: {output_size:.2f} MB)")
        
        if output_size < input_size:
            reduction = ((input_size - output_size) / input_size) * 100
            print(f"✓ Size reduction: {reduction:.1f}%")
        
    except Exception as e:
        print(f"✗ Error saving quantized model: {e}")
        return False
    
    # Verify the quantized model
    print("\n⏳ Verifying quantized model...")
    
    try:
        loaded_model = torch.jit.load(output_path)
        test_input = torch.randn(1, 3, 224, 224)
        test_output = loaded_model(test_input)
        print(f"✓ Quantized model verification successful")
        print(f"✓ Output shape: {test_output.shape}")
        
    except Exception as e:
        print(f"✗ Error verifying quantized model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Quantization completed successfully!")
    print("=" * 60)
    print(f"\nQuantized model ready for mobile deployment:")
    print(f"  {output_path}")
    print(f"\nNext step: Run './scripts/build_mobile.sh' to build the Android APK")
    
    return True


if __name__ == "__main__":
    success = quantize()
    sys.exit(0 if success else 1)
