#!/usr/bin/env python3
"""
Export perception model to TorchScript for mobile deployment.
This script creates a traced TorchScript model optimized for Arm mobile devices.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class SmallPerceptionModel(nn.Module):
    """
    Lightweight perception model for mobile UI element detection.
    Optimized for Arm processors with reduced parameters.
    """
    def __init__(self, num_classes=10):
        super(SmallPerceptionModel, self).__init__()
        
        # Efficient convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def export():
    """Export the model to TorchScript format."""
    print("=" * 60)
    print("AutoRL Model Export for Arm Mobile Devices")
    print("=" * 60)
    
    # Create model instance
    model = SmallPerceptionModel(num_classes=10)
    model.eval()
    
    print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create example input (batch_size=1, channels=3, height=224, width=224)
    example_input = torch.randn(1, 3, 224, 224)
    
    print("✓ Example input created: shape", example_input.shape)
    
    # Trace the model
    print("\n⏳ Tracing model...")
    try:
        traced_model = torch.jit.trace(model, example_input)
        print("✓ Model traced successfully")
    except Exception as e:
        print(f"✗ Error tracing model: {e}")
        return False
    
    # Save the traced model
    output_path = os.path.join(os.path.dirname(__file__), "model_mobile.pt")
    try:
        traced_model.save(output_path)
        print(f"✓ Model saved to: {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Model size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False
    
    # Verify the saved model can be loaded
    print("\n⏳ Verifying saved model...")
    try:
        loaded_model = torch.jit.load(output_path)
        test_output = loaded_model(example_input)
        print(f"✓ Model verification successful")
        print(f"✓ Output shape: {test_output.shape}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Export completed successfully!")
    print("=" * 60)
    print(f"\nNext step: Run 'python3 model/quantize_model.py' to quantize the model")
    
    return True


if __name__ == "__main__":
    success = export()
    sys.exit(0 if success else 1)
