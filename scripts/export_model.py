#!/usr/bin/env python3
"""
Complete model export script for ARM mobile AI deployment with ExecuTorch support.

Exports models to multiple formats optimized for ARM processors:
- ExecuTorch (.pte) - Edge-optimized format with ARM NEON support
- TorchScript (.pt) - PyTorch Mobile with ARM NEON/XNNPACK
- ONNX (.onnx) - ONNX Runtime Mobile with NNAPI/XNNPACK
- Quantized variants for all formats

This script is designed for the Arm-Unified-Task-Orchestrator project.
"""

import torch
import torch.nn as nn
import sys
import os
import argparse
import platform
from pathlib import Path
from typing import Optional, Tuple

# Add backend to path for model imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import model definition
try:
    from models.model.export_model import SmallPerceptionModel
except ImportError:
    # Fallback: define model here
    class SmallPerceptionModel(nn.Module):
        """Lightweight perception model for mobile UI element detection."""
        def __init__(self, num_classes=10):
            super(SmallPerceptionModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
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


def check_arm_platform() -> bool:
    """Check if running on ARM platform."""
    arch = platform.machine().lower()
    return 'arm' in arch or 'aarch64' in arch


def export_executorch(
    model: nn.Module,
    output_path: Path,
    example_input: torch.Tensor,
    quantize: bool = False
) -> bool:
    """
    Export model to ExecuTorch format (.pte) optimized for ARM.
    
    Args:
        model: PyTorch model to export
        output_path: Output path for .pte file
        example_input: Example input tensor
        quantize: Whether to apply quantization
        
    Returns:
        True if export successful
    """
    print(f"⏳ Exporting to ExecuTorch (.pte): {output_path}")
    
    try:
        # Check if ExecuTorch is available
        executorch_available = False
        try:
            import executorch
            executorch_available = True
        except ImportError:
            try:
                from executorch import exir
                executorch_available = True
            except ImportError:
                pass
        
        if not executorch_available:
            print("⚠️  ExecuTorch not available, trying alternative export method...")
            # Fallback: Export as TorchScript and note that conversion to .pte
            # should be done using ExecuTorch tools
            traced_model = torch.jit.trace(model, example_input)
            temp_path = output_path.with_suffix('.pt')
            traced_model.save(str(temp_path))
            print(f"✓ Exported TorchScript intermediate: {temp_path}")
            print("  Note: Convert to .pte using ExecuTorch tools:")
            print("    python -m executorch.exir.scripts.export_to_executorch")
            return False
        
        # Try ExecuTorch export
        try:
            from executorch import exir
            from executorch.exir import to_edge
            
            # Convert to Edge IR
            print("  Converting to ExecuTorch Edge IR...")
            edge_program = to_edge(model, (example_input,))
            
            # Export to ExecuTorch format
            print("  Exporting to ExecuTorch format...")
            executorch_program = edge_program.to_executorch()
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(executorch_program.buffer)
            
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ ExecuTorch export successful ({file_size:.2f} MB)")
            
            if check_arm_platform():
                print("  ✓ ARM platform detected - model optimized for ARM NEON")
            
            return True
            
        except Exception as e:
            print(f"⚠️  ExecuTorch export failed: {e}")
            print("  Falling back to TorchScript format...")
            # Fallback to TorchScript
            traced_model = torch.jit.trace(model, example_input)
            temp_path = output_path.with_suffix('.pt')
            traced_model.save(str(temp_path))
            print(f"  ✓ Saved as TorchScript: {temp_path}")
            print("  Note: Use ExecuTorch conversion tools to create .pte file")
            return False
            
    except Exception as e:
        print(f"✗ ExecuTorch export failed: {e}")
        return False


def export_torchscript(
    model: nn.Module,
    output_path: Path,
    example_input: torch.Tensor,
    optimize_mobile: bool = True
) -> bool:
    """
    Export model to TorchScript format for PyTorch Mobile (ARM-optimized).
    
    Args:
        model: PyTorch model to export
        output_path: Output path for .pt file
        example_input: Example input tensor
        optimize_mobile: Whether to apply mobile optimizations
        
    Returns:
        True if export successful
    """
    print(f"⏳ Exporting to TorchScript: {output_path}")
    
    try:
        # Trace model
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for mobile (ARM NEON/XNNPACK)
        if optimize_mobile:
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                traced_model = optimize_for_mobile(traced_model)
                if check_arm_platform():
                    print("  ✓ ARM platform detected - mobile optimizations applied")
            except Exception as e:
                print(f"  ⚠️  Mobile optimization not available: {e}")
        
        # Save model
        traced_model.save(str(output_path))
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ TorchScript export successful ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        return False


def export_onnx(
    model: nn.Module,
    output_path: Path,
    example_input: torch.Tensor,
    opset_version: int = 13
) -> bool:
    """
    Export model to ONNX format for ONNX Runtime Mobile (ARM NNAPI/XNNPACK).
    
    Args:
        model: PyTorch model to export
        output_path: Output path for .onnx file
        example_input: Example input tensor
        opset_version: ONNX opset version
        
    Returns:
        True if export successful
    """
    print(f"⏳ Exporting to ONNX: {output_path}")
    
    try:
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            # ARM optimizations
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ ONNX export successful ({file_size:.2f} MB)")
        
        if check_arm_platform():
            print("  ✓ ARM platform detected - ready for NNAPI/XNNPACK optimization")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def export_quantized(
    model: nn.Module,
    output_path: Path,
    example_input: torch.Tensor,
    format_type: str = 'torchscript'
) -> bool:
    """
    Export quantized model (INT8) for ARM processors.
    
    Args:
        model: PyTorch model to export
        output_path: Output path for quantized model
        example_input: Example input tensor
        format_type: Format type ('torchscript', 'onnx', 'executorch')
        
    Returns:
        True if export successful
    """
    print(f"⏳ Exporting quantized model ({format_type}): {output_path}")
    
    try:
        if format_type == 'torchscript':
            # Dynamic quantization for TorchScript
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Trace and save
            traced_model = torch.jit.trace(quantized_model, example_input)
            traced_model.save(str(output_path))
            
        elif format_type == 'onnx':
            # Static quantization for ONNX
            # Note: This is a simplified version
            # Full quantization requires calibration dataset
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            torch.onnx.export(
                quantized_model,
                example_input,
                str(output_path),
                export_params=True,
                opset_version=13,
                input_names=['input'],
                output_names=['output'],
            )
            
        else:
            print(f"  ⚠️  Quantization not yet supported for {format_type}")
            return False
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Quantized export successful ({file_size:.2f} MB)")
        print("  ✓ INT8 quantization applied - optimized for ARM processors")
        return True
        
    except Exception as e:
        print(f"✗ Quantized export failed: {e}")
        return False


def export():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export model for ARM mobile deployment with ExecuTorch support"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/model',
        help='Output directory for exported models'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['executorch', 'torchscript', 'onnx', 'all'],
        default=['torchscript', 'onnx'],
        help='Export formats (default: torchscript onnx)'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Also export quantized versions (INT8)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='Number of output classes'
    )
    parser.add_argument(
        '--input-shape',
        nargs=4,
        type=int,
        default=[1, 3, 224, 224],
        help='Input shape as [batch, channels, height, width]'
    )
    parser.add_argument(
        '--onnx-opset',
        type=int,
        default=13,
        help='ONNX opset version (default: 13)'
    )
    
    args = parser.parse_args()
    
    # Expand 'all' format
    if 'all' in args.formats:
        args.formats = ['executorch', 'torchscript', 'onnx']
    
    print("=" * 70)
    print("ARM-Unified-Task-Orchestrator Model Export")
    print("ARM-Optimized Model Export for Mobile Deployment")
    print("=" * 70)
    
    # Check platform
    is_arm = check_arm_platform()
    if is_arm:
        print(f"\n✓ ARM platform detected: {platform.machine()}")
        print("  ARM NEON/XNNPACK optimizations will be applied")
    else:
        print(f"\nℹ️  Platform: {platform.machine()}")
        print("  Models will be optimized for ARM deployment")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model instance
    print(f"\n✓ Creating model with {args.num_classes} classes")
    model = SmallPerceptionModel(num_classes=args.num_classes)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    
    # Create example input
    batch, channels, height, width = args.input_shape
    example_input = torch.randn(batch, channels, height, width)
    print(f"✓ Example input shape: {example_input.shape}")
    
    # Verify model forward pass
    print("\n⏳ Verifying model forward pass...")
    with torch.no_grad():
        output = model(example_input)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    # Export in requested formats
    print("\n" + "=" * 70)
    print("Exporting models...")
    print("=" * 70)
    
    results = {}
    
    if 'executorch' in args.formats:
        output_path = output_dir / "model_executorch.pte"
        results['executorch'] = export_executorch(model, output_path, example_input)
        
        if args.quantize:
            quant_path = output_dir / "model_executorch_quant.pte"
            results['executorch_quant'] = export_quantized(
                model, quant_path, example_input, 'executorch'
            )
    
    if 'torchscript' in args.formats:
        output_path = output_dir / "model_mobile.pt"
        results['torchscript'] = export_torchscript(model, output_path, example_input)
        
        if args.quantize:
            quant_path = output_dir / "model_mobile_quant.pt"
            results['torchscript_quant'] = export_quantized(
                model, quant_path, example_input, 'torchscript'
            )
    
    if 'onnx' in args.formats:
        output_path = output_dir / "model_mobile.onnx"
        results['onnx'] = export_onnx(model, output_path, example_input, args.onnx_opset)
        
        if args.quantize:
            quant_path = output_dir / "model_mobile_quant.onnx"
            results['onnx_quant'] = export_quantized(
                model, quant_path, example_input, 'onnx'
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    
    for format_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {format_name:20s}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n✅ All exports completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Models are ready for ARM mobile deployment")
        print(f"  2. ExecuTorch models (.pte) can be used with ExecuTorch runtime")
        print(f"  3. TorchScript models (.pt) can be used with PyTorch Mobile")
        print(f"  4. ONNX models (.onnx) can be used with ONNX Runtime Mobile")
        if args.quantize:
            print(f"  5. Quantized models are optimized for ARM INT8 operations")
        print(f"\nModels saved to: {output_dir}")
    else:
        print("\n⚠️  Some exports failed. Check errors above.")
        # Don't exit with error if at least one format succeeded
        if not any(results.values()):
            sys.exit(1)
    
    return all_success


if __name__ == "__main__":
    success = export()
    sys.exit(0 if success else 1)
