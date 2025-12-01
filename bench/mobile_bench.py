#!/usr/bin/env python3
"""
Mobile AI benchmark harness for Arm devices.

Inspired by mobile-ai-bench patterns for standardized performance metrics.
Measures inference latency, memory usage, and throughput.

References:
- https://github.com/XiaoMi/mobile-ai-bench
"""

import time
import statistics
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from inference.runtime import Runtime
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False
    Runtime = None


def measure_latency(
    runtime: Runtime,
    input_shape: tuple,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Measure inference latency statistics.
    
    Args:
        runtime: Runtime instance
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with latency statistics
    """
    latencies_ms = []
    
    # Create input tensor
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    print(f"Warming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        try:
            _ = runtime.run(dummy_input)
        except Exception as e:
            print(f"Warning: Warmup run failed: {e}")
            break
    
    # Benchmark
    print(f"Running {num_runs} benchmark iterations...")
    for i in range(num_runs):
        start_time = time.perf_counter()
        try:
            _ = runtime.run(dummy_input)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies_ms.append(latency_ms)
        except Exception as e:
            print(f"Warning: Benchmark run {i+1} failed: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_runs} runs...")
    
    if not latencies_ms:
        raise RuntimeError("All benchmark runs failed")
    
    # Calculate statistics
    latencies_ms.sort()
    
    stats = {
        'mean_ms': statistics.mean(latencies_ms),
        'median_ms': statistics.median(latencies_ms),
        'min_ms': min(latencies_ms),
        'max_ms': max(latencies_ms),
        'std_ms': statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        'p50_ms': latencies_ms[len(latencies_ms) // 2],
        'p90_ms': latencies_ms[int(len(latencies_ms) * 0.9)] if len(latencies_ms) >= 10 else latencies_ms[-1],
        'p95_ms': latencies_ms[int(len(latencies_ms) * 0.95)] if len(latencies_ms) >= 20 else latencies_ms[-1],
        'p99_ms': latencies_ms[int(len(latencies_ms) * 0.99)] if len(latencies_ms) >= 100 else latencies_ms[-1],
        'num_runs': len(latencies_ms),
    }
    
    return stats


def benchmark_model(
    model_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
    prefer_backend: Optional[str] = None
) -> Dict[str, Any]:
    """
    Benchmark a model's inference performance.
    
    Args:
        model_path: Path to model file
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        prefer_backend: Preferred backend ('executorch', 'onnx', 'pytorch')
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 70)
    print("Mobile AI Benchmark")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Input shape: {input_shape}")
    print(f"Runs: {num_runs} (warmup: {warmup_runs})")
    
    if not RUNTIME_AVAILABLE:
        raise RuntimeError("Runtime abstraction not available")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load runtime
    print(f"\nLoading model...")
    runtime = Runtime.load(model_path, prefer=prefer_backend)
    print(f"Backend: {runtime.backend}")
    
    # Get model info
    model_info = runtime.get_info()
    
    # Measure latency
    print("\n" + "-" * 70)
    print("Latency Benchmark")
    print("-" * 70)
    latency_stats = measure_latency(runtime, input_shape, num_runs, warmup_runs)
    
    # Calculate throughput
    throughput_fps = 1000.0 / latency_stats['mean_ms'] if latency_stats['mean_ms'] > 0 else 0
    
    # Compile results
    results = {
        'model_path': model_path,
        'backend': runtime.backend,
        'model_info': model_info,
        'input_shape': input_shape,
        'latency_stats': latency_stats,
        'throughput_fps': throughput_fps,
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print(f"\nBackend: {runtime.backend}")
    print(f"\nLatency Statistics:")
    print(f"  Mean:    {latency_stats['mean_ms']:.2f} ms")
    print(f"  Median:  {latency_stats['median_ms']:.2f} ms")
    print(f"  Min:     {latency_stats['min_ms']:.2f} ms")
    print(f"  Max:     {latency_stats['max_ms']:.2f} ms")
    print(f"  Std:     {latency_stats['std_ms']:.2f} ms")
    print(f"  P50:     {latency_stats['p50_ms']:.2f} ms")
    print(f"  P90:     {latency_stats['p90_ms']:.2f} ms")
    print(f"  P95:     {latency_stats['p95_ms']:.2f} ms")
    print(f"  P99:     {latency_stats['p99_ms']:.2f} ms")
    print(f"\nThroughput: {throughput_fps:.2f} FPS")
    
    return results


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark mobile AI models")
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to model file'
    )
    parser.add_argument(
        '--input-shape',
        nargs=4,
        type=int,
        default=[1, 3, 224, 224],
        help='Input shape [batch, channels, height, width]'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of benchmark runs'
    )
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=10,
        help='Number of warmup runs'
    )
    parser.add_argument(
        '--backend',
        choices=['executorch', 'onnx', 'pytorch'],
        default=None,
        help='Preferred backend (auto-detect if not specified)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    try:
        results = benchmark_model(
            args.model_path,
            tuple(args.input_shape),
            args.num_runs,
            args.warmup_runs,
            args.backend
        )
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

