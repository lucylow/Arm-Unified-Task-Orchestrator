"""
ARM Device Detection Module

Detects ARM architecture, chipset information, and hardware capabilities
for optimizing AI inference on ARM-based mobile devices.
"""

import platform
import subprocess
import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ARMDeviceDetector:
    """Detects and reports ARM device capabilities"""
    
    def __init__(self):
        self.device_info = self._detect_device_info()
        
    def _detect_device_info(self) -> Dict:
        """Detect comprehensive ARM device information"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'is_arm': self._is_arm_architecture(),
            'architecture': self._get_arm_architecture(),
            'chipset': self._get_chipset_name(),
            'cpu_cores': self._get_cpu_cores(),
            'cpu_frequencies': self._get_cpu_frequencies(),
            'has_neon': self._has_neon_support(),
            'has_npu': self._has_npu_support(),
            'has_gpu': self._has_gpu_support(),
            'ram_total_mb': self._get_total_ram(),
        }
        
        logger.info(f"Detected ARM device: {info}")
        return info
    
    def _is_arm_architecture(self) -> bool:
        """Check if running on ARM architecture"""
        machine = platform.machine().lower()
        return any(arch in machine for arch in ['arm', 'aarch64', 'arm64'])
    
    def _get_arm_architecture(self) -> str:
        """Get ARM architecture version"""
        machine = platform.machine().lower()
        
        if 'aarch64' in machine or 'arm64' in machine:
            return 'ARM64 (ARMv8)'
        elif 'armv7' in machine:
            return 'ARM32 (ARMv7)'
        elif 'arm' in machine:
            return 'ARM (Unknown version)'
        else:
            return 'Non-ARM'
    
    def _get_chipset_name(self) -> str:
        """Attempt to detect chipset name (Android-specific)"""
        try:
            # Try to read from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            # Look for Hardware or Model name
            hardware_match = re.search(r'Hardware\s*:\s*(.+)', cpuinfo)
            if hardware_match:
                return hardware_match.group(1).strip()
            
            model_match = re.search(r'Model\s*:\s*(.+)', cpuinfo)
            if model_match:
                return model_match.group(1).strip()
                
        except Exception as e:
            logger.debug(f"Could not detect chipset: {e}")
        
        # Fallback to common ARM chipsets based on CPU info
        return self._guess_chipset_from_cpu()
    
    def _guess_chipset_from_cpu(self) -> str:
        """Guess chipset based on CPU configuration"""
        cores = self._get_cpu_cores()
        
        # Common ARM configurations
        if cores == 8:
            return "Snapdragon 8-series or equivalent (8 cores)"
        elif cores == 6:
            return "MediaTek Dimensity or equivalent (6 cores)"
        elif cores == 4:
            return "Snapdragon 6-series or equivalent (4 cores)"
        else:
            return f"ARM Processor ({cores} cores)"
    
    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores"""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except Exception:
            return 4  # Default fallback
    
    def _get_cpu_frequencies(self) -> Dict[str, float]:
        """Get CPU frequency information"""
        try:
            # Try to read from /sys/devices/system/cpu/
            frequencies = {}
            cores = self._get_cpu_cores()
            
            for i in range(cores):
                try:
                    with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq', 'r') as f:
                        freq_khz = int(f.read().strip())
                        frequencies[f'cpu{i}'] = freq_khz / 1000  # Convert to MHz
                except Exception:
                    pass
            
            if frequencies:
                return {
                    'current': frequencies,
                    'max': max(frequencies.values()),
                    'min': min(frequencies.values()),
                }
        except Exception as e:
            logger.debug(f"Could not read CPU frequencies: {e}")
        
        return {'current': {}, 'max': 0, 'min': 0}
    
    def _has_neon_support(self) -> bool:
        """Check if ARM NEON SIMD is supported"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return 'neon' in cpuinfo or 'asimd' in cpuinfo
        except Exception:
            # Assume NEON is available on ARMv8+
            return 'arm64' in platform.machine().lower()
    
    def _has_npu_support(self) -> bool:
        """Check if Neural Processing Unit (NPU) is available"""
        try:
            # Check for common NPU device files
            import os
            npu_paths = [
                '/dev/npu',
                '/dev/mali',
                '/sys/class/npu',
            ]
            return any(os.path.exists(path) for path in npu_paths)
        except Exception:
            return False
    
    def _has_gpu_support(self) -> bool:
        """Check if GPU is available"""
        try:
            # Check for common GPU device files
            import os
            gpu_paths = [
                '/dev/mali0',
                '/dev/kgsl-3d0',  # Qualcomm Adreno
                '/dev/pvr_sync',  # PowerVR
            ]
            return any(os.path.exists(path) for path in gpu_paths)
        except Exception:
            return False
    
    def _get_total_ram(self) -> int:
        """Get total RAM in MB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                match = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
                if match:
                    return int(match.group(1)) // 1024  # Convert to MB
        except Exception:
            pass
        
        return 0  # Unknown
    
    def get_device_info(self) -> Dict:
        """Get complete device information"""
        return self.device_info
    
    def is_arm_device(self) -> bool:
        """Check if device is ARM-based"""
        return self.device_info['is_arm']
    
    def get_optimization_flags(self) -> Dict[str, bool]:
        """Get recommended optimization flags for this device"""
        return {
            'use_neon': self.device_info['has_neon'],
            'use_npu': self.device_info['has_npu'],
            'use_gpu': self.device_info['has_gpu'],
            'use_quantization': True,  # Always recommended for ARM
            'use_memory_mapping': self.device_info['ram_total_mb'] < 4096,
        }
    
    def get_device_summary(self) -> str:
        """Get human-readable device summary"""
        info = self.device_info
        summary = f"""
ARM Device Information:
-----------------------
Architecture: {info['architecture']}
Chipset: {info['chipset']}
CPU Cores: {info['cpu_cores']}
RAM: {info['ram_total_mb']} MB
NEON Support: {'Yes' if info['has_neon'] else 'No'}
NPU Available: {'Yes' if info['has_npu'] else 'No'}
GPU Available: {'Yes' if info['has_gpu'] else 'No'}
"""
        return summary.strip()


# Singleton instance
_detector_instance: Optional[ARMDeviceDetector] = None


def get_arm_device_detector() -> ARMDeviceDetector:
    """Get singleton ARM device detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ARMDeviceDetector()
    return _detector_instance


if __name__ == '__main__':
    # Test device detection
    detector = ARMDeviceDetector()
    print(detector.get_device_summary())
    print("\nOptimization Flags:")
    for flag, value in detector.get_optimization_flags().items():
        print(f"  {flag}: {value}")
