#!/usr/bin/env python3
"""
GPU Acceleration Controller Module.

This module provides centralized control over GPU acceleration settings
for Metal on Apple Silicon Macs. It provides functions to enable/disable
GPU acceleration and configure various optimization parameters.
"""

import os
import sys
import platform
import subprocess
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_controller")

# Constants for Metal GPU Acceleration
ENV_METAL_ENABLED = "ENABLE_METAL"
ENV_METAL_LAYERS = "METAL_LAYERS"
ENV_METAL_MEMORY_LIMIT = "METAL_MEMORY_LIMIT"
ENV_METAL_BUFFER_SIZE = "METAL_BUFFER_SIZE"
ENV_OPTIMIZE_FOR_APPLE = "OPTIMIZE_FOR_APPLE"
ENV_DEFAULT_CONTEXT_SIZE = "DEFAULT_CONTEXT_SIZE"

# Default settings by chip type
M1_DEFAULT_SETTINGS = {
    ENV_METAL_ENABLED: "1",
    ENV_METAL_LAYERS: "24",  # More conservative for M1
    ENV_METAL_MEMORY_LIMIT: "4096",  # 4GB limit for most M1 devices
    ENV_METAL_BUFFER_SIZE: "512",
    ENV_OPTIMIZE_FOR_APPLE: "1",
    ENV_DEFAULT_CONTEXT_SIZE: "2048"
}

M2_DEFAULT_SETTINGS = {
    ENV_METAL_ENABLED: "1",
    ENV_METAL_LAYERS: "28",  # More layers for M2
    ENV_METAL_MEMORY_LIMIT: "6144",  # 6GB limit for most M2 devices
    ENV_METAL_BUFFER_SIZE: "1024",
    ENV_OPTIMIZE_FOR_APPLE: "1",
    ENV_DEFAULT_CONTEXT_SIZE: "4096"
}

M3_DEFAULT_SETTINGS = {
    ENV_METAL_ENABLED: "1",
    ENV_METAL_LAYERS: "32",  # Maximum layers for M3
    ENV_METAL_MEMORY_LIMIT: "8192",  # 8GB limit for most M3 devices
    ENV_METAL_BUFFER_SIZE: "2048",
    ENV_OPTIMIZE_FOR_APPLE: "1",
    ENV_DEFAULT_CONTEXT_SIZE: "8192"
}

CPU_ONLY_SETTINGS = {
    ENV_METAL_ENABLED: "0",
    ENV_METAL_LAYERS: "0",
    ENV_METAL_MEMORY_LIMIT: "0",
    ENV_METAL_BUFFER_SIZE: "0",
    ENV_OPTIMIZE_FOR_APPLE: "0"
}

class GPUController:
    """Controller for GPU acceleration settings and operations."""
    
    def __init__(self):
        """Initialize the GPU controller with default settings"""
        self.is_apple_silicon = self._detect_apple_silicon()
        self.chip_model = self._detect_chip_model() if self.is_apple_silicon else ""
        self.current_settings = self._get_current_settings()
        
        # Initialize GPU if available
        self._initialize_gpu()
        
        # Log initialization
        if self.is_apple_silicon:
            logger.info(f"GPU Controller initialized. Detected: {self.chip_model}")
        else:
            logger.info("GPU Controller initialized. No Apple Silicon detected.")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if the system is running on Apple Silicon."""
        return (
            platform.system() == "Darwin" and
            platform.machine() == "arm64"
        )
    
    def _detect_chip_model(self) -> str:
        """Detect the Apple Silicon chip model (M1, M2, M3, etc.)."""
        if not self.is_apple_silicon:
            return "Not Apple Silicon"
        
        try:
            sysctl_output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True
            ).strip()
            
            if "Apple M1" in sysctl_output:
                return "Apple M1"
            elif "Apple M2" in sysctl_output:
                return "Apple M2"
            elif "Apple M3" in sysctl_output:
                return "Apple M3"
            elif "Apple M4" in sysctl_output:
                return "Apple M4"
            else:
                return sysctl_output
        except Exception as e:
            logger.error(f"Error detecting chip model: {str(e)}")
            return "Unknown Apple Silicon"
    
    def _get_current_settings(self) -> Dict[str, str]:
        """Get the current GPU acceleration settings from environment variables."""
        return {
            ENV_METAL_ENABLED: os.environ.get(ENV_METAL_ENABLED, "0"),
            ENV_METAL_LAYERS: os.environ.get(ENV_METAL_LAYERS, "0"),
            ENV_METAL_MEMORY_LIMIT: os.environ.get(ENV_METAL_MEMORY_LIMIT, "0"),
            ENV_METAL_BUFFER_SIZE: os.environ.get(ENV_METAL_BUFFER_SIZE, "0"),
            ENV_OPTIMIZE_FOR_APPLE: os.environ.get(ENV_OPTIMIZE_FOR_APPLE, "0"),
            ENV_DEFAULT_CONTEXT_SIZE: os.environ.get(ENV_DEFAULT_CONTEXT_SIZE, "2048")
        }
    
    def get_optimal_settings(self) -> Dict[str, str]:
        """Get the optimal GPU settings based on detected hardware."""
        if not self.is_apple_silicon:
            return CPU_ONLY_SETTINGS
        
        if "M1" in self.chip_model:
            return M1_DEFAULT_SETTINGS
        elif "M2" in self.chip_model:
            return M2_DEFAULT_SETTINGS
        elif "M3" in self.chip_model or "M4" in self.chip_model:
            return M3_DEFAULT_SETTINGS
        else:
            # Default to M1 settings for unknown Apple Silicon
            return M1_DEFAULT_SETTINGS
    
    def apply_settings(self, settings: Dict[str, str]) -> None:
        """Apply GPU acceleration settings to environment variables."""
        for key, value in settings.items():
            os.environ[key] = value
        
        # Update current settings
        self.current_settings = self._get_current_settings()
        logger.info(f"Applied GPU settings: {settings}")
    
    def enable_gpu_acceleration(self, mode: str = "optimal") -> None:
        """
        Enable GPU acceleration with the specified mode.
        
        Args:
            mode: One of "optimal", "basic", "advanced", or "custom".
        """
        if not self.is_apple_silicon:
            logger.warning("Cannot enable GPU acceleration on non-Apple Silicon device")
            return
        
        if mode == "optimal":
            settings = self.get_optimal_settings()
        elif mode == "basic":
            # Basic mode uses fewer GPU layers
            settings = self.get_optimal_settings()
            settings[ENV_METAL_LAYERS] = str(max(1, int(settings[ENV_METAL_LAYERS]) - 8))
            settings[ENV_METAL_MEMORY_LIMIT] = str(int(int(settings[ENV_METAL_MEMORY_LIMIT]) * 0.75))
        elif mode == "advanced":
            # Advanced mode uses more aggressive settings
            settings = self.get_optimal_settings()
            settings[ENV_METAL_LAYERS] = str(max(1, int(settings[ENV_METAL_LAYERS]) + 1))
            settings[ENV_METAL_MEMORY_LIMIT] = str(int(int(settings[ENV_METAL_MEMORY_LIMIT]) * 1.25))
        elif mode == "custom":
            # Custom mode uses current settings
            settings = self.current_settings
            settings[ENV_METAL_ENABLED] = "1"
        else:
            logger.error(f"Unknown GPU acceleration mode: {mode}")
            return
        
        self.apply_settings(settings)
        logger.info(f"Enabled GPU acceleration with {mode} mode")
    
    def disable_gpu_acceleration(self) -> None:
        """Disable GPU acceleration."""
        self.apply_settings(CPU_ONLY_SETTINGS)
        logger.info("Disabled GPU acceleration")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get detailed system information including GPU capability.
        
        Returns:
            Dictionary containing system information.
        """
        # Common info
        info = {
            "os_version": platform.platform(),
            "python_version": sys.version,
            "memory_info": self._get_memory_info(),
            "metal_enabled": self.current_settings[ENV_METAL_ENABLED] == "1",
            "gpu_mode": "Disabled" if self.current_settings[ENV_METAL_ENABLED] == "0" else "Enabled",
            "gpu_layers": f"{self.current_settings[ENV_METAL_LAYERS]}/{self.current_settings[ENV_METAL_LAYERS]}" 
                          if self.current_settings[ENV_METAL_ENABLED] == "1" else "N/A"
        }
        
        # Apple Silicon specific info
        if self.is_apple_silicon:
            info["is_apple_silicon"] = True
            info["chip_model"] = self.chip_model
            
            try:
                # Get GPU info using system_profiler
                gpu_info_raw = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    text=True
                )
                gpu_info = json.loads(gpu_info_raw)
                
                if "SPDisplaysDataType" in gpu_info and len(gpu_info["SPDisplaysDataType"]) > 0:
                    displays = gpu_info["SPDisplaysDataType"][0]
                    if "spdisplays_metalFamilyName" in displays:
                        info["metal_family"] = displays["spdisplays_metalFamilyName"]
                    if "spdisplays_vram" in displays:
                        info["gpu_memory"] = displays["spdisplays_vram"]
            except Exception as e:
                logger.error(f"Error getting GPU info: {str(e)}")
        else:
            info["is_apple_silicon"] = False
            info["chip_model"] = platform.processor()
        
        return info
    
    def _get_memory_info(self) -> str:
        """Get system memory information."""
        try:
            if platform.system() == "Darwin":
                mem_info = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True
                ).strip()
                memory_gb = int(mem_info) / 1024 / 1024 / 1024
                return f"{memory_gb:.2f} GB"
            else:
                return "Unknown"
        except Exception as e:
            logger.error(f"Error getting memory info: {str(e)}")
            return "Unknown"
    
    def run_gpu_test(self) -> Dict[str, Any]:
        """
        Run a simple GPU test to verify acceleration is working.
        
        Returns:
            Dictionary containing test results.
        """
        import numpy as np
        
        # Skip if not Apple Silicon
        if not self.is_apple_silicon:
            return {
                "status": "skipped",
                "message": "GPU test requires Apple Silicon"
            }
        
        # Skip if Metal is not enabled
        if self.current_settings[ENV_METAL_ENABLED] != "1":
            return {
                "status": "skipped",
                "message": "GPU test requires Metal to be enabled"
            }
        
        # Try to import torch with Metal backend
        try:
            import torch
            
            # Check if Metal is available
            if not torch.backends.mps.is_available():
                return {
                    "status": "failed",
                    "message": "Metal is not available for PyTorch"
                }
            
            # Create tensors on CPU and MPS
            start_time = time.time()
            cpu_tensor = torch.randn(2000, 2000)
            
            # Move to MPS
            mps_device = torch.device("mps")
            mps_tensor = cpu_tensor.to(mps_device)
            
            # Simple matrix multiplication
            cpu_start = time.time()
            cpu_result = torch.matmul(cpu_tensor, cpu_tensor.t())
            cpu_time = time.time() - cpu_start
            
            # Run on GPU
            mps_start = time.time()
            mps_result = torch.matmul(mps_tensor, mps_tensor.t())
            # Force synchronization to get accurate timing
            mps_result.cpu()
            mps_time = time.time() - mps_start
            
            # Calculate speedup
            speedup = cpu_time / mps_time
            
            return {
                "status": "success",
                "cpu_time": cpu_time,
                "gpu_time": mps_time,
                "speedup": speedup,
                "matrix_size": 2000,
                "message": f"Metal GPU test completed. Speedup: {speedup:.2f}x"
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "PyTorch is required for GPU testing"
            }
        except Exception as e:
            logger.error(f"Error during GPU test: {str(e)}")
            return {
                "status": "error",
                "message": f"GPU test failed: {str(e)}"
            }
    
    def get_server_pid(self) -> Optional[int]:
        """
        Get the PID of the running server process.
        
        Returns:
            PID as integer or None if not found.
        """
        try:
            # Look for a running uvicorn process serving our app specifically
            if platform.system() == "Darwin":
                # Use ps command to find uvicorn processes
                output = subprocess.check_output(
                    ["ps", "-ef"],
                    text=True
                ).strip()
                
                # Look for lines containing both "uvicorn" and "src.app:app"
                for line in output.split('\n'):
                    if "uvicorn" in line and "src.app:app" in line:
                        # Extract PID which is the second field
                        fields = line.split()
                        if len(fields) >= 2:
                            try:
                                pid = int(fields[1])
                                logger.info(f"Found server process with PID: {pid}")
                                return pid
                            except ValueError:
                                pass
                
                logger.info("No server process found running src.app:app")
                # Fallback - check if this process is part of the server
                # If we're able to execute this code, the server must be running
                import os
                current_pid = os.getpid()
                logger.info(f"Using current process PID as fallback: {current_pid}")
                return current_pid
            else:
                # For non-Darwin platforms, use the original method
                output = subprocess.check_output(
                    ["pgrep", "-f", "uvicorn src.app:app"],
                    text=True
                ).strip()
                
                if output:
                    # Get the first PID
                    return int(output.split()[0])
                else:
                    import os
                    current_pid = os.getpid()
                    logger.info(f"Using current process PID as fallback: {current_pid}")
                    return current_pid
        except subprocess.CalledProcessError:
            # No process found - but we're running, so return current PID
            import os
            current_pid = os.getpid()
            logger.info(f"Using current process PID as fallback: {current_pid}")
            return current_pid
        except Exception as e:
            logger.error(f"Error getting server PID: {str(e)}")
            # If we're executing this code, the server must be running
            import os
            return os.getpid()
    
    def restart_server(self, mode: str = "advanced_gpu") -> Dict[str, Any]:
        """
        Restart the server with the specified GPU acceleration mode.
        
        Args:
            mode: One of "cpu", "basic_gpu", or "advanced_gpu".
            
        Returns:
            Dictionary with restart status.
        """
        # Map mode to script
        scripts = {
            "cpu": "./start_server.sh",
            "basic_gpu": "./restart_with_metal.sh",
            "advanced_gpu": "./restart_with_metal_advanced.sh"
        }
        
        if mode not in scripts:
            return {
                "status": "error",
                "message": f"Unknown server mode: {mode}"
            }
        
        # Check if the script exists
        if not os.path.isfile(scripts[mode]):
            return {
                "status": "error",
                "message": f"Restart script not found: {scripts[mode]}"
            }
        
        try:
            # Get current PID
            current_pid = self.get_server_pid()
            
            # Run the script in the background
            subprocess.Popen(
                [scripts[mode]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            
            return {
                "status": "restarting",
                "message": f"Server is restarting with {mode} mode",
                "previous_pid": current_pid
            }
        except Exception as e:
            logger.error(f"Error restarting server: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to restart server: {str(e)}"
            }
    
    def stop_server(self) -> Dict[str, Any]:
        """
        Stop the running server process.
        
        Returns:
            Dictionary with stop status.
        """
        pid = self.get_server_pid()
        
        if pid is None:
            return {
                "status": "not_running",
                "message": "Server is not running"
            }
        
        try:
            # Try to gracefully terminate the process
            os.kill(pid, 15)  # SIGTERM
            
            # Wait for the process to terminate
            for _ in range(10):
                time.sleep(0.5)
                if not self._is_process_running(pid):
                    return {
                        "status": "stopped",
                        "message": f"Server stopped (PID: {pid})"
                    }
            
            # Force kill if still running
            os.kill(pid, 9)  # SIGKILL
            return {
                "status": "force_stopped",
                "message": f"Server force stopped (PID: {pid})"
            }
        except Exception as e:
            logger.error(f"Error stopping server: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to stop server: {str(e)}"
            }
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def _initialize_gpu(self):
        """Initialize GPU resources if available"""
        if self.is_apple_silicon:
            try:
                # Try to import and initialize PyTorch MPS
                import torch
                if torch.backends.mps.is_available():
                    # Create a dummy tensor to initialize MPS
                    _ = torch.zeros(1, device="mps")
                    logger.info("Successfully initialized Metal GPU resources")
                else:
                    logger.info("Metal is not available for PyTorch")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not initialize Metal GPU: {str(e)}")

# Create a singleton instance
controller = GPUController()

if __name__ == "__main__":
    # Simple CLI interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Acceleration Controller")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--test", action="store_true", help="Run GPU test")
    parser.add_argument("--enable", choices=["optimal", "basic", "advanced"], help="Enable GPU acceleration")
    parser.add_argument("--disable", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    if args.info:
        info = controller.get_system_info()
        print(json.dumps(info, indent=2))
    
    if args.test:
        result = controller.run_gpu_test()
        print(json.dumps(result, indent=2))
    
    if args.enable:
        controller.enable_gpu_acceleration(args.enable)
        print(f"Enabled GPU acceleration with {args.enable} mode")
    
    if args.disable:
        controller.disable_gpu_acceleration()
        print("Disabled GPU acceleration") 