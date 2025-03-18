#!/usr/bin/env python3
"""
Script to display system information relevant to Metal GPU acceleration.
"""

import os
import sys
import platform
import subprocess
import psutil
import re
from datetime import datetime

def print_colored(text, color_code):
    """Print colored text to the console."""
    print(f"\033[{color_code}m{text}\033[0m")

def print_header(text):
    """Print a header in blue."""
    print("\n" + "=" * 50)
    print_colored(text.center(50), "1;34")
    print("=" * 50)

def print_section(text):
    """Print a section header in cyan."""
    print("\n" + "-" * 50)
    print_colored(text, "1;36")
    print("-" * 50)

def print_info(text):
    """Print information text in normal color."""
    print(f"  {text}")

def print_key_value(key, value):
    """Print a key-value pair with the key in yellow."""
    print_colored(f"  {key}:", "1;33")
    print(f" {value}")

def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_macos_version():
    """Get detailed macOS version information."""
    macos_version = platform.mac_ver()[0]
    macos_name = ""
    
    # Map of macOS version numbers to names
    version_names = {
        "10.15": "Catalina",
        "11": "Big Sur",
        "12": "Monterey",
        "13": "Ventura",
        "14": "Sonoma"
    }
    
    # Find matching version name
    for version, name in version_names.items():
        if macos_version.startswith(version):
            macos_name = name
            break
    
    if macos_name:
        return f"{macos_version} ({macos_name})"
    else:
        return macos_version

def get_chip_model():
    """Get Apple Silicon chip model (M1, M2, etc.)."""
    try:
        sysctl_output = run_command("sysctl -n machdep.cpu.brand_string")
        chip_match = re.search(r'Apple (M\d+)(?:\s+(Pro|Max|Ultra))?', sysctl_output)
        if chip_match:
            chip = chip_match.group(1)
            variant = chip_match.group(2) if chip_match.group(2) else ""
            return f"{chip} {variant}".strip()
        else:
            return "Unknown Apple Silicon"
    except Exception:
        return "Could not determine chip model"

def get_gpu_info():
    """Get information about the GPU."""
    try:
        # Check if system_profiler is available
        if platform.system() == "Darwin":
            # For macOS, use system_profiler to get GPU information
            gpu_info = run_command("system_profiler SPDisplaysDataType | grep -A 10 'Chipset Model'")
            return gpu_info
        else:
            return "GPU information not available on this platform"
    except Exception as e:
        return f"Error retrieving GPU information: {str(e)}"

def get_memory_info():
    """Get system memory information."""
    try:
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)
        available_gb = memory.available / (1024 ** 3)
        used_gb = memory.used / (1024 ** 3)
        percent = memory.percent
        
        return {
            "total": f"{total_gb:.2f} GB",
            "available": f"{available_gb:.2f} GB",
            "used": f"{used_gb:.2f} GB",
            "percent": f"{percent}%"
        }
    except Exception as e:
        return f"Error retrieving memory information: {str(e)}"

def get_cpu_info():
    """Get CPU information."""
    try:
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        # Get CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["current_frequency"] = f"{cpu_freq.current:.2f} MHz"
                if cpu_freq.max:
                    cpu_info["max_frequency"] = f"{cpu_freq.max:.2f} MHz"
        except Exception:
            pass
            
        return cpu_info
    except Exception as e:
        return f"Error retrieving CPU information: {str(e)}"

def get_python_info():
    """Get Python version and environment information."""
    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
    }
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        python_info["virtual_env"] = "Yes (path: " + sys.prefix + ")"
    else:
        python_info["virtual_env"] = "No"
        
    return python_info

def get_pytorch_info():
    """Get PyTorch information if installed."""
    try:
        import torch
        
        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False,
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A",
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "mps_built": torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False,
        }
        
        # Get device information
        if torch_info["mps_available"]:
            torch_info["default_device"] = "mps"
        elif torch_info["cuda_available"]:
            torch_info["default_device"] = "cuda"
            torch_info["device_count"] = torch.cuda.device_count()
            torch_info["current_device"] = torch.cuda.current_device()
            torch_info["device_name"] = torch.cuda.get_device_name(0)
        else:
            torch_info["default_device"] = "cpu"
            
        return torch_info
    except ImportError:
        return "PyTorch is not installed"
    except Exception as e:
        return f"Error retrieving PyTorch information: {str(e)}"

def get_metal_env_vars():
    """Get Metal-related environment variables."""
    metal_vars = {
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "Not set"),
        "METAL_DEVICE_WRAPPING_ENABLED": os.environ.get("METAL_DEVICE_WRAPPING_ENABLED", "Not set"),
        "MPS_FALLBACK_ENABLED": os.environ.get("MPS_FALLBACK_ENABLED", "Not set"),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "Not set"),
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "Not set"),
        "PYTORCH_MPS_ALLOCATOR_POLICY": os.environ.get("PYTORCH_MPS_ALLOCATOR_POLICY", "Not set")
    }
    return metal_vars

def main():
    """Display system information."""
    print_header("Metal GPU Acceleration System Information")
    print_info(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System information
    print_section("System Information")
    print_key_value("OS", f"{platform.system()} {platform.release()}")
    print_key_value("macOS Version", get_macos_version())
    print_key_value("Hostname", platform.node())
    print_key_value("Processor", platform.processor())
    print_key_value("Chip Model", get_chip_model())
    
    # CPU information
    print_section("CPU Information")
    cpu_info = get_cpu_info()
    if isinstance(cpu_info, dict):
        print_key_value("Physical Cores", cpu_info.get("physical_cores", "Unknown"))
        print_key_value("Logical Cores", cpu_info.get("logical_cores", "Unknown"))
        print_key_value("Current CPU Usage", f"{cpu_info.get('cpu_percent', 'Unknown')}%")
        if "current_frequency" in cpu_info:
            print_key_value("Current Frequency", cpu_info.get("current_frequency"))
        if "max_frequency" in cpu_info:
            print_key_value("Max Frequency", cpu_info.get("max_frequency"))
    else:
        print_info(cpu_info)
    
    # Memory information
    print_section("Memory Information")
    memory_info = get_memory_info()
    if isinstance(memory_info, dict):
        print_key_value("Total Memory", memory_info.get("total", "Unknown"))
        print_key_value("Available Memory", memory_info.get("available", "Unknown"))
        print_key_value("Used Memory", memory_info.get("used", "Unknown"))
        print_key_value("Memory Usage", memory_info.get("percent", "Unknown"))
    else:
        print_info(memory_info)
    
    # GPU information
    print_section("GPU Information")
    gpu_info = get_gpu_info()
    if gpu_info:
        print_info(gpu_info)
    else:
        print_info("No GPU information available")
    
    # Python information
    print_section("Python Environment")
    python_info = get_python_info()
    print_key_value("Python Version", python_info.get("version", "Unknown"))
    print_key_value("Implementation", python_info.get("implementation", "Unknown"))
    print_key_value("Compiler", python_info.get("compiler", "Unknown"))
    print_key_value("Executable", python_info.get("executable", "Unknown"))
    print_key_value("Virtual Environment", python_info.get("virtual_env", "Unknown"))
    
    # PyTorch information
    print_section("PyTorch Information")
    pytorch_info = get_pytorch_info()
    if isinstance(pytorch_info, dict):
        print_key_value("PyTorch Version", pytorch_info.get("version", "Unknown"))
        print_key_value("MPS Available", pytorch_info.get("mps_available", "Unknown"))
        print_key_value("MPS Built", pytorch_info.get("mps_built", "Unknown"))
        print_key_value("CUDA Available", pytorch_info.get("cuda_available", "Unknown"))
        print_key_value("CUDA Version", pytorch_info.get("cuda_version", "Unknown"))
        print_key_value("Default Device", pytorch_info.get("default_device", "Unknown"))
        
        if "device_count" in pytorch_info:
            print_key_value("CUDA Device Count", pytorch_info.get("device_count"))
        if "device_name" in pytorch_info:
            print_key_value("CUDA Device Name", pytorch_info.get("device_name"))
    else:
        print_info(pytorch_info)
    
    # Metal environment variables
    print_section("Metal Environment Variables")
    metal_vars = get_metal_env_vars()
    for var, value in metal_vars.items():
        print_key_value(var, value)
    
    print("\n" + "=" * 50)
    print_colored("End of System Information Report".center(50), "1;34")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main() 