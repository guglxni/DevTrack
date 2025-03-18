#!/usr/bin/env python3
"""
Test script to verify Metal GPU acceleration is properly configured.
"""

import os
import sys
import time
import platform

def print_colored(text, color_code):
    """Print colored text to the console."""
    print(f"\033[{color_code}m{text}\033[0m")

def print_info(text):
    """Print information text in blue."""
    print_colored(text, "1;34")

def print_success(text):
    """Print success text in green."""
    print_colored(text, "1;32")

def print_warning(text):
    """Print warning text in yellow."""
    print_colored(text, "1;33")

def print_error(text):
    """Print error text in red."""
    print_colored(text, "1;31")

def check_platform():
    """Check if running on macOS with Apple Silicon."""
    system = platform.system()
    processor = platform.processor()
    
    if system != "Darwin":
        print_error(f"Current system is {system}, but Metal GPU acceleration requires macOS")
        return False
    
    if "arm" not in processor.lower():
        print_error(f"Current processor is {processor}, but Metal GPU acceleration requires Apple Silicon (ARM)")
        return False
    
    print_success(f"Running on compatible system: macOS {platform.mac_ver()[0]} with {processor}")
    return True

def check_environment():
    """Check if the necessary environment variables are set."""
    metal_vars = {
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        "METAL_DEVICE_WRAPPING_ENABLED": os.environ.get("METAL_DEVICE_WRAPPING_ENABLED", ""),
        "MPS_FALLBACK_ENABLED": os.environ.get("MPS_FALLBACK_ENABLED", "")
    }
    
    all_set = True
    for var, value in metal_vars.items():
        if value:
            print_success(f"✓ {var} is set to '{value}'")
        else:
            print_warning(f"✗ {var} is not set")
            all_set = False
    
    return all_set

def detect_gpu():
    """Try to detect and use the Metal GPU."""
    try:
        import torch
        
        print_info("\nChecking PyTorch installation:")
        print(f"PyTorch version: {torch.__version__}")
        
        # Check if MPS (Metal Performance Shaders) is available
        if not torch.backends.mps.is_available():
            print_error("MPS is not available. Metal GPU acceleration is not properly configured.")
            if torch.backends.mps.is_built():
                print_warning("MPS is built but not available. This could be due to an older macOS version.")
            else:
                print_error("PyTorch is not built with MPS support. Consider reinstalling PyTorch.")
            return False
        
        print_success("MPS is available!")
        
        # Try to create an MPS device and perform a simple operation
        print_info("\nTesting MPS device with a simple tensor operation:")
        device = torch.device("mps")
        
        # Start timer
        start_time = time.time()
        
        # Create tensors on MPS
        a = torch.ones(1000, 1000, device=device)
        b = torch.ones(1000, 1000, device=device)
        
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        
        # Force synchronization
        _ = c.cpu().numpy()
        
        end_time = time.time()
        
        print_success(f"✓ Successfully performed computation on MPS device")
        print_success(f"✓ Operation took {end_time - start_time:.4f} seconds")
        
        # Perform the same operation on CPU for comparison
        start_time = time.time()
        
        a_cpu = torch.ones(1000, 1000)
        b_cpu = torch.ones(1000, 1000)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        _ = c_cpu.numpy()
        
        end_time = time.time()
        
        print_info(f"Same operation on CPU took {end_time - start_time:.4f} seconds")
        
        return True
        
    except ImportError:
        print_error("PyTorch is not installed. Install it with:")
        print("pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print_error(f"Error during GPU test: {str(e)}")
        return False

def main():
    """Main function to test GPU acceleration."""
    print_info("=== Metal GPU Acceleration Test ===\n")
    
    if not check_platform():
        print_error("\nFailed: Platform check")
        sys.exit(1)
    
    env_check = check_environment()
    if not env_check:
        print_warning("\nWarning: Some environment variables are not set")
        print_info("Continuing with test anyway...\n")
    
    if detect_gpu():
        print_success("\n✓ Metal GPU acceleration is working properly!")
    else:
        print_error("\n✗ Metal GPU acceleration test failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 