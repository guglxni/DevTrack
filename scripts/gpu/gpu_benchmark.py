#!/usr/bin/env python3
"""
A simple benchmark script to measure the performance of the Metal GPU
acceleration on Apple Silicon compared to CPU-only execution.

The benchmark performs different types of operations (matrix multiplication,
convolution, etc.) and compares the execution times.
"""

import time
import argparse
import os
import platform
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Please install it to run this benchmark.")

# Set color codes for console output
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
BLUE = '\033[34m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_colored(text, color):
    """Print colored text to the console."""
    print(f"{color}{text}{RESET}")

def print_header(text):
    """Print a section header."""
    print(f"\n{BLUE}{BOLD}{text}{RESET}")
    print("-" * 80)

def print_result(operation, cpu_time, gpu_time, speedup):
    """Print a benchmark result with nice formatting."""
    print(f"{operation:30} | CPU: {cpu_time:8.4f}s | GPU: {gpu_time:8.4f}s | {speedup:6.2f}x speedup")

def is_mps_available():
    """Check if MPS (Metal Performance Shaders) is available."""
    if not TORCH_AVAILABLE:
        return False
    
    return torch.backends.mps.is_available()

def get_device():
    """Get the appropriate device for computations."""
    if not TORCH_AVAILABLE:
        return None
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def benchmark_matrix_multiply(sizes, device, iterations=3):
    """Benchmark matrix multiplication for different sizes."""
    results = []
    
    print_header(f"Matrix Multiplication Benchmark ({iterations} iterations each)")
    
    for size in sizes:
        print(f"Size: {size}x{size}")
        
        # CPU benchmark
        cpu_times = []
        for i in range(iterations):
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            
            # Warm-up
            _ = torch.matmul(a_cpu, b_cpu)
            
            start = time.time()
            c_cpu = torch.matmul(a_cpu, b_cpu)
            # Force synchronization
            _ = c_cpu.numpy()
            end = time.time()
            cpu_time = end - start
            cpu_times.append(cpu_time)
            print(f"  CPU Iteration {i+1}: {cpu_time:.4f}s")
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        print(f"  {YELLOW}CPU Average: {avg_cpu_time:.4f}s{RESET}")
        
        # GPU benchmark
        if device and device.type == "mps":
            gpu_times = []
            for i in range(iterations):
                a_gpu = torch.randn(size, size, device=device)
                b_gpu = torch.randn(size, size, device=device)
                
                # Warm-up
                _ = torch.matmul(a_gpu, b_gpu)
                
                # Synchronize before starting the timer
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
                
                start = time.time()
                c_gpu = torch.matmul(a_gpu, b_gpu)
                # Force synchronization
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
                _ = c_gpu.cpu().numpy()
                end = time.time()
                gpu_time = end - start
                gpu_times.append(gpu_time)
                print(f"  GPU Iteration {i+1}: {gpu_time:.4f}s")
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            print(f"  {GREEN}GPU Average: {avg_gpu_time:.4f}s | Speedup: {speedup:.2f}x{RESET}")
        else:
            avg_gpu_time = 0
            speedup = 0
            print(f"  {RED}GPU not available{RESET}")
        
        results.append({
            "operation": f"Matrix Multiplication {size}x{size}",
            "cpu_time": avg_cpu_time,
            "gpu_time": avg_gpu_time if avg_gpu_time > 0 else None,
            "speedup": speedup
        })
    
    return results

def benchmark_convolution(image_sizes, device, iterations=3):
    """Benchmark 2D convolution for different image sizes."""
    results = []
    
    print_header(f"2D Convolution Benchmark ({iterations} iterations each)")
    
    for size in image_sizes:
        print(f"Image size: {size}x{size}")
        
        # CPU benchmark
        cpu_times = []
        for i in range(iterations):
            # Create input with batch_size=1, channels=3, height=size, width=size
            input_cpu = torch.randn(1, 3, size, size)
            # Create a 3x3 kernel with 3 input channels and 16 output channels
            kernel_cpu = torch.randn(16, 3, 3, 3)
            
            # Warm-up
            _ = torch.nn.functional.conv2d(input_cpu, kernel_cpu)
            
            start = time.time()
            output_cpu = torch.nn.functional.conv2d(input_cpu, kernel_cpu)
            # Force synchronization
            _ = output_cpu.numpy()
            end = time.time()
            cpu_time = end - start
            cpu_times.append(cpu_time)
            print(f"  CPU Iteration {i+1}: {cpu_time:.4f}s")
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        print(f"  {YELLOW}CPU Average: {avg_cpu_time:.4f}s{RESET}")
        
        # GPU benchmark
        if device and device.type == "mps":
            gpu_times = []
            for i in range(iterations):
                input_gpu = torch.randn(1, 3, size, size, device=device)
                kernel_gpu = torch.randn(16, 3, 3, 3, device=device)
                
                # Warm-up
                _ = torch.nn.functional.conv2d(input_gpu, kernel_gpu)
                
                # Synchronize before starting the timer
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
                
                start = time.time()
                output_gpu = torch.nn.functional.conv2d(input_gpu, kernel_gpu)
                # Force synchronization
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
                _ = output_gpu.cpu().numpy()
                end = time.time()
                gpu_time = end - start
                gpu_times.append(gpu_time)
                print(f"  GPU Iteration {i+1}: {gpu_time:.4f}s")
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            print(f"  {GREEN}GPU Average: {avg_gpu_time:.4f}s | Speedup: {speedup:.2f}x{RESET}")
        else:
            avg_gpu_time = 0
            speedup = 0
            print(f"  {RED}GPU not available{RESET}")
        
        results.append({
            "operation": f"2D Convolution {size}x{size}",
            "cpu_time": avg_cpu_time,
            "gpu_time": avg_gpu_time if avg_gpu_time > 0 else None,
            "speedup": speedup
        })
    
    return results

def benchmark_neural_network(device, iterations=3):
    """Benchmark a simple neural network forward and backward pass."""
    if not TORCH_AVAILABLE:
        print(f"{RED}PyTorch not available. Skipping neural network benchmark.{RESET}")
        return []
    
    results = []
    
    print_header(f"Neural Network Benchmark ({iterations} iterations each)")
    
    # Define a simple convolutional neural network
    class SimpleConvNet(torch.nn.Module):
        def __init__(self):
            super(SimpleConvNet, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.fc1 = torch.nn.Linear(64 * 4 * 4, 512)
            self.fc2 = torch.nn.Linear(512, 10)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 64 * 4 * 4)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create a batch of 32 random 3x32x32 images (RGB, 32x32 pixels)
    batch_size = 32
    
    print("Model: Simple ConvNet with 3 conv layers and 2 fully connected layers")
    print(f"Input: Batch of {batch_size} RGB images (3x32x32)")
    
    # CPU benchmark
    cpu_times_forward = []
    cpu_times_backward = []
    
    # Create model and tensors on CPU
    model_cpu = SimpleConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    
    for i in range(iterations):
        inputs_cpu = torch.randn(batch_size, 3, 32, 32)
        targets_cpu = torch.randint(0, 10, (batch_size,))
        
        # Warm-up
        outputs_cpu = model_cpu(inputs_cpu)
        loss_cpu = criterion(outputs_cpu, targets_cpu)
        loss_cpu.backward()
        
        # Forward pass
        model_cpu.zero_grad()
        start = time.time()
        outputs_cpu = model_cpu(inputs_cpu)
        end = time.time()
        cpu_forward_time = end - start
        cpu_times_forward.append(cpu_forward_time)
        
        # Backward pass
        loss_cpu = criterion(outputs_cpu, targets_cpu)
        start = time.time()
        loss_cpu.backward()
        end = time.time()
        cpu_backward_time = end - start
        cpu_times_backward.append(cpu_backward_time)
        
        print(f"  CPU Iteration {i+1}: Forward: {cpu_forward_time:.4f}s | Backward: {cpu_backward_time:.4f}s")
    
    avg_cpu_forward = sum(cpu_times_forward) / len(cpu_times_forward)
    avg_cpu_backward = sum(cpu_times_backward) / len(cpu_times_backward)
    print(f"  {YELLOW}CPU Average: Forward: {avg_cpu_forward:.4f}s | Backward: {avg_cpu_backward:.4f}s{RESET}")
    
    # GPU benchmark
    if device and device.type == "mps":
        gpu_times_forward = []
        gpu_times_backward = []
        
        # Create model and tensors on GPU
        model_gpu = SimpleConvNet().to(device)
        
        for i in range(iterations):
            inputs_gpu = torch.randn(batch_size, 3, 32, 32, device=device)
            targets_gpu = torch.randint(0, 10, (batch_size,), device=device)
            
            # Warm-up
            outputs_gpu = model_gpu(inputs_gpu)
            loss_gpu = criterion(outputs_gpu, targets_gpu)
            loss_gpu.backward()
            
            # Forward pass
            model_gpu.zero_grad()
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            start = time.time()
            outputs_gpu = model_gpu(inputs_gpu)
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            end = time.time()
            gpu_forward_time = end - start
            gpu_times_forward.append(gpu_forward_time)
            
            # Backward pass
            loss_gpu = criterion(outputs_gpu, targets_gpu)
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            start = time.time()
            loss_gpu.backward()
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            end = time.time()
            gpu_backward_time = end - start
            gpu_times_backward.append(gpu_backward_time)
            
            print(f"  GPU Iteration {i+1}: Forward: {gpu_forward_time:.4f}s | Backward: {gpu_backward_time:.4f}s")
        
        avg_gpu_forward = sum(gpu_times_forward) / len(gpu_times_forward)
        avg_gpu_backward = sum(gpu_times_backward) / len(gpu_times_backward)
        
        speedup_forward = avg_cpu_forward / avg_gpu_forward if avg_gpu_forward > 0 else 0
        speedup_backward = avg_cpu_backward / avg_gpu_backward if avg_gpu_backward > 0 else 0
        
        print(f"  {GREEN}GPU Average: Forward: {avg_gpu_forward:.4f}s | Backward: {avg_gpu_backward:.4f}s{RESET}")
        print(f"  {GREEN}Speedup: Forward: {speedup_forward:.2f}x | Backward: {speedup_backward:.2f}x{RESET}")
        
        results.append({
            "operation": "Neural Network Forward Pass",
            "cpu_time": avg_cpu_forward,
            "gpu_time": avg_gpu_forward,
            "speedup": speedup_forward
        })
        
        results.append({
            "operation": "Neural Network Backward Pass",
            "cpu_time": avg_cpu_backward,
            "gpu_time": avg_gpu_backward,
            "speedup": speedup_backward
        })
    else:
        print(f"  {RED}GPU not available{RESET}")
        
        results.append({
            "operation": "Neural Network Forward Pass",
            "cpu_time": avg_cpu_forward,
            "gpu_time": None,
            "speedup": 0
        })
        
        results.append({
            "operation": "Neural Network Backward Pass",
            "cpu_time": avg_cpu_backward,
            "gpu_time": None,
            "speedup": 0
        })
    
    return results

def plot_results(results, save_path=None):
    """Plot benchmark results as a bar chart."""
    operations = [result["operation"] for result in results]
    cpu_times = [result["cpu_time"] for result in results]
    gpu_times = [result["gpu_time"] if result["gpu_time"] is not None else 0 for result in results]
    speedups = [result["speedup"] for result in results]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # First subplot: execution times
    bar_width = 0.35
    index = np.arange(len(operations))
    
    ax1.bar(index, cpu_times, bar_width, label='CPU', color='cornflowerblue')
    ax1.bar(index + bar_width, gpu_times, bar_width, label='GPU (Metal)', color='forestgreen')
    
    ax1.set_xlabel('Operation')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('CPU vs GPU Execution Time Comparison')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(operations, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Second subplot: speedups
    ax2.bar(index, speedups, color='orangered')
    
    ax2.set_xlabel('Operation')
    ax2.set_ylabel('Speedup (X times faster)')
    ax2.set_title('GPU Speedup Compared to CPU')
    ax2.set_xticks(index)
    ax2.set_xticklabels(operations, rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add speedup values on top of bars
    for i, v in enumerate(speedups):
        if v > 0:  # Only label bars with positive speedup
            ax2.text(i, v + 0.1, f'{v:.2f}x', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def save_results(results, output_path):
    """Save benchmark results to a JSON file."""
    result_data = {
        "benchmark_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "Not available",
            "mps_available": is_mps_available(),
        },
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Saved results to {output_path}")

def print_summary(results):
    """Print a summary of the benchmark results."""
    print_header("Benchmark Summary")
    
    if not results:
        print("No benchmark results available.")
        return
    
    # Calculate overall averages
    cpu_times = [r["cpu_time"] for r in results]
    gpu_times = [r["gpu_time"] for r in results if r["gpu_time"] is not None]
    speedups = [r["speedup"] for r in results if r["speedup"] > 0]
    
    avg_cpu_time = sum(cpu_times) / len(cpu_times) if cpu_times else 0
    avg_gpu_time = sum(gpu_times) / len(gpu_times) if gpu_times else 0
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0
    max_speedup = max(speedups) if speedups else 0
    
    # Find operation with max speedup
    max_speedup_op = ""
    for r in results:
        if r["speedup"] == max_speedup:
            max_speedup_op = r["operation"]
            break
    
    # Print individual results
    for r in results:
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] > 0 else "N/A"
        gpu_time_str = f"{r['gpu_time']:.4f}s" if r["gpu_time"] is not None else "N/A"
        print_result(r["operation"], r["cpu_time"], r["gpu_time"] or 0, r["speedup"])
    
    print("\n" + "=" * 80)
    print(f"{BOLD}Overall Statistics:{RESET}")
    print(f"Average CPU time: {avg_cpu_time:.4f}s")
    print(f"Average GPU time: {avg_gpu_time:.4f}s")
    print(f"Average speedup: {BLUE}{avg_speedup:.2f}x{RESET}")
    print(f"Maximum speedup: {GREEN}{max_speedup:.2f}x{RESET} ({max_speedup_op})")
    
    if not gpu_times:
        print(f"\n{RED}Warning: No GPU benchmarks were run. Metal GPU acceleration may not be available.{RESET}")

def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description="Metal GPU Acceleration Benchmark")
    parser.add_argument("-m", "--matrix-sizes", nargs="+", type=int, default=[1000, 2000, 3000],
                        help="Sizes of matrices for matrix multiplication benchmark")
    parser.add_argument("-c", "--conv-sizes", nargs="+", type=int, default=[128, 256, 512],
                        help="Image sizes for convolution benchmark")
    parser.add_argument("-i", "--iterations", type=int, default=3,
                        help="Number of iterations for each benchmark")
    parser.add_argument("-o", "--output", type=str, help="Path to save benchmark results as JSON")
    parser.add_argument("-p", "--plot", type=str, help="Path to save benchmark plot")
    parser.add_argument("--no-nn", action="store_true", help="Skip neural network benchmark")
    parser.add_argument("--no-conv", action="store_true", help="Skip convolution benchmark")
    parser.add_argument("--no-matrix", action="store_true", help="Skip matrix multiplication benchmark")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    
    # Print system information
    print_header("System Information")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    
    if not TORCH_AVAILABLE:
        print(f"{RED}PyTorch is not installed. Please install it to run the benchmark.{RESET}")
        return
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for Metal support
    if is_mps_available():
        print(f"{GREEN}Metal Performance Shaders (MPS) is available.{RESET}")
        device = get_device()
        print(f"Using device: {device}")
    else:
        print(f"{RED}Metal Performance Shaders (MPS) is not available. Benchmarks will run on CPU only.{RESET}")
        print(f"Possible reasons:")
        print(f"1. Your Mac doesn't have Apple Silicon (M1/M2/M3/M4) chip")
        print(f"2. Your PyTorch version doesn't support MPS")
        print(f"3. You need to set environment variables (PYTORCH_ENABLE_MPS_FALLBACK=1)")
        device = get_device()
    
    # Run benchmarks
    all_results = []
    
    if not args.no_matrix:
        matrix_results = benchmark_matrix_multiply(args.matrix_sizes, device, args.iterations)
        all_results.extend(matrix_results)
    
    if not args.no_conv:
        conv_results = benchmark_convolution(args.conv_sizes, device, args.iterations)
        all_results.extend(conv_results)
    
    if not args.no_nn:
        nn_results = benchmark_neural_network(device, args.iterations)
        all_results.extend(nn_results)
    
    # Print summary
    print_summary(all_results)
    
    # Save results if requested
    if args.output:
        save_results(all_results, args.output)
    
    # Plot results if requested
    if args.plot and all_results:
        plot_results(all_results, args.plot)

if __name__ == "__main__":
    main() 