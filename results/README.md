# Metal GPU Acceleration Benchmark Results

This directory contains the results of Metal GPU acceleration benchmarks and tests for Apple Silicon Macs.

## Files in this Directory

- `system_info_*.txt`: Contains detailed information about the system, including:
  - OS version and platform details
  - Processor and chip information
  - Memory and GPU information
  - Python environment details
  - PyTorch configuration and Metal settings

- `gpu_benchmark_*.json`: Raw benchmark data in JSON format, including:
  - Matrix multiplication benchmarks
  - Convolution operation benchmarks 
  - Neural network forward/backward pass benchmarks
  - Speedup measurements for each operation

- `gpu_benchmark_*.png`: Visualization of benchmark results, showing:
  - Comparison of CPU vs GPU execution times
  - Speedup factors for each operation

- `performance_analysis_*.png`: Analysis of performance monitoring data, including:
  - GPU memory usage over time
  - CPU utilization patterns
  - Performance bottlenecks

## Understanding the Results

### Speedup Interpretation

Speedup values represent how many times faster the Metal GPU is compared to CPU-only execution:
- Values > 1: GPU is faster (e.g., 2.5x means GPU is 2.5 times faster)
- Values ≈ 1: GPU and CPU performance is similar
- Values < 1: CPU is faster than GPU (rare, but can happen for small workloads due to overhead)

### Optimal Operations for GPU

Metal GPUs typically excel at:
- Large matrix multiplications (2000x2000 and larger)
- Convolutional operations on larger images
- Neural network forward and backward passes

Small operations may not see significant speedup due to the overhead of transferring data to/from the GPU.

### Memory Usage Patterns

The performance analysis plots show GPU memory usage over time, which can help identify:
- Memory leaks (steadily increasing usage)
- Inefficient memory patterns (frequent spikes)
- Underutilization of GPU resources

## Running New Benchmarks

To run a new set of benchmarks, execute:

```
./run_benchmarks.sh
```

This will create new result files with the current timestamp in this directory. 