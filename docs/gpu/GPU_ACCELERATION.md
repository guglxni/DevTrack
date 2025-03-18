# Metal GPU Acceleration for LLM Inference

This guide explains how to use Metal GPU acceleration for faster LLM inference on Apple Silicon Macs.

## Overview

The DevTrack Assessment API can now leverage Apple's Metal GPU acceleration to significantly speed up LLM inference. Our benchmarks show:

- **Average Speedup**: 2.3x faster inference with Metal GPU acceleration
- **Best Case Speedup**: 7.1x faster in some cases
- **Improved Response Times**:
  - **CPU-only Mode**: 13.62s average response time
  - **Metal GPU Mode**: 5.86s average response time
- **Consistency**: More consistent response times with less variation

## Requirements

- Apple Silicon Mac (M1, M2, M3, or M4 chip)
- macOS 12 (Monterey) or later
- Python 3.8 or later
- Required Python packages: `pandas`, `matplotlib`, `torch` (for GPU acceleration and performance analysis)

## Quick Start

The easiest way to start the server with optimized Metal GPU acceleration is:

```bash
./start_optimized.sh
```

This script will:
1. Detect your Apple Silicon chip model
2. Apply optimized settings for your specific hardware
3. Start the server with optimal GPU acceleration
4. Enable performance monitoring

## GPU Acceleration Dashboard

A dedicated dashboard is now available to monitor and control GPU acceleration:

### Accessing the Dashboard

Open a web browser and navigate to:
```
http://localhost:8003/gpu-acceleration/
```

### Dashboard Features

The GPU Acceleration Dashboard provides:

1. **System Information**:
   - Hardware details (chip model, memory)
   - GPU acceleration status
   - Current settings

2. **Server Control**:
   - Start optimized GPU acceleration
   - Start basic GPU acceleration
   - Start in CPU-only mode
   - Stop server

3. **Performance Monitoring**:
   - Real-time GPU memory usage
   - CPU utilization
   - Performance visualization

4. **Benchmarking Tools**:
   - Run matrix multiplication benchmarks
   - Test LLM inference performance
   - View and compare benchmark results

5. **Status Log**:
   - Real-time operation feedback
   - Error reporting and troubleshooting assistance

### Using the Dashboard

1. **View System Information**:
   The top card displays your hardware details and GPU acceleration status.
   - Click "Refresh" to update system information.
   - Click "Show Detailed System Information" for advanced details.

2. **Control the Server**:
   Use the server control card to:
   - Start the server with optimized GPU settings
   - Start with basic GPU settings
   - Start in CPU-only mode
   - Stop the currently running server

3. **Run Benchmarks**:
   - Click "Run Matrix Benchmark" to test matrix operations
   - Click "Run LLM Benchmark" to test language model inference
   - View results in the charts and tables below

4. **Monitor Performance**:
   Select tabs to view:
   - GPU Memory usage over time
   - Benchmark results comparison
   - CPU usage statistics

## Available Scripts

### 1. Optimized Startup (Recommended)

```bash
./start_optimized.sh
```

- Automatic chip detection
- Model-specific optimizations
- Performance monitoring
- Comprehensive feedback

### 2. Basic GPU Acceleration

```bash
./restart_with_metal.sh
```

- Simple Metal configuration
- No chip-specific optimizations
- Good for testing

### 3. Advanced GPU Acceleration

```bash
./restart_with_metal_advanced.sh
```

- Manual configuration
- Fixed settings for all chip types
- No monitoring

### 4. Standard CPU-only Mode

```bash
./restart_integrated_server.sh
```

- CPU-only operation
- Slower but compatible with any hardware

### 5. Interactive Mode

```bash
./start_server.sh
```

- Interactive menu to choose startup mode
- Provides multiple options

## Performance Monitoring

The optimized startup script automatically collects performance data in the `monitoring/` directory. To analyze this data:

```bash
python3 analyze_performance.py
```

This will:
- Load the most recent performance log
- Analyze CPU and GPU usage
- Provide insights on performance
- Generate a visualization of resource usage over time

### Additional Options

```bash
# Specify a specific log file
python3 analyze_performance.py --log monitoring/gpu_memory_20250101_120000.log

# Save the plot to a file
python3 analyze_performance.py --output performance.png

# Only show analysis without plotting
python3 analyze_performance.py --analyze-only
```

## API Endpoints

The following API endpoints are available for GPU acceleration:

### System Information

- **GET `/gpu-acceleration/system-info`**
  - Returns basic system information (OS, Python version, chip model, GPU status)

- **GET `/gpu-acceleration/detailed-system-info`**
  - Returns detailed system information including packages, environment variables

### Server Control

- **GET `/gpu-acceleration/server-status`**
  - Returns current server status (running state, PID, acceleration mode)

- **POST `/gpu-acceleration/restart-server`**
  - Restarts the server with specified GPU acceleration mode
  - Query parameter: `mode` ("cpu", "basic_gpu", or "advanced_gpu")

- **POST `/gpu-acceleration/stop-server`**
  - Stops the running server

### GPU Settings

- **POST `/gpu-acceleration/settings`**
  - Updates GPU acceleration settings
  - Request body: `{"mode": "optimal|basic|advanced|custom|disabled", "restart_server": true|false}`

### Testing and Benchmarking

- **GET `/gpu-acceleration/test-gpu`**
  - Runs a simple test to verify GPU acceleration is working

- **GET `/gpu-acceleration/benchmarks`**
  - Returns historical benchmark results

- **POST `/gpu-acceleration/run-benchmark`**
  - Runs a benchmark to measure GPU acceleration performance
  - Query parameters: `matrix_size` (default 2000), `iterations` (default 5)

- **POST `/gpu-acceleration/run-benchmarks`**
  - Alternative endpoint for running benchmarks with default parameters

### Monitoring

- **GET `/gpu-acceleration/monitoring-data`**
  - Returns performance monitoring data for visualization (memory usage, CPU usage)

## Troubleshooting

### Dashboard Issues

If you encounter issues with the dashboard:

1. **"Error fetching monitoring data"**:
   - Verify the server is running with GPU acceleration enabled
   - Check that the data directory exists: `mkdir -p data`
   - Restart the server with `./restart_integrated_server.sh`

2. **"Error fetching benchmark results"**:
   - Run a benchmark first using the "Run Matrix Benchmark" button
   - Check that the data directory exists: `mkdir -p data`

3. **"Server is not running" (but it is)**:
   - Refresh the page
   - Check server processes with: `ps -ef | grep uvicorn`
   - Restart the server with `./restart_integrated_server.sh`

### Memory Issues

If you encounter memory errors:

1. Reduce the number of GPU layers:
   ```bash
   export N_GPU_LAYERS=16  # Instead of 32
   ```

2. Disable full offload:
   ```bash
   export GGML_METAL_FULL_OFFLOAD=0
   ```

3. For older M1 chips with limited memory, try:
   ```bash
   export N_GPU_LAYERS=8
   ```

### Performance Issues

If inference is slow:

1. Verify Metal is being used:
   ```bash
   grep "Metal" api_server.log
   ```

2. Check for layers offloaded to GPU:
   ```bash
   grep "offloaded.*layers to GPU" api_server.log
   ```

3. Ensure no other GPU-intensive apps are running

4. Use the dashboard to enable optimal settings for your chip:
   - Navigate to `http://localhost:8003/gpu-acceleration/`
   - Click "Start Optimized"

### Compatibility Issues

If the server won't start:

1. Try the basic Metal configuration:
   ```bash
   ./restart_with_metal.sh
   ```

2. Fall back to CPU-only mode:
   ```bash
   ./restart_integrated_server.sh
   ```

## Advanced Configuration

For advanced users who want to manually tune settings, here are the key environment variables:

```bash
# Metal enablement
export USE_METAL=true                   # Enable Metal backend

# Layer configuration
export N_GPU_LAYERS=32                  # Number of layers to offload to GPU
export LLAMA_N_GPU_LAYERS=32            # For llama.cpp compatibility
export GGML_N_GPU_LAYERS=32             # For GGML backend
export GGML_METAL_FULL_OFFLOAD=1        # Full model offload to GPU

# Memory optimization
export F16_KV=true                      # Use half-precision for key/value cache

# Performance tuning
export PYTORCH_ENABLE_MPS_FALLBACK=1    # Enable MPS fallback
export TOKENIZERS_PARALLELISM=true      # Enable tokenizer parallelism
export GGML_METAL_PATH_RESOURCES="..."  # Custom resource path

# Thread optimization
export OMP_NUM_THREADS=5                # OpenMP thread count
export MKL_NUM_THREADS=5                # MKL thread count
export NUMEXPR_NUM_THREADS=5            # NumExpr thread count
```

## Chip-Specific Recommendations

### M1/M2 Chips

- Use conservative GPU offloading (16-24 layers)
- Set `GGML_METAL_FULL_OFFLOAD=0`
- Monitor memory usage carefully

### M3/M4 Chips

- Use aggressive GPU offloading (all 32 layers)
- Enable `GGML_METAL_FULL_OFFLOAD=1`
- Set higher batch sizes if needed

## Benchmarking

To run a benchmark test of LLM performance:

```bash
python3 benchmark_llm.py -n 3
```

This will test 5 different responses with 3 iterations each and report average timings.

## Recent Updates

- Added GPU Acceleration Dashboard at `/gpu-acceleration/`
- Fixed monitoring data collection and visualization
- Improved server status detection
- Added benchmark comparison visualization
- Enhanced support for M4 chip

## Credits

This GPU acceleration implementation uses the Metal backend built into the GGML library, which powers the llama.cpp inference engine.

## Support

For issues or questions about GPU acceleration, please contact the DevTrack Assessment API team. 