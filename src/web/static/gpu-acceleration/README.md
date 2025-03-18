# GPU Acceleration Dashboard

A web-based dashboard for monitoring and controlling Metal GPU acceleration on Apple Silicon Macs.

## Overview

The GPU Acceleration Dashboard provides a user-friendly interface to:

- Monitor GPU and CPU performance
- Control server acceleration modes
- Run benchmarks to measure GPU performance
- Configure GPU acceleration settings
- View detailed system information

## Accessing the Dashboard

The dashboard is accessible at:
```
http://localhost:8003/gpu-acceleration/
```

Alternative URLs:
- `http://localhost:8003/gpu-acceleration/index.html`
- `http://localhost:8003/gpu-acceleration/dashboard`

## Features

### System Information Card

Displays hardware and GPU acceleration status:
- Chip model and OS version
- Memory information
- GPU acceleration status
- Current GPU mode and layers

Click "Refresh" to update this information, or "Show Detailed System Information" for advanced details.

### Server Control Card

Manage the API server with different GPU acceleration options:
- **Start Optimized**: Start with optimized GPU settings for your chip
- **Start Basic GPU**: Start with basic GPU acceleration
- **Start CPU-only**: Start without GPU acceleration
- **Stop Server**: Stop the currently running server

### Benchmarking Card

Run benchmarks to measure GPU acceleration performance:
- **Run Matrix Benchmark**: Test matrix multiplication operations
- **Run LLM Benchmark**: Test language model operations
- **View Results**: Display detailed benchmark results

### Performance Visualization

Interactive charts for monitoring performance:
- **GPU Memory**: Shows GPU memory usage over time
- **Benchmark Results**: Compares CPU vs. GPU performance
- **CPU Usage**: Shows CPU utilization

### Status Log

Real-time feedback on operations:
- Information messages
- Warning messages
- Error messages
- Click "Clear" to reset the log

## Troubleshooting

### Common Issues

1. **"Error fetching monitoring data"**:
   - Verify the server is running with GPU acceleration enabled
   - Check that the data directory exists (`mkdir -p data`)
   - Restart the server

2. **"Error fetching benchmark results"**:
   - Run a benchmark first using the "Run Matrix Benchmark" button
   - Check that the data directory exists

3. **Status shows "Server is not running" but the server is active**:
   - Click "Refresh" to update the status
   - Restart the dashboard page
   - Restart the server using `./restart_integrated_server.sh`

4. **Charts not updating**:
   - Wait for the 10-second automatic refresh
   - Click "Analyze Performance" to manually refresh

5. **GPU acceleration not showing improved performance**:
   - Ensure your Apple Silicon Mac meets the requirements
   - Try increasing the matrix size for benchmarks
   - Verify no other GPU-intensive applications are running

## API Endpoints

The dashboard interacts with these API endpoints:

- `GET /gpu-acceleration/system-info`: Get basic system information
- `GET /gpu-acceleration/detailed-system-info`: Get detailed system info
- `GET /gpu-acceleration/server-status`: Check server status
- `POST /gpu-acceleration/restart-server`: Restart server
- `POST /gpu-acceleration/stop-server`: Stop server
- `POST /gpu-acceleration/settings`: Update GPU settings
- `GET /gpu-acceleration/test-gpu`: Test GPU acceleration
- `GET /gpu-acceleration/benchmarks`: Get benchmark results
- `POST /gpu-acceleration/run-benchmark`: Run benchmark
- `GET /gpu-acceleration/monitoring-data`: Get monitoring data

For more details, see the main API Documentation.
