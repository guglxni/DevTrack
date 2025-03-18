#!/usr/bin/env python3
"""
GPU Acceleration API Routes.

This module defines FastAPI routes for GPU acceleration features,
including system information, benchmark tests, and server management.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

# Import GPU controller
from src.api.gpu_controller import controller, ENV_METAL_ENABLED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_routes")

# Define Pydantic models for request/response handling
class SystemInfo(BaseModel):
    """Response model for system information."""
    os_version: str
    python_version: str
    memory_info: str
    is_apple_silicon: bool
    chip_model: str = ""
    metal_enabled: bool
    gpu_mode: str
    gpu_layers: str
    metal_family: Optional[str] = None
    gpu_memory: Optional[str] = None

class DetailedSystemInfo(SystemInfo):
    """More detailed system information response."""
    python_packages: Dict[str, str]
    environment_variables: Dict[str, str]
    gpu_settings: Dict[str, str]
    server_pid: Optional[int] = None

class GPUTestResult(BaseModel):
    """Response model for GPU test results."""
    status: str
    message: str
    cpu_time: Optional[float] = None
    gpu_time: Optional[float] = None
    speedup: Optional[float] = None
    matrix_size: Optional[int] = None

class ServerManagementResponse(BaseModel):
    """Response model for server management operations."""
    status: str
    message: str
    previous_pid: Optional[int] = None

class BenchmarkResult(BaseModel):
    """Response model for benchmark results."""
    benchmark_id: str
    timestamp: str
    duration: float
    operations_per_second: int
    speedup_vs_cpu: Optional[float] = None
    model_name: Optional[str] = None
    status: str
    message: str

class GPUSettings(BaseModel):
    """Request model for updating GPU settings."""
    mode: str = Field(..., description="GPU acceleration mode: optimal, basic, advanced, custom, or disabled")
    restart_server: bool = Field(False, description="Whether to restart the server with new settings")

# Create API router
router = APIRouter(prefix="/gpu-acceleration", tags=["gpu-acceleration"])

# Define routes
@router.get("/system-info", response_model=SystemInfo)
async def get_system_info():
    """Get basic system information including GPU capabilities."""
    try:
        info = controller.get_system_info()
        return info
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system info: {str(e)}")

@router.get("/detailed-system-info", response_model=DetailedSystemInfo)
async def get_detailed_system_info():
    """Get detailed system information including environment and packages."""
    try:
        # Get basic system info
        info = controller.get_system_info()
        
        # Add Python packages
        import pkg_resources
        packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        info["python_packages"] = packages
        
        # Add filtered environment variables (excluding sensitive info)
        env_vars = {}
        for key, value in os.environ.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in ["key", "secret", "password", "token"]):
                continue
            env_vars[key] = value
        info["environment_variables"] = env_vars
        
        # Add GPU settings
        info["gpu_settings"] = controller.current_settings
        
        # Add server PID
        info["server_pid"] = controller.get_server_pid()
        
        return info
    except Exception as e:
        logger.error(f"Error getting detailed system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting detailed system info: {str(e)}")

@router.get("/server-status")
async def get_server_status():
    """Get current server status."""
    try:
        pid = controller.get_server_pid()
        
        if pid is None:
            return {
                "status": "not_running",
                "message": "Server is not running",
                "pid": None,
                "running": False
            }
        else:
            # Try to determine the mode by checking the environment variables
            mode = "unknown"
            if os.environ.get(ENV_METAL_ENABLED) == "1":
                mode = "gpu_accelerated"
            else:
                mode = "cpu_only"
                
            return {
                "status": "running",
                "message": f"Server is running with PID {pid}",
                "pid": pid,
                "running": True,
                "mode": mode
            }
    except Exception as e:
        logger.error(f"Error checking server status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking server status: {str(e)}")

@router.get("/test-gpu", response_model=GPUTestResult)
async def test_gpu():
    """Run a test to verify GPU acceleration is working."""
    try:
        result = controller.run_gpu_test()
        return result
    except Exception as e:
        logger.error(f"Error during GPU test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during GPU test: {str(e)}")

@router.post("/restart-server", response_model=ServerManagementResponse)
async def restart_server(mode: str = "advanced_gpu"):
    """Restart the server with specified GPU acceleration mode."""
    try:
        if mode not in ["cpu", "basic_gpu", "advanced_gpu"]:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
        
        result = controller.restart_server(mode)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error restarting server: {str(e)}")

@router.post("/stop-server", response_model=ServerManagementResponse)
async def stop_server():
    """Stop the running server."""
    try:
        result = controller.stop_server()
        return result
    except Exception as e:
        logger.error(f"Error stopping server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping server: {str(e)}")

@router.post("/settings", response_model=Dict[str, Any])
async def update_gpu_settings(settings: GPUSettings, background_tasks: BackgroundTasks):
    """Update GPU acceleration settings."""
    try:
        # Apply appropriate settings based on mode
        if settings.mode == "disabled":
            controller.disable_gpu_acceleration()
            message = "GPU acceleration disabled"
        else:
            controller.enable_gpu_acceleration(settings.mode)
            message = f"GPU acceleration enabled with {settings.mode} mode"
        
        # Get the updated settings
        updated_settings = controller.current_settings
        
        # Restart server in background if requested
        if settings.restart_server:
            # Map mode to server restart mode
            mode_map = {
                "disabled": "cpu",
                "basic": "basic_gpu",
                "optimal": "advanced_gpu",
                "advanced": "advanced_gpu",
                "custom": "advanced_gpu"
            }
            server_mode = mode_map.get(settings.mode, "advanced_gpu")
            
            # Schedule restart in background
            background_tasks.add_task(controller.restart_server, server_mode)
            message += ". Server restart scheduled."
        
        return {
            "status": "success",
            "message": message,
            "settings": updated_settings,
            "restart_scheduled": settings.restart_server
        }
    except Exception as e:
        logger.error(f"Error updating GPU settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating GPU settings: {str(e)}")

@router.get("/benchmarks", response_model=List[BenchmarkResult])
async def get_benchmarks():
    """Get historical benchmark results."""
    try:
        # Define benchmark results file path
        benchmark_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "benchmarks.json")
        
        # Create empty list if file doesn't exist
        if not os.path.exists(benchmark_file):
            return []
        
        # Read and return benchmark results
        with open(benchmark_file, "r") as f:
            benchmarks = json.load(f)
            
        return benchmarks
    except Exception as e:
        logger.error(f"Error getting benchmark results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting benchmark results: {str(e)}")

@router.post("/run-benchmark", response_model=BenchmarkResult)
async def run_benchmark(matrix_size: int = 2000, iterations: int = 5):
    """Run a benchmark to measure GPU acceleration performance."""
    try:
        # Safety check for parameters
        if matrix_size < 100 or matrix_size > 5000:
            raise HTTPException(status_code=400, detail="Matrix size must be between 100 and 5000")
        
        if iterations < 1 or iterations > 20:
            raise HTTPException(status_code=400, detail="Iterations must be between 1 and 20")
        
        # Import necessary libraries
        import torch
        import numpy as np
        import uuid
        from datetime import datetime
        
        # Check if Metal is available
        if not torch.backends.mps.is_available():
            return {
                "benchmark_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "duration": 0,
                "operations_per_second": 0,
                "speedup_vs_cpu": None,
                "model_name": "N/A",
                "status": "failed",
                "message": "Metal is not available for PyTorch"
            }
        
        # Initialize result
        benchmark_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create tensors on CPU and MPS
        mps_device = torch.device("mps")
        
        # Get common info
        info = controller.get_system_info()
        model_name = info.get("chip_model", "Unknown")
        
        # Initialize measurement variables
        cpu_times = []
        gpu_times = []
        
        # Run benchmark iterations
        for i in range(iterations):
            # Create fresh tensor for each iteration
            cpu_tensor = torch.randn(matrix_size, matrix_size)
            mps_tensor = cpu_tensor.to(mps_device)
            
            # CPU benchmark
            cpu_start = time.time()
            cpu_result = torch.matmul(cpu_tensor, cpu_tensor.t())
            cpu_times.append(time.time() - cpu_start)
            
            # GPU benchmark
            mps_start = time.time()
            mps_result = torch.matmul(mps_tensor, mps_tensor.t())
            mps_result.cpu()  # Force synchronization
            gpu_times.append(time.time() - mps_start)
        
        # Calculate average times (excluding first run for warmup)
        avg_cpu_time = sum(cpu_times[1:]) / (iterations - 1) if iterations > 1 else cpu_times[0]
        avg_gpu_time = sum(gpu_times[1:]) / (iterations - 1) if iterations > 1 else gpu_times[0]
        
        # Calculate speedup
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
        
        # Calculate operations per second (approximate)
        # Matrix multiplication is roughly 2*n^3 operations
        operations = 2 * (matrix_size ** 3)
        operations_per_second = int(operations / avg_gpu_time)
        
        # Create benchmark result
        result = {
            "benchmark_id": benchmark_id,
            "timestamp": timestamp,
            "duration": avg_gpu_time,
            "operations_per_second": operations_per_second,
            "speedup_vs_cpu": speedup,
            "model_name": model_name,
            "status": "success",
            "message": f"Benchmark completed with {iterations} iterations. Matrix size: {matrix_size}x{matrix_size}"
        }
        
        # Save benchmark result to file
        benchmark_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "benchmarks.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(benchmark_file), exist_ok=True)
        
        # Read existing benchmarks or create empty list
        benchmarks = []
        if os.path.exists(benchmark_file):
            try:
                with open(benchmark_file, "r") as f:
                    benchmarks = json.load(f)
            except:
                benchmarks = []
        
        # Add new benchmark and save
        benchmarks.append(result)
        
        # Keep only the last 20 benchmarks
        benchmarks = benchmarks[-20:]
        
        with open(benchmark_file, "w") as f:
            json.dump(benchmarks, f, indent=2)
        
        return result
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running benchmark: {str(e)}")

@router.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Get the GPU Acceleration Dashboard HTML page
    """
    dashboard_path = os.path.join("src", "web", "static", "gpu-acceleration", "index.html")
    logger.info(f"Trying to serve dashboard from path: {dashboard_path}")
    try:
        if not os.path.exists(dashboard_path):
            logger.error(f"Dashboard file not found at path: {dashboard_path}")
            raise HTTPException(status_code=404, detail=f"Dashboard file not found at {dashboard_path}")
            
        with open(dashboard_path, "r") as f:
            content = f.read()
            logger.info(f"Successfully read dashboard file ({len(content)} bytes)")
            return content
    except Exception as e:
        logger.error(f"Error serving GPU dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving GPU dashboard: {str(e)}")

@router.get("/index.html", response_class=HTMLResponse)
async def get_dashboard_index():
    """
    Alternative route for the dashboard that handles /index.html path
    """
    return await get_dashboard()

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard_alt():
    """
    Alternative route for the dashboard that handles /dashboard path
    """
    return await get_dashboard()

@router.get("/benchmark-results", response_model=List[BenchmarkResult])
async def get_benchmark_results():
    """Alternative endpoint for getting benchmark results to match dashboard expectations."""
    return await get_benchmarks()

@router.get("/monitoring-data")
async def get_monitoring_data():
    """
    Get GPU monitoring data for the dashboard.
    
    This endpoint returns memory usage and CPU usage data for visualization.
    """
    try:
        # Create some basic monitoring data structure
        # In a production system, this would come from actual monitoring
        monitoring_data = {
            "timestamps": [],
            "memory_usage": [],
            "cpu_usage": []
        }
        
        # Try to read from a data file if it exists
        monitor_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gpu_monitoring.json")
        
        if os.path.exists(monitor_file):
            try:
                with open(monitor_file, "r") as f:
                    stored_data = json.load(f)
                    monitoring_data = stored_data
                    logger.info(f"Loaded monitoring data with {len(monitoring_data['timestamps'])} data points")
            except Exception as e:
                logger.error(f"Error reading monitoring data: {str(e)}")
        else:
            # Generate sample data for testing if no data file exists
            import time
            from datetime import datetime, timedelta
            import random
            
            now = datetime.now()
            for i in range(10):
                timestamp = (now - timedelta(minutes=9-i)).isoformat()
                monitoring_data["timestamps"].append(timestamp)
                monitoring_data["memory_usage"].append(random.uniform(0.1, 0.5))  # GB of memory
                monitoring_data["cpu_usage"].append(random.uniform(10, 40))  # CPU percentage
            
            logger.info(f"Generated sample monitoring data with {len(monitoring_data['timestamps'])} data points")
            
            # Save this data for future use
            os.makedirs(os.path.dirname(monitor_file), exist_ok=True)
            with open(monitor_file, "w") as f:
                json.dump(monitoring_data, f)
        
        return monitoring_data
    except Exception as e:
        logger.error(f"Error getting monitoring data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting monitoring data: {str(e)}")

@router.post("/run-benchmarks", response_model=BenchmarkResult)
async def run_benchmarks():
    """Alternative endpoint for running benchmarks to match dashboard expectations."""
    return await run_benchmark(matrix_size=2000, iterations=5)

def add_routes_to_app(app):
    """Add GPU acceleration routes to the main FastAPI app."""
    app.include_router(router)
    logger.info("GPU acceleration routes have been registered")

if __name__ == "__main__":
    # For testing with uvicorn directly
    import uvicorn
    
    # Allow running this file directly with proper Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Run API
    uvicorn.run("gpu_acceleration_routes:router", host="0.0.0.0", port=8000) 