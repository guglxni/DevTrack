# DevTrack Dashboard Interfaces

DevTrack provides a comprehensive set of web-based interfaces for interacting with the system. This document provides an overview of each interface and its capabilities.

## Main Dashboard

**URL:** `/`

The main dashboard serves as the central hub for accessing all DevTrack interfaces. It provides:

- Quick access to all specialized dashboards
- Overview of available API endpoints
- Brief descriptions of each interface's purpose

## Core Scoring Interface

**URL:** `/core-scoring/`

The Core Scoring interface provides access to the fundamental scoring capabilities of DevTrack:

### Features:

- **Score Single Response**: Submit a caregiver response and milestone behavior to receive a developmental score
  - API Endpoint: `POST /score-response`
  - Provides score, domain, and confidence metrics

- **Send Manual Score**: Manually set a score for a specific milestone
  - API Endpoint: `POST /send-score`
  - Useful for expert overrides or corrections

- **Generate Assessment Report**: Create a comprehensive developmental assessment report
  - API Endpoint: `GET /generate-report`
  - Displays domain quotients and individual milestone scores

## Improved Scoring Interface

**URL:** `/improved-scoring/`

The Improved Scoring interface provides access to DevTrack's enhanced scoring capabilities:

### Features:

- **Enhanced Scoring**: Score responses with domain-specific knowledge and contextual understanding
  - API Endpoint: `POST /improved-scoring/score`
  - Provides detailed component scores and confidence metrics
  - Includes options for domain selection and age range specification

- **Expert Feedback**: Submit expert feedback on scoring results
  - API Endpoint: `POST /improved-scoring/reviews/{review_id}/feedback`
  - Contributes to continuous improvement of the scoring system

## Batch Processing Interface

**URL:** `/batch-processing/`

The Batch Processing interface enables processing multiple responses simultaneously:

### Features:

- **Batch Score Responses**: Process multiple responses in parallel
  - API Endpoint: `POST /batch-score`
  - Add responses individually or upload a CSV file
  - View results in a tabular format with color-coded scores

- **Export Results**: Download batch processing results as a CSV file
  - Includes all milestone data, scores, and full response text

- **Generate Assessment Report**: Create a comprehensive report based on batch-processed data
  - API Endpoint: `GET /generate-report`
  - Visualizes domain quotients with progress bars
  - Lists individual milestone scores in a sortable table

## Model Performance Dashboard

**URL:** `/metrics-dashboard/`

The Model Performance Dashboard provides insights into the scoring system's performance:

### Features:

- **Performance Metrics**: View accuracy, precision, recall, and F1 scores
  - API Endpoint: `GET /improved-scoring/metrics`
  - Breakdown by scoring method (keyword, semantic, transformer, LLM)

- **Domain Performance**: Analyze performance across different developmental domains
  - Visualizes domain-specific accuracy and confidence metrics

- **Active Learning Statistics**: Track system improvement over time
  - API Endpoint: `GET /active-learning/statistics`
  - Displays total examples, completed reviews, and pending reviews

## Active Learning Dashboard

**URL:** `/active-learning/`

The Active Learning Dashboard facilitates expert review and system improvement:

### Features:

- **Pending Reviews**: Browse examples that need expert review, ordered by priority
  - API Endpoint: `GET /active-learning/pending-reviews`
  - Displays response, milestone context, and predicted score

- **Provide Feedback**: Submit expert feedback on model predictions
  - API Endpoint: `POST /active-learning/feedback`
  - Includes options for corrected score and explanatory notes

- **System Statistics**: Monitor active learning progress
  - API Endpoint: `GET /active-learning/statistics`
  - Displays counts for pending reviews, completed reviews, and total examples

- **Model Versions**: Track improvements across model versions
  - API Endpoint: `GET /active-learning/model-versions`
  - Shows version history with performance metrics

- **Trigger Retraining**: Manually initiate model retraining
  - API Endpoint: `POST /active-learning/trigger-retraining`
  - Useful after submitting significant feedback

## GPU Acceleration Dashboard

**URL:** `/gpu-acceleration/`

The GPU Acceleration Dashboard provides a user-friendly interface for monitoring and controlling Metal GPU acceleration on Apple Silicon Macs.

### Features:

- **System Information**: View detailed hardware and GPU information
  - API Endpoint: `GET /gpu-acceleration/system-info`
  - Displays chip model, memory, OS version, and GPU status
  - Provides a detailed system information panel for advanced diagnostics

- **Server Control**: Manage the API server with different GPU acceleration options
  - API Endpoints: 
    - `POST /gpu-acceleration/restart-server`
    - `POST /gpu-acceleration/stop-server`
  - Start server with optimized GPU settings, basic settings, or CPU-only mode
  - Stop the running server

- **GPU Settings**: Configure and update GPU acceleration settings
  - API Endpoint: `POST /gpu-acceleration/settings`
  - Choose from optimal, basic, advanced, or custom acceleration modes
  - Apply settings with or without server restart

- **Performance Monitoring**: Visualize GPU and CPU performance metrics
  - API Endpoint: `GET /gpu-acceleration/monitoring-data`
  - Real-time GPU memory usage graph
  - CPU utilization tracking
  - Historical performance data

- **Benchmarking Tools**: Measure and compare GPU acceleration performance
  - API Endpoints:
    - `POST /gpu-acceleration/run-benchmark`
    - `GET /gpu-acceleration/benchmarks`
  - Run matrix multiplication benchmarks with configurable sizes
  - View historical benchmark results with speedup metrics
  - Compare CPU vs. GPU performance visually

- **Status Log**: Monitor operations and troubleshoot issues
  - Real-time feedback on operations
  - Error reporting with detailed messages
  - Action history for debugging

### Accessing the Dashboard:

1. Start the DevTrack API server with GPU acceleration enabled:
   ```bash
   ./start_optimized.sh
   ```

2. Navigate to the GPU Acceleration Dashboard in your web browser:
   ```
   http://localhost:8003/gpu-acceleration/
   ```

### Key Workflows:

1. **Enabling Optimal GPU Acceleration**:
   - Access the dashboard
   - In the Server Control card, click "Start Optimized"
   - Verify GPU status in the System Information card shows "Available"

2. **Running Performance Benchmarks**:
   - In the Benchmarking card, click "Run Matrix Benchmark"
   - View results in the benchmark chart
   - Click "View Results" for detailed performance metrics

3. **Monitoring GPU Resource Usage**:
   - Select the "GPU Memory" tab in the Performance Visualization section
   - Monitor real-time memory usage during operations
   - Use the data to optimize your application's GPU utilization

## Integration with API Documentation

All dashboard interfaces are designed to work seamlessly with the DevTrack API. For detailed API documentation, refer to [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Accessing the Dashboards

To access these dashboards:

1. Start the DevTrack API server:
   ```bash
   python main.py --api
   ```

2. Navigate to the appropriate URL in your web browser:
   - Main Dashboard: `http://localhost:8003/`
   - Core Scoring: `http://localhost:8003/core-scoring/`
   - Improved Scoring: `http://localhost:8003/improved-scoring/`
   - Batch Processing: `http://localhost:8003/batch-processing/`
   - Model Performance: `http://localhost:8003/metrics-dashboard/`
   - Active Learning: `http://localhost:8003/active-learning/`
