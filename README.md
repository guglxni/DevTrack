# DevTrack

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2.0-blue)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

<div align="center">
  <img src="https://via.placeholder.com/800x300?text=DevTrack+Platform" alt="DevTrack Platform Banner" width="800"/>
  <p><em>A comprehensive platform for developmental assessment and tracking</em></p>
</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Active Learning System](#-active-learning-system)
- [R2R Integration](#-r2r-integration)
- [Smart Scoring System](#-smart-scoring-system)
- [Directory Structure](#-directory-structure)
- [Components](#-components)
- [API Service](#-api-service)
- [Testing Framework](#-testing-framework)
- [Documentation](#-documentation)
- [License](#-license)

## 🔍 Overview

DevTrack provides a comprehensive API and demonstration web application for processing and analyzing developmental milestone assessments. The system supports question processing, keyword management, score recording, and response analysis to facilitate the assessment of developmental milestones in children.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>🧠 Advanced NLP Analysis</b></td>
      <td align="center"><b>🔄 Continuous Learning</b></td>
      <td align="center"><b>📊 Detailed Reporting</b></td>
    </tr>
    <tr>
      <td align="center">State-of-the-art language models for assessment</td>
      <td align="center">Improves accuracy over time with active learning</td>
      <td align="center">Comprehensive visualization of developmental progress</td>
    </tr>
  </table>
</div>

## ✨ Features

- **Smart Response Analysis**: Sophisticated NLP techniques to analyze caregiver responses
- **Multi-tier Scoring System**: Combines LLM, keyword-based, and ensemble approaches
- **Active Learning**: Continuously improves with expert feedback
- **R2R Integration**: Enhanced reasoning and retrieval for complex cases
- **Comprehensive Assessment API**: Unified API endpoints for all assessment needs
- **Interactive Web Demo**: User-friendly interface for demonstrating capabilities
- **Extensive Testing Framework**: Robust testing and performance reporting

## 🚀 Quick Start

For a comprehensive guide on using the API and web interface, see our [API Quick Start Guide](API_README.md).

### Starting the Application

We've provided several scripts to make it easy to start the application:

```bash
# Start the API server
./start_api_server.sh
```

<div align="center">

| Component | URL | Description |
|-----------|-----|-------------|
| API Server | http://localhost:8003 | Endpoints for assessment processing |
| Active Learning Dashboard | http://localhost:8003/active-learning/ | Review and improve model performance |
| Expert Feedback Interface | http://localhost:8003/feedback/ | Input from domain experts |
| R2R Dashboard | http://localhost:8003/r2r-dashboard/ | Enhanced reasoning visualization |

</div>

## 🧠 Active Learning System

DevTrack includes an Active Learning system that helps improve scoring accuracy over time:

- 🎯 **Targeted Learning**: Identifies valuable examples for expert review based on uncertainty and disagreement
- 📊 **Prioritization**: Focuses on examples that would provide the most information gain
- 📈 **Model Tracking**: Monitors versions and performance improvements over time
- 🔄 **Feedback Loop**: Provides an interface for expert feedback

To start the API server with Active Learning enabled:

```bash
# Using the provided script
./start_active_learning.sh

# Or manually
export ENABLE_ACTIVE_LEARNING=true
python3 main.py --api
```

## 🔍 R2R Integration

The system includes a fully implemented and production-ready R2R (Reason to Retrieve) integration that enhances developmental assessment accuracy:

<div align="center">
  <table>
    <tr>
      <td align="center">🚀</td>
      <td><b>40% improved accuracy</b> on edge cases compared to the standard system</td>
    </tr>
    <tr>
      <td align="center">📝</td>
      <td><b>Detailed reasoning</b> for each scoring decision</td>
    </tr>
    <tr>
      <td align="center">🧩</td>
      <td><b>Better handling</b> of context-dependent, contradictory, and ambiguous parent responses</td>
    </tr>
    <tr>
      <td align="center">🔍</td>
      <td><b>Advanced retrieval</b> capabilities for developmental research and guidelines</td>
    </tr>
  </table>
</div>

To start the API server with R2R enabled:

```bash
# Using the provided script
./start_with_r2r.sh

# Or manually
export ENABLE_R2R=true
python3 main.py --api
```

For detailed information and benchmarking results, see the [R2R Integration Documentation](docs/R2R_INTEGRATION.md).

## 🎯 Smart Scoring System

DevTrack's smart scoring system provides reliable and accurate assessments through multiple approaches:

### Enhanced Keyword Scoring

The system detects various response patterns:

| Response Type | Keywords | Score |
|---------------|----------|-------|
| **Positive** | "yes", "always", "consistently" | INDEPENDENT |
| **Emerging** | "sometimes", "occasionally", "starting to" | EMERGING |
| **Support-needed** | "with help", "with assistance" | WITH_SUPPORT |
| **Regression** | "used to", "stopped", "regressed" | LOST_SKILL |
| **Negative** | "no", "never", "not at all" | CANNOT_DO |

### Tiered Scoring Approach

<div align="center">
  <img src="https://via.placeholder.com/600x200?text=Scoring+Pipeline+Diagram" alt="Scoring Pipeline" width="600"/>
</div>

1. **Primary**: LLM-based scoring (when available)
2. **Secondary**: Keyword-based scoring as reliable fallback
3. **Tertiary**: Ensemble scoring for uncertain cases

### Advanced Techniques

- ✅ **Word boundary-aware keyword matching**: Prevents false matches
- ✅ **Negation detection**: Correctly identifies negated phrases
- ✅ **Milestone-specific pattern matching**: Uses custom patterns for each milestone type
- ✅ **Special phrase handling**: Correctly interprets complex phrases

For more details on the reliable scoring system, see [Reliable Scoring Documentation](docs/RELIABLE_SCORING.md).

## 📁 Directory Structure

```
project/
├── src/                  # Core source code
│   ├── api/              # FastAPI application code
│   ├── core/             # Core scoring and NLP functionality
│   ├── utils/            # Utility functions and helpers
│   ├── testing/          # Testing framework and tools
├── scripts/              # Scripts for running, testing, managing
│   ├── start/            # Scripts for starting servers
│   ├── test/             # Scripts for running tests
│   ├── debug/            # Scripts for debugging issues
│   └── fixes/            # Scripts for fixing specific issues
├── tests/                # Python test files
├── examples/             # Example usage of the API
├── docs/                 # Documentation files
├── data/                 # Data files for milestones and scoring
├── test_data/            # Test data for API testing
├── test_results/         # Output directory for test results
└── logs/                 # Log files
```

## 🧩 Components

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%"><b>🔄 API Service</b></td>
      <td align="center" width="50%"><b>🧪 Testing Framework</b></td>
    </tr>
    <tr>
      <td>FastAPI-based REST API for developmental milestone assessments</td>
      <td>Comprehensive testing tools and performance reporting</td>
    </tr>
  </table>
</div>

## 🔄 API Service

The API service provides the following endpoints:

| Endpoint | Description |
|----------|-------------|
| `/question` | Processes questions related to developmental milestones |
| `/keywords` | Manages keywords used for analyzing responses |
| `/send-score` | Records scores for specific developmental milestones |
| `/score-response` | Analyzes parent/caregiver responses |
| `/comprehensive-assessment` | Combines all functionality in a single call |

For detailed API documentation, see [API Documentation](docs/API_DOCUMENTATION.md).

### Running the API Service

```bash
# Using Python directly
python3 -m uvicorn src.api.app:app --port 8003

# Or with the provided script
./scripts/start/start_api.sh
```

## 🧪 Testing Framework

The testing framework includes tools for API testing and performance reporting:

<div align="center">
  <img src="https://via.placeholder.com/600x300?text=Test+Results+Dashboard" alt="Test Results Dashboard" width="600"/>
</div>

### Running Tests

```bash
# Run all tests
./run_tests.sh

# Test a single endpoint
./scripts/test/test_single_endpoint.sh /endpoint_path iterations data_file

# Test the comprehensive endpoint
./scripts/test/test_comprehensive_endpoint.sh test_data/comprehensive_test.json

# Run tests for all endpoints
./scripts/test/run_all_tests.sh
```

Test results are saved to the `test_results` directory, including JSON results, performance charts, and HTML reports.

For detailed testing documentation, see [Testing Documentation](docs/README_TESTING.md).

## 📚 Documentation

Additional documentation is available in the `docs/` directory:

- [API Documentation](docs/API_DOCUMENTATION.md)
- [R2R Integration Documentation](docs/R2R_INTEGRATION.md)
- [Reliable Scoring Documentation](docs/RELIABLE_SCORING.md)
- [Testing Documentation](docs/README_TESTING.md)
- [API Quick Start Guide](API_README.md)

## 📄 License

This software is proprietary and owned by Aaryan Guglani. All rights reserved.

Copyright (c) 2025 Aaryan Guglani

This software is provided under a proprietary license. See the [LICENSE](LICENSE) file for details.

## Contact

For inquiries regarding commercial use or licensing, please contact guglaniaaryan@gmail.com

---

<div align="center">
  <p>Made with ❤️ for improving developmental assessment and tracking</p>
  <p>© 2025 Aaryan Guglani</p>
</div>