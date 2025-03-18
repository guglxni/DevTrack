# Developmental Milestone Scoring System - Testing & Benchmarking Framework

This directory contains a comprehensive testing and benchmarking framework for the improved developmental milestone scoring system. The framework allows for systematic evaluation of both accuracy and performance across all components of the scoring pipeline.

## Overview

The framework consists of the following main components:

1. **Unit Testing** - Tests for individual scoring components and their behavior
2. **Test Data Generation** - Tools to create realistic test datasets with known expected scores
3. **Performance Benchmarking** - Tools to measure throughput, latency, and memory usage
4. **Accuracy Benchmarking** - Tools to measure accuracy compared to gold standard data
5. **Configuration Optimization** - Tools to find optimal parameter settings for the scoring system
6. **LLM Integration** - Tools for working with the Mistral 7B LLM for advanced scoring

## Getting Started

### Installation

Ensure you have all necessary dependencies installed:

```bash
pip install pytest pytest-benchmark matplotlib pandas numpy tqdm psutil
```

### Quick Start

To run all tests and benchmarks:

```bash
python src/testing/run_scoring_tests.py all
```

This will run unit tests, generate test data, and run both performance and accuracy benchmarks.

## Command Reference

### Running Unit Tests

Run unit tests for the scoring components:

```bash
python src/testing/run_scoring_tests.py unit
```

Options:
- `--verbose` or `-v`: Show detailed test output
- `--filter` or `-k`: Filter tests by keyword
- `--skip-slow`: Skip tests marked as slow

### Generating Test Data

Generate synthetic test data for benchmarking:

```bash
python src/testing/run_scoring_tests.py generate --count 200 --include-edge-cases
```

Options:
- `--output`: Path to save the generated data (default: `test_data/scoring/benchmark_data.json`)
- `--count`: Number of test cases to generate
- `--domain`: Specific domain to generate data for (`motor`, `communication`, `social`, `cognitive`)
- `--include-edge-cases`: Include challenging edge cases in the dataset

### Running Benchmarks

#### Performance Benchmarks

Measure performance metrics like throughput, latency, and memory usage:

```bash
python src/testing/run_scoring_tests.py benchmark --type performance --memory
```

Options:
- `--data`: Path to test data file
- `--output`: Prefix for output filenames
- `--threads`: Number of threads for throughput testing
- `--memory`: Include memory usage benchmarks

#### Accuracy Benchmarks

Measure accuracy metrics compared to gold standard data:

```bash
python src/testing/run_scoring_tests.py benchmark --type accuracy --confusion --components
```

Options:
- `--data`: Path to test data file
- `--output`: Prefix for output filenames
- `--confusion`: Generate confusion matrix
- `--components`: Compare individual scoring components

#### Configuration Benchmarks

Test different parameter combinations to find optimal settings:

```bash
python src/testing/run_scoring_tests.py benchmark --type config --params param_grid.json
```

Options:
- `--data`: Path to test data file
- `--output`: Prefix for output filenames
- `--params`: Path to parameter grid JSON file (if not specified, a default grid will be created)

#### LLM Prompt Optimization

Optimize the prompting strategy for the LLM-based scorer:

```bash
python src/testing/run_scoring_tests.py benchmark --type llm-prompts --data test_data.json
```

Options:
- `--data`: Path to test data file
- `--output`: Prefix for output filenames

### Testing the LLM-based Scorer

Test the LLM-based scorer with a specific response and milestone:

```bash
python src/testing/run_scoring_tests.py llm --response "My child can do this independently" --milestone "Walks without support" --verbose
```

Options:
- `--response`: The text of the parent/caregiver response
- `--milestone`: The milestone behavior description
- `--criteria`: Optional criteria for the milestone
- `--age-range`: Optional age range for the milestone
- `--verbose` or `-v`: Show detailed output including reasoning

## Test Data Format

The test data should be in JSON format with the following structure:

```json
[
  {
    "response": "Yes, my child can do this independently",
    "milestone_context": {
      "id": "motor_01",
      "domain": "motor",
      "behavior": "Walks independently",
      "criteria": "Child walks without support for at least 10 steps",
      "age_range": "12-18 months"
    },
    "expected_score": "INDEPENDENT",
    "expected_score_value": 4
  },
  ...
]
```

## Output and Reports

The benchmarking framework generates several types of outputs:

1. **JSON Results** - Raw benchmark data in JSON format
2. **HTML Reports** - Human-readable reports with tables and charts
3. **Visualizations** - PNG images showing key metrics

All outputs are saved to the `test_results/benchmarks/` directory by default.

## Advanced Usage

### Custom Parameter Grids

For configuration benchmarks, you can specify a custom parameter grid in JSON format:

```json
{
  "enable_keyword_scorer": [true, false],
  "enable_embedding_scorer": [true, false],
  "score_weights": [
    {"keyword": 0.7, "embedding": 0.3},
    {"keyword": 0.5, "embedding": 0.5},
    {"keyword": 0.3, "embedding": 0.7}
  ],
  "keyword_scorer": [
    {"confidence_threshold": 0.6},
    {"confidence_threshold": 0.7},
    {"confidence_threshold": 0.8}
  ]
}
```

### Extending the Framework

The framework is designed to be extensible. To add new tests or benchmarks:

1. Add new test cases to `test_scoring_framework.py`
2. Add new benchmark methods to `benchmark_framework.py`
3. Update the command line interface in `run_scoring_tests.py`

### Using the LLM-based Scorer

The Mistral 7B model provides advanced reasoning capabilities for scoring responses. To enable it, modify your configuration:

```python
config = {
    "enable_llm_scorer": True,
    "score_weights": {
        "keyword": 0.3,
        "embedding": 0.2,
        "llm": 0.5  # Higher weight for LLM
    },
    "llm_scorer": {
        "temperature": 0.1,
        "gpu_layers": 0  # Increase to use GPU acceleration
    }
}

engine = ImprovedDevelopmentalScoringEngine(config)
```

The LLM-based scorer may be slower but provides more detailed reasoning and can handle complex linguistic patterns.

### LLM Requirements

To use the LLM-based scorer, you need:

1. The `llama-cpp-python` package installed:
   ```bash
   pip install llama-cpp-python
   ```

2. The Mistral 7B model file in the `/models` directory:
   - For this project, we're using the quantized GGUF version: `mistral-7b-instruct-v0.2.Q3_K_S.gguf`

## Best Practices

1. **Generate Diverse Test Data** - Include samples across all scoring categories
2. **Include Edge Cases** - Test with challenging inputs
3. **Run Benchmarks Regularly** - Track performance over time
4. **Version Test Data** - Keep gold standard datasets versioned
5. **Automate Benchmarking** - Include benchmarks in your CI/CD pipeline

## Troubleshooting

If you encounter issues:

- Ensure all dependencies are installed
- Check that the scoring system is correctly importable
- Verify test data is in the expected format
- Check logs in the `logs/` directory

## Contributing

To contribute to the testing framework:

1. Add new test cases for any edge cases or bugs discovered
2. Improve test data generation to include more realistic scenarios
3. Enhance visualization and reporting capabilities
4. Add new benchmark types as needed 