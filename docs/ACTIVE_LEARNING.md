# Active Learning System Documentation

The Active Learning system in the ASD Assessment API is designed to continuously improve the scoring accuracy of developmental milestone assessments. This document provides an overview of the system, its components, and how to use it.

## Overview

The Active Learning system:

1. Identifies valuable examples for expert review
2. Prioritizes examples that would provide the most information gain
3. Tracks model versions and performance improvements
4. Provides an interface for expert feedback

## Starting the System

To start the API server with Active Learning enabled:

```bash
# Using the provided script
./start_active_learning.sh

# Or manually
export ENABLE_ACTIVE_LEARNING=true
python3 main.py --api
```

## Available Interfaces

- **Active Learning Dashboard**: http://localhost:8003/active-learning/
  - View system statistics
  - Monitor model versions
  - Access pending reviews

- **Expert Feedback Interface**: http://localhost:8003/feedback/
  - Review and provide feedback on prioritized examples
  - Submit expert assessments

## API Endpoints

The following API endpoints are available for the Active Learning system:

- `GET /active-learning/statistics` - Get system statistics
- `GET /active-learning/model-versions` - Get model version history
- `GET /active-learning/pending-reviews` - Get examples pending expert review
- `POST /active-learning/feedback` - Submit expert feedback
- `GET /active-learning/export-interface` - Get data for the feedback interface
- `POST /active-learning/trigger-retraining` - Trigger model retraining

## How It Works

### Example Selection

The system selects examples for expert review based on:

1. **Uncertainty** - Examples where the model has low confidence
2. **Disagreement** - Examples where different scoring components disagree
3. **Linguistic Novelty** - Examples with unusual linguistic patterns
4. **Domain Coverage** - Examples that help ensure balanced coverage across domains

### Information Gain Calculation

Each example is assigned a priority score based on its potential information gain:

```
information_gain = (
    uncertainty_weight * uncertainty_score +
    disagreement_weight * disagreement_score +
    novelty_weight * linguistic_novelty_score +
    coverage_weight * domain_coverage_score
)
```

### Model Versioning

The system tracks model versions and performance metrics over time. Each version includes:

- Version number (semantic versioning)
- Timestamp
- Description
- Performance metrics
- Number of training examples used

### Expert Feedback Loop

1. System identifies valuable examples
2. Experts review and provide correct scores
3. Feedback is incorporated into training data
4. When sufficient new data is collected, model is retrained
5. New model version is created with updated performance metrics

## Configuration

The Active Learning system can be configured through the following parameters:

- `uncertainty_threshold` - Threshold for considering examples with uncertain predictions
- `disagreement_threshold` - Threshold for considering examples with high disagreement
- `max_prioritized_examples` - Maximum number of examples to prioritize for review
- `info_gain_weights` - Weights for different components in information gain calculation
- `use_enhanced_ambiguity_detection` - Whether to use enhanced ambiguity detection
- `min_training_examples_per_category` - Minimum examples per category for balanced training

## Troubleshooting

If you encounter issues with the Active Learning system:

1. Check that the `ENABLE_ACTIVE_LEARNING` environment variable is set to `true`
2. Ensure the data directory exists and is writable
3. Check the logs for specific error messages
4. Verify that all required dependencies are installed

## Further Development

Future enhancements planned for the Active Learning system:

- Integration with external expert review systems
- Support for batch feedback submission
- Enhanced visualization of model improvements
- Automated performance regression testing