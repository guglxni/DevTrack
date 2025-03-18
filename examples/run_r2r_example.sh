#!/bin/bash

# Run R2R Enhanced Scorer Example
# This script runs the R2R enhanced scorer example with the proper environment setup

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if the local model file exists
MODEL_PATH="$PROJECT_ROOT/models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Local model file not found at: $MODEL_PATH"
    echo "The example will run in limited demo mode"
    echo ""
    read -p "Continue without local model? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please ensure the model file exists in the models directory."
        exit 1
    fi
fi

# Move to project root
cd "$PROJECT_ROOT"

# Make sure the examples directory is in PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Print a header
echo "=============================================="
echo "  R2R Enhanced Scorer Example"
echo "  Using Local Mistral LLM for Developmental Assessment"
echo "=============================================="
echo ""

# Check Python version
python3 --version
echo ""

# Run the example
python3 "$SCRIPT_DIR/r2r_scorer_example.py"

# Return to original directory
cd - > /dev/null 