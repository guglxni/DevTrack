#!/bin/bash

# Install dependencies for LLM-based scoring
echo "Installing dependencies for LLM-based scoring..."

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install Python 3 and pip3 first."
    exit 1
fi

# Install llama-cpp-python with OpenBLAS for better performance
echo "Installing llama-cpp-python with OpenBLAS..."
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip3 install llama-cpp-python --upgrade

# Install other dependencies
echo "Installing other dependencies..."
pip3 install requests tqdm sentence-transformers

echo "Dependencies installed successfully!"
echo "You can now download a model using: python3 scripts/download_model.py" 
 