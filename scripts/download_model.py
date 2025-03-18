#!/usr/bin/env python3
"""
Download Mistral Model Script

This script downloads the Mistral 7B model for local inference.
"""

import os
import sys
import argparse
import requests
from pathlib import Path

# Model URLs
MODEL_URLS = {
    "mistral-7b-instruct-v0.2.Q3_K_S.gguf": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q3_K_S.gguf",
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "mistral-7b-instruct-v0.2.Q5_K_M.gguf": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
}

def download_file(url, destination):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading {url} to {destination}")
    print(f"File size: {total_size / (1024 * 1024):.2f} MB")
    
    with open(destination, 'wb') as file:
        downloaded = 0
        for data in response.iter_content(block_size):
            size = file.write(data)
            downloaded += size
            percent = int(100 * downloaded / total_size)
            sys.stdout.write(f"\r{percent}% downloaded")
            sys.stdout.flush()
    
    print("\nDownload complete!")

def main():
    parser = argparse.ArgumentParser(description="Download Mistral model for local inference")
    parser.add_argument(
        "--model", 
        choices=list(MODEL_URLS.keys()), 
        default="mistral-7b-instruct-v0.2.Q3_K_S.gguf",
        help="Model version to download (default: mistral-7b-instruct-v0.2.Q3_K_S.gguf)"
    )
    parser.add_argument(
        "--output-dir", 
        default="models",
        help="Directory to save the model (default: models)"
    )
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the model
    model_url = MODEL_URLS[args.model]
    output_path = output_dir / args.model
    
    if output_path.exists():
        print(f"Model already exists at {output_path}")
        overwrite = input("Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Download cancelled")
            return
    
    try:
        download_file(model_url, output_path)
        print(f"Model downloaded successfully to {output_path}")
        
        # Set environment variable
        print(f"\nTo use this model, set the following environment variable:")
        print(f"export LLM_MODEL_PATH={output_path}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
