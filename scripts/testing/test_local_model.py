#!/usr/bin/env python3
"""
Test script for using the local Mistral model with llama-cpp-python.
"""

import os
import sys
import time
from llama_cpp import Llama

def main():
    # Path to the model
    model_path = os.path.join("models", "mistral-7b-instruct-v0.2.Q3_K_S.gguf")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    
    # Create Llama instance
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window size
            n_gpu_layers=-1  # Auto-detect GPU layers
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Test prompt
        test_prompt = "<s>[INST] You are a helpful assistant. Answer this question briefly: What is child development? [/INST]"
        
        print("\nSending test prompt to model...")
        gen_start_time = time.time()
        
        # Generate response
        output = llm(
            test_prompt,
            max_tokens=512,
            temperature=0.1,
            echo=False
        )
        
        gen_time = time.time() - gen_start_time
        
        # Print results
        print(f"\nGeneration completed in {gen_time:.2f} seconds")
        print("\n--- MODEL OUTPUT ---")
        print(output["choices"][0]["text"])
        print("-------------------")
        
        # Test model in RAG context
        rag_prompt = """<s>[INST] You are a developmental assessment expert. 
        
Based on the following information:

Children typically learn to crawl between 7-10 months of age. Crawling involves coordinating both sides of the body and is an important developmental milestone. It helps strengthen muscles needed for walking.

Answer this question: At what age do children typically start crawling, and why is it important? [/INST]"""
        
        print("\nTesting RAG-style prompt...")
        rag_start_time = time.time()
        
        # Generate response for RAG prompt
        rag_output = llm(
            rag_prompt,
            max_tokens=512,
            temperature=0.1,
            echo=False
        )
        
        rag_time = time.time() - rag_start_time
        
        # Print results
        print(f"\nRAG generation completed in {rag_time:.2f} seconds")
        print("\n--- RAG MODEL OUTPUT ---")
        print(rag_output["choices"][0]["text"])
        print("------------------------")
        
    except Exception as e:
        print(f"Error initializing or using the model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 