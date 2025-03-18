"""
API dependencies for FastAPI.

This module provides dependency injection functions for the FastAPI application.
"""

import logging
import os
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def get_scoring_engine():
    """Get the scoring engine from the app module."""
    try:
        from src.app import engine
        logger.info("Successfully retrieved engine from src.app")
        return engine
    except ImportError:
        logger.error("Failed to import engine from src.app")
        try:
            from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
            
            # Set the model path environment variable
            os.environ["LLM_MODEL_PATH"] = os.path.join(os.getcwd(), "models", "mistral-7b-instruct-v0.2.Q3_K_S.gguf")
            
            logger.info(f"Creating new engine instance with continuous learning enabled and model path: {os.environ['LLM_MODEL_PATH']}")
            
            return ImprovedDevelopmentalScoringEngine({
                "enable_continuous_learning": True,
                "enable_llm_scorer": True,
                "llm_scorer": {
                    "model_path": os.environ["LLM_MODEL_PATH"],
                    "n_ctx": 2048,
                    "n_batch": 512,
                    "n_gpu_layers": 0,
                    "n_threads": 4,
                    "f16_kv": True,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 256
                }
            })
        except ImportError:
            logger.error("Failed to import ImprovedDevelopmentalScoringEngine")
            raise HTTPException(status_code=500, detail="Scoring engine not available") 