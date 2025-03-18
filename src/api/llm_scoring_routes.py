"""
LLM Scoring Routes

This module provides direct access to LLM-based scoring capabilities,
bypassing the ensemble scoring mechanism for more advanced reasoning.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import traceback

from src.core.scoring.base import Score
from src.core.scoring.llm_scorer import LLMBasedScorer
from src.core.knowledge import format_prompt_with_context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_scoring_routes")

# Create router
router = APIRouter(prefix="/llm-scoring", tags=["llm-scoring"])

# Define a local fallback format function in case the import fails
def format_with_age_domain_context(response: str, milestone_context: Dict[str, Any]) -> str:
    """
    Fallback function to format a prompt with age and domain context.
    This is used when the import from knowledge engineering module fails.
    
    Args:
        response: The response text from the parent
        milestone_context: Context about the milestone being assessed
        
    Returns:
        A formatted prompt string
    """
    domain = milestone_context.get("domain", "Unknown")
    age_range = milestone_context.get("age_range", "Unknown")
    behavior = milestone_context.get("behavior", "Unknown milestone")
    
    # Create a basic prompt template
    prompt = f"""
    You are a developmental assessment expert analyzing a parent's response about their child's developmental milestone.
    
    Milestone: {behavior}
    Domain: {domain}
    Age Range: {age_range}
    
    Parent's Response: "{response}"
    
    Based solely on the parent's response, please assess the child's ability with this milestone.
    Rate the child's ability on a scale of 1-7, where:
    1-2 = CANNOT_DO (Child cannot perform this skill at all)
    3 = LOST_SKILL (Child used to have this skill but has lost it)
    4-5 = EMERGING (Child is beginning to show this skill sometimes)
    6 = WITH_SUPPORT (Child can do this with help or prompting)
    7 = INDEPENDENT (Child consistently performs this skill independently)
    
    Provide your assessment in this format:
    Score: [1-7]
    Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
    Reasoning: [Your explanation]
    """
    
    return prompt

# Models
class LLMScoringRequest(BaseModel):
    """Request model for LLM-based scoring."""
    response: str = Field(..., description="Parent/caregiver response describing the child's behavior")
    milestone_behavior: str = Field(..., description="The milestone behavior being assessed")
    domain: Optional[str] = Field(None, description="Developmental domain (e.g., 'GM', 'FM', 'RL')")
    age_range: Optional[str] = Field(None, description="Age range for the milestone (e.g., '0-6 months')")
    criteria: Optional[str] = Field(None, description="Specific criteria for the milestone")

class DirectTestRequest(BaseModel):
    """Request model for direct testing of LLM scoring."""
    question: str = Field(..., description="The question about the milestone")
    milestone: str = Field(..., description="The milestone behavior being assessed")
    response: str = Field(..., description="Parent/caregiver response describing the child's behavior")
    domain: Optional[str] = Field(None, description="Developmental domain (e.g., 'GM', 'FM', 'RL')")
    age_range: Optional[str] = Field(None, description="Age range for the milestone (e.g., '0-6 months')")

class LLMScoringResponse(BaseModel):
    """Response model for LLM-based scoring."""
    score: int = Field(..., description="The determined score (0-4)")
    score_label: str = Field(..., description="The score category (e.g., INDEPENDENT)")
    confidence: float = Field(..., description="Confidence level of the score determination (0-1)")
    reasoning: str = Field(..., description="Reasoning behind the score determination")
    generated_text: Optional[str] = Field(None, description="Raw generated text from the LLM")

# Dependency for LLM scorer
def get_llm_scorer() -> LLMBasedScorer:
    """
    Get the LLM-based scorer.
    
    Returns:
        LLMBasedScorer instance
    """
    try:
        if os.environ.get("LLM_MODEL_PATH"):
            model_path = os.environ.get("LLM_MODEL_PATH")
            logger.info(f"Using local model at {model_path}")
            # Configure for local model
            config = {
                "model_path": model_path,
                "n_ctx": 2048,
                "n_batch": 512,
                "n_gpu_layers": 0,  # Use CPU by default
                "f16_kv": True,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 256,
                "n_threads": 4  # Explicitly set number of threads
            }
            
            # Use Apple Silicon GPU if available
            if os.environ.get("USE_METAL", "false").lower() == "true":
                config["n_gpu_layers"] = -1  # Use all layers on GPU
                logger.info("Using Metal GPU for LLM inference")
                
            return LLMBasedScorer(config)
        elif os.environ.get("OPENAI_API_KEY"):
            # Configure for OpenAI API
            config = {
                "use_openai": True,
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "model_name": os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 256
            }
            return LLMBasedScorer(config)
        else:
            # No configuration available
            raise ValueError("No valid LLM configuration found. Set LLM_MODEL_PATH or OPENAI_API_KEY.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM scorer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM scorer: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Check if the LLM scoring service is available.
    
    Returns:
        JSON response with status
    """
    try:
        # Check if model path is set
        if os.environ.get("LLM_MODEL_PATH"):
            model_path = os.environ.get("LLM_MODEL_PATH")
            if os.path.exists(model_path):
                # Extract model name from path
                model_name = os.path.basename(model_path)
                return {"status": "available", "mode": "local_model", "model": model_name}
            else:
                return {"status": "error", "message": f"Model file not found at {model_path}"}
        elif os.environ.get("OPENAI_API_KEY"):
            # For OpenAI, use a default model name or the one specified in env
            openai_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
            return {"status": "available", "mode": "openai_api", "model": openai_model}
        else:
            return {"status": "error", "message": "No LLM configuration found"}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/score", response_model=LLMScoringResponse)
async def score_with_llm(
    request: LLMScoringRequest,
    scorer: LLMBasedScorer = Depends(get_llm_scorer)
):
    """
    Score a parent's response using the LLM-based scorer.
    
    Args:
        request: The scoring request with parent response and milestone context
        scorer: The LLM scorer dependency
        
    Returns:
        Scoring result with score, confidence, and reasoning
    """
    try:
        # Create milestone context dictionary
        milestone_context = {
            "behavior": request.milestone_behavior,
            "domain": request.domain,
            "age_range": request.age_range,
            "criteria": request.criteria
        }
        
        # Get the score from the LLM - the scorer will internally use our template functions
        logger.info(f"Scoring response with LLM for milestone: {request.milestone_behavior}")
        result = scorer.score(request.response, milestone_context)
        
        # Extract generated text from details if available
        generated_text = None
        if result.details and "generated_text" in result.details:
            generated_text = result.details["generated_text"]
        
        return {
            "score": result.score.value,
            "score_label": result.score.name,
            "confidence": result.confidence,
            "reasoning": result.reasoning or "No reasoning provided",
            "generated_text": generated_text
        }
    except Exception as e:
        logger.error(f"Error in LLM scoring: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in LLM scoring: {str(e)}")

@router.post("/batch-score")
async def batch_score_with_llm(
    requests: List[LLMScoringRequest],
    scorer: LLMBasedScorer = Depends(get_llm_scorer)
):
    """
    Score multiple parent responses in batch using the LLM-based scorer.
    
    Args:
        requests: List of scoring requests
        scorer: The LLM scorer dependency
        
    Returns:
        List of scoring results
    """
    try:
        results = []
        
        for request in requests:
            try:
                # Create milestone context dictionary
                milestone_context = {
                    "behavior": request.milestone_behavior,
                    "domain": request.domain,
                    "age_range": request.age_range,
                    "criteria": request.criteria
                }
                
                # Get the score from the LLM - the scorer will internally use our template functions
                logger.info(f"Scoring response with LLM for milestone: {request.milestone_behavior}")
                result = scorer.score(request.response, milestone_context)
                
                # Extract generated text from details if available
                generated_text = None
                if result.details and "generated_text" in result.details:
                    generated_text = result.details["generated_text"]
                
                results.append({
                    "score": result.score.value,
                    "score_label": result.score.name,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning or "No reasoning provided",
                    "generated_text": generated_text
                })
            except Exception as e:
                logger.error(f"Error scoring request: {e}")
                results.append({
                    "error": str(e),
                    "score": -1,
                    "score_label": "NOT_RATED",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                })
        
        return results
    except Exception as e:
        logger.error(f"Error in batch LLM scoring: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in batch LLM scoring: {str(e)}")

@router.post("/direct-test", response_model=LLMScoringResponse)
async def direct_test(
    request: DirectTestRequest,
    scorer: LLMBasedScorer = Depends(get_llm_scorer)
):
    """
    Test the LLM scoring directly with a question, milestone, and response.
    
    Args:
        request: The request containing the question, milestone, and response
        
    Returns:
        LLMScoringResponse: The scoring result
    """
    logger.info(f"Direct test for milestone: {request.milestone}")
    
    # Create milestone context
    milestone_context = {
        "behavior": request.milestone,
        "domain": request.domain,
        "age_range": request.age_range,
        "criteria": None
    }
    
    try:
        # Score the response
        result = scorer.score(request.response, milestone_context)
        
        # Extract generated text if available
        generated_text = None
        if hasattr(result, 'details') and result.details and 'generated_text' in result.details:
            generated_text = result.details['generated_text']
        
        # Return the result
        return {
            "score": result.score.value,
            "score_label": result.score.name,
            "confidence": result.confidence,
            "reasoning": result.reasoning or "No reasoning provided",
            "generated_text": generated_text
        }
    except Exception as e:
        logger.error(f"Error in direct test: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in direct test: {str(e)}")

def add_routes_to_app(app):
    """
    Add LLM scoring routes to the FastAPI app.
    
    Args:
        app: The FastAPI application
    """
    app.include_router(router)
    logger.info("LLM scoring routes have been registered") 
 