import asyncio
from typing import Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from src.core.enhanced_assessment_engine import EnhancedAssessmentEngine, Score, DevelopmentalMilestone
import argparse
import os
import re
import functools
import json
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer

# Initialize the enhanced assessment engine
engine = EnhancedAssessmentEngine(use_embeddings=True)
print("Assessment engine initialized and ready")

# Import our new hybrid scorer for reliable scoring
try:
    from src.api.hybrid_scorer import score_response as hybrid_score_response
    HYBRID_SCORER_AVAILABLE = True
    print("Hybrid scorer available - using enhanced scoring for better reliability")
except ImportError as e:
    HYBRID_SCORER_AVAILABLE = False
    print(f"Hybrid scorer not available: {str(e)} - falling back to keyword-based scoring")

app = FastAPI(
    title="ASD Developmental Milestone Assessment API",
    description="API for assessing developmental milestones in children",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class ChildInfo(BaseModel):
    age: int = Field(..., description="Child's age in months", ge=0, le=36)
    name: Optional[str] = Field(None, description="Child's name (optional)")

class MilestoneResponse(BaseModel):
    response: str = Field(..., description="Caregiver's response describing the child's behavior")
    milestone_behavior: str = Field(..., description="The behavior being assessed")

class ResponseBatch(BaseModel):
    responses: List[MilestoneResponse] = Field(..., description="Batch of responses to analyze")

class ScoreResult(BaseModel):
    milestone: str = Field(..., description="Milestone behavior")
    domain: str = Field(..., description="Developmental domain")
    score: int = Field(..., description="Numeric score")
    score_label: str = Field(..., description="Score label (e.g., INDEPENDENT)")
    age_range: str = Field(..., description="Age range for this milestone")

class ReportResult(BaseModel):
    scores: List[ScoreResult] = Field(..., description="Individual milestone scores")
    domain_quotients: Dict[str, float] = Field(..., description="Domain quotients (percentages)")

class BatchResponseData(BaseModel):
    responses: List[MilestoneResponse] = Field(..., description="Batch of responses to analyze")

# New API models for the requested endpoints
class Question(BaseModel):
    text: str = Field(..., description="The question text")
    milestone_id: Optional[str] = Field(None, description="Associated milestone ID (optional)")

class KeywordCategory(BaseModel):
    category: str = Field(..., description="Scoring category (e.g., CANNOT_DO)")
    keywords: List[str] = Field(..., description="List of keywords for this category")

class ScoreData(BaseModel):
    milestone_id: str = Field(..., description="The milestone ID")
    score: int = Field(..., description="The numeric score value (0-4)")
    score_label: str = Field(..., description="The score label (e.g., CANNOT_DO)")

# New comprehensive assessment model that combines multiple endpoints
class ComprehensiveAssessment(BaseModel):
    question: str = Field(..., description="The question text about the child's behavior")
    milestone_behavior: str = Field(..., description="The milestone behavior being assessed")
    parent_response: str = Field(..., description="Parent/caregiver response describing the child's behavior")
    keywords: Optional[Dict[str, List[str]]] = Field(None, description="Optional dictionary of keywords by category")

class ComprehensiveResult(BaseModel):
    question_processed: bool = Field(..., description="Whether the question was successfully processed")
    milestone_found: bool = Field(..., description="Whether the milestone was found")
    milestone_details: Optional[Dict] = Field(None, description="Details about the milestone if found")
    keywords_updated: Optional[List[str]] = Field(None, description="Categories that were updated with new keywords")
    score: int = Field(..., description="The determined score (0-4)")
    score_label: str = Field(..., description="The score category (e.g., INDEPENDENT)")
    confidence: float = Field(..., description="Confidence level of the score determination (0-1)")
    domain: Optional[str] = Field(None, description="Developmental domain of the milestone")

@app.post("/set-child-age", status_code=200)
async def set_child_age(child_info: ChildInfo):
    """Set the child's age to filter appropriate milestones"""
    try:
        engine.set_child_age(child_info.age)
        return {"message": f"Child age set to {child_info.age} months", "total_milestones": len(engine.active_milestones)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting child age: {str(e)}")

@app.get("/next-milestone", response_model=dict)
async def get_next_milestone():
    """Get the next milestone to assess"""
    try:
        milestone = engine.get_next_milestone()
        if not milestone:
            return {"message": "No more milestones to assess", "complete": True}
        
        return {
            "behavior": milestone.behavior,
            "criteria": milestone.criteria,
            "domain": milestone.domain,
            "age_range": milestone.age_range,
            "complete": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting next milestone: {str(e)}")

@app.post("/score-response")
async def score_response(response_data: MilestoneResponse):
    """
    Analyze a single response to determine the milestone score
    
    This endpoint processes a parent/caregiver response about a specific milestone behavior 
    and returns a score for how well the child can perform the milestone.
    """
    try:
        # Validate the response data
        if not response_data.response or not response_data.milestone_behavior:
            raise HTTPException(status_code=400, detail="Missing response or milestone behavior")
        
        # Try to use the hybrid scorer for more reliable scoring if available
        if HYBRID_SCORER_AVAILABLE:
            # Use our enhanced hybrid scoring approach
            result = hybrid_score_response(response_data.milestone_behavior, response_data.response)
            
            # Find the milestone for the domain
            milestone = engine.find_milestone_by_name(response_data.milestone_behavior)
            domain = milestone.domain if milestone else None
            
            # Return the score with the format expected by the API
            return {
                "milestone": response_data.milestone_behavior,
                "domain": domain,
                "score": result["score"],
                "score_label": result["score_label"],
                "confidence": result["confidence"]
            }
        else:
            # Fall back to the original scoring logic
            milestone = engine.find_milestone_by_name(response_data.milestone_behavior)
            if not milestone:
                raise HTTPException(status_code=404, detail=f"Milestone '{response_data.milestone_behavior}' not found")
            
            # Use the assessment engine to analyze the response
            score = await engine.analyze_response(response_data.response, milestone)
            
            # Set the score for the milestone
            engine.set_milestone_score(milestone, score)
            
            # Return the score info
            return {
                "milestone": milestone.behavior,
                "domain": milestone.domain,
                "score": score.value,
                "score_label": score.name,
                "confidence": 0.8  # Default confidence
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring response: {str(e)}")

@app.post("/batch-score")
async def batch_score(batch_data: BatchResponseData):
    """Score multiple responses in parallel"""
    if engine is None:
        raise HTTPException(status_code=500, detail="Assessment engine not initialized")
    
    results = []
    
    # Process each response
    for item in batch_data.responses:
        # Score the response using our enhanced method
        score = engine.score_response(item.milestone_behavior, item.response)
        
        # Find the milestone to get domain info
        milestone = engine.find_milestone_by_name(item.milestone_behavior)
        
        if milestone:
            results.append({
                "milestone": milestone.behavior,
                "domain": milestone.domain,
                "score": score.value,
                "score_label": score.name
            })
    
    return results

@app.get("/generate-report", response_model=ReportResult)
async def generate_report():
    """Generate a comprehensive assessment report"""
    try:
        df, domain_quotients = engine.generate_report()
        
        # Convert DataFrame to list of dictionaries
        scores = []
        for _, row in df.iterrows():
            # Only include assessed milestones
            if row['Score Label'] != 'NOT_RATED':
                scores.append({
                    "milestone": row['Behavior'],
                    "domain": row['Domain'],
                    "score": int(row['Score']),
                    "score_label": row['Score Label'],
                    "age_range": row['Age Range']
                })
        
        return {
            "scores": scores,
            "domain_quotients": domain_quotients
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post("/reset", status_code=200)
async def reset_assessment():
    """Reset the assessment engine for a new assessment"""
    engine.reset_scores()
    return {"message": "Assessment engine reset"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API server is running"}

@app.get("/all-milestones")
async def get_all_milestones():
    """Get all available milestone behaviors"""
    if engine is None:
        raise HTTPException(status_code=500, detail="Assessment engine not initialized")
    
    milestones = []
    # Check if active_milestones exists, otherwise use milestones
    if hasattr(engine, 'active_milestones') and engine.active_milestones:
        milestone_list = engine.active_milestones
    else:
        milestone_list = engine.milestones
    
    for milestone in milestone_list:
        milestones.append({
            "behavior": milestone.behavior,
            "criteria": milestone.criteria,
            "domain": milestone.domain,
            "age_range": milestone.age_range
        })
    
    return {"milestones": milestones}

@app.post("/question", status_code=200)
async def receive_question(question: Question):
    """
    Endpoint to receive and process questions
    
    This can be used to submit questions about a child's behavior for assessment
    """
    try:
        # Log or process the received question
        print(f"Received question: {question.text}")
        
        # If a milestone ID is provided, find the corresponding milestone
        milestone = None
        if question.milestone_id:
            # Here you might want to find the milestone by ID if you implement IDs
            # For now, we can use the behavior as the ID
            milestone = engine.find_milestone_by_name(question.milestone_id)
        
        return {
            "status": "success",
            "message": "Question received successfully",
            "question": question.text,
            "milestone_found": milestone is not None,
            "milestone_details": milestone.__dict__ if milestone else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/keywords", status_code=200)
async def update_keywords(keyword_data: KeywordCategory):
    """
    Endpoint to receive and update keywords for a scoring category
    
    This allows updating the keywords used for automatic scoring of responses
    """
    try:
        category = keyword_data.category
        keywords = keyword_data.keywords
        
        # Validate the category
        valid_categories = [score.name for score in Score]
        if category not in valid_categories:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category: {category}. Valid categories are: {', '.join(valid_categories)}"
            )
        
        # Find the score enum from the category name
        score_enum = None
        for score in Score:
            if score.name == category:
                score_enum = score
                break
        
        if not score_enum:
            raise HTTPException(status_code=500, detail=f"Error finding score enum for category: {category}")
        
        # Update the phrase map in the engine's keyword cache
        # This is a simplified approach - in a production system you might want a more robust solution
        # that persists these keywords and loads them on startup
        
        # Iterate through all milestone keys in the cache
        for milestone_key in engine._scoring_keywords_cache:
            # Get the keyword map for this milestone
            keyword_map = engine._scoring_keywords_cache[milestone_key]
            
            # Remove any existing keywords for this category
            keys_to_remove = []
            for key, score in keyword_map.items():
                if score == score_enum:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del keyword_map[key]
            
            # Add the new keywords
            for keyword in keywords:
                keyword_map[keyword.lower()] = score_enum
        
        return {
            "status": "success",
            "message": f"Keywords for category {category} updated successfully",
            "category": category,
            "keywords": keywords
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating keywords: {str(e)}")

@app.post("/send-score", status_code=200)
async def send_score(score_data: ScoreData):
    """
    Endpoint to send a score for a specific milestone
    
    This allows manual scoring of a milestone instead of using the automatic scoring system
    """
    try:
        # Validate the score value
        if score_data.score < 0 or score_data.score > 4:
            raise HTTPException(status_code=400, detail="Score must be between 0 and 4")
        
        # Find the milestone by ID (behavior in this case)
        milestone = engine.find_milestone_by_name(score_data.milestone_id)
        if not milestone:
            raise HTTPException(status_code=404, detail=f"Milestone '{score_data.milestone_id}' not found")
        
        # Convert the numeric score to the Score enum
        score_enum = None
        for score in Score:
            if score.value == score_data.score:
                score_enum = score
                break
        
        if not score_enum:
            raise HTTPException(status_code=400, detail=f"Invalid score value: {score_data.score}")
        
        # Validate that the score label matches the enum name
        if score_enum.name != score_data.score_label:
            raise HTTPException(
                status_code=400, 
                detail=f"Score label '{score_data.score_label}' does not match the expected label '{score_enum.name}' for score value {score_data.score}"
            )
        
        # Set the score for the milestone
        engine.set_milestone_score(milestone, score_enum)
        
        return {
            "status": "success",
            "message": f"Score for milestone '{score_data.milestone_id}' set successfully",
            "milestone": milestone.behavior,
            "domain": milestone.domain,
            "score": score_data.score,
            "score_label": score_data.score_label
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting score: {str(e)}")

@app.post("/comprehensive-assessment", status_code=200, response_model=ComprehensiveResult)
async def comprehensive_assessment(assessment_data: ComprehensiveAssessment):
    """
    Comprehensive endpoint that combines question processing, keyword management,
    response analysis, and score recording in a single call.
    
    This endpoint provides a streamlined way to process a full assessment in one request.
    """
    try:
        # Step 1: Process the question (similar to /question endpoint)
        print(f"Processing question: {assessment_data.question}")
        milestone = engine.find_milestone_by_name(assessment_data.milestone_behavior)
        
        if not milestone:
            raise HTTPException(status_code=404, detail=f"Milestone '{assessment_data.milestone_behavior}' not found")
        
        milestone_details = {
            "behavior": milestone.behavior,
            "criteria": milestone.criteria,
            "domain": milestone.domain,
            "age_range": milestone.age_range
        }
        
        # Step 2: Update keywords if provided (similar to /keywords endpoint)
        keywords_updated = []
        if assessment_data.keywords:
            # Get the specific milestone key for this milestone only
            milestone_key = engine._get_milestone_key(milestone)
            
            # Initialize the keyword map for this milestone if it doesn't exist
            if milestone_key not in engine._scoring_keywords_cache:
                engine._scoring_keywords_cache[milestone_key] = {}
            
            # Process each category of keywords
            for category, keywords in assessment_data.keywords.items():
                # Skip invalid categories
                try:
                    score_enum = Score[category]
                except (KeyError, ValueError):
                    continue
                
                # Update the scoring keywords for this milestone
                for keyword in keywords:
                    engine._scoring_keywords_cache[milestone_key][keyword.lower()] = score_enum
                
                # Add this category to the list of updated categories
                keywords_updated.append(category)
        
        # Step 3: Analyze the response (similar to /score-response endpoint)
        # Try to use the hybrid scorer for more reliable scoring if available
        if HYBRID_SCORER_AVAILABLE:
            # Use our enhanced hybrid scoring approach with the provided keywords
            result = hybrid_score_response(
                assessment_data.milestone_behavior, 
                assessment_data.parent_response,
                assessment_data.keywords
            )
            
            score_value = result["score"]
            score_label = result["score_label"]
            confidence = result["confidence"]
            
            # Map score value to Score enum for the engine
            score_enum = None
            for score in Score:
                if score.value == score_value:
                    score_enum = score
                    break
            
            # Record the score in the engine if we have a valid score
            if score_enum is not None:
                engine.set_milestone_score(milestone, score_enum)
                print(f"Scored {milestone.behavior} ({milestone.domain}): {score_enum.name}")
        else:
            # Fall back to the original scoring logic
            score_enum = await engine.analyze_response(assessment_data.parent_response, milestone)
            score_value = score_enum.value
            score_label = score_enum.name
            confidence = 0.85  # Default confidence
            
            # Record the score
            engine.set_milestone_score(milestone, score_enum)
            print(f"Scored {milestone.behavior} ({milestone.domain}): {score_enum.name}")
        
        # Step 4: Return the comprehensive result
        return {
            "question_processed": True,
            "milestone_found": True,
            "milestone_details": milestone_details,
            "keywords_updated": keywords_updated,
            "score": score_value,
            "score_label": score_label,
            "confidence": confidence,
            "domain": milestone.domain
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing assessment: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ASD Assessment API Server")
    parser.add_argument("--port", type=int, default=8003, help="Port to run the server on")
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=True) 