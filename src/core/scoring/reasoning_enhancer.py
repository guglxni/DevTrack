#!/usr/bin/env python3
"""
Reasoning Enhancement Module

This module provides techniques to improve the explainability and reasoning 
capabilities of the scoring system using advanced LLM techniques.
"""

import os
import sys
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

from .base import BaseScorer, Score, ScoringResult
from ..retrieval.r2r_client import R2RClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reasoning_enhancer")

@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning chain."""
    step_number: int
    description: str
    conclusion: str
    confidence: float
    supporting_evidence: Optional[List[str]] = None

@dataclass
class ReasoningChain:
    """Represents a complete chain of reasoning steps."""
    steps: List[ReasoningStep]
    final_conclusion: str
    score: Score
    confidence: float
    reasoning_quality: float  # 0-1 scale for quality assessment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "steps": [vars(step) for step in self.steps],
            "final_conclusion": self.final_conclusion,
            "score": self.score.value if isinstance(self.score, Score) else self.score,
            "confidence": self.confidence,
            "reasoning_quality": self.reasoning_quality
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format for human readability."""
        markdown = "# Reasoning Process\n\n"
        
        for step in self.steps:
            markdown += f"## Step {step.step_number}: {step.description}\n\n"
            
            if step.supporting_evidence:
                markdown += "**Supporting Evidence:**\n\n"
                for evidence in step.supporting_evidence:
                    markdown += f"- {evidence}\n"
                markdown += "\n"
                
            markdown += f"**Conclusion:** {step.conclusion}\n\n"
            markdown += f"**Confidence:** {step.confidence:.2f}\n\n"
            markdown += "---\n\n"
            
        markdown += f"# Final Assessment\n\n"
        markdown += f"**Score:** {self.score}\n\n"
        markdown += f"**Conclusion:** {self.final_conclusion}\n\n"
        markdown += f"**Overall Confidence:** {self.confidence:.2f}\n\n"
        
        return markdown

class ReasoningEnhancer:
    """
    Enhances the reasoning capabilities of scoring models.
    
    This class provides techniques to:
    1. Extract reasoning steps from model outputs
    2. Improve reasoning with retrieval augmentation
    3. Track and improve reasoning patterns
    4. Provide explainable scores with detailed reasoning chains
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning enhancer.
        
        Args:
            config: Configuration dictionary for customization
        """
        self.config = config or self._default_config()
        
        # Initialize R2R client if configured
        if self.config.get("use_r2r", True):
            try:
                self.r2r_client = R2RClient()
                self.r2r_available = True
            except Exception as e:
                logger.warning(f"R2R client initialization failed: {str(e)}")
                self.r2r_available = False
        else:
            self.r2r_available = False
            
        # Set up storage for reasoning patterns
        self.reasoning_patterns_dir = Path(self.config.get("reasoning_patterns_dir", "data/reasoning_patterns"))
        self.reasoning_patterns_dir.mkdir(parents=True, exist_ok=True)
        
        # Load reasoning patterns if available
        self.reasoning_patterns = self._load_reasoning_patterns()
        
        # Feedback collection
        self.feedback_collection = self.config.get("feedback_collection", "reasoning_feedback")
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "use_r2r": True,
            "reasoning_patterns_dir": "data/reasoning_patterns",
            "feedback_collection": "reasoning_feedback",
            "max_reasoning_steps": 5,
            "min_step_confidence": 0.6,
            "extract_reasoning_prompt": """
Extract the reasoning steps from the following scoring assessment. For each step, identify:
1. The specific reasoning action being taken
2. The conclusion reached in that step
3. Any evidence or observations mentioned

Text to analyze:
{text}

Format your response as a JSON object with a 'steps' array containing objects with 'step_number', 'description', 'conclusion', and 'supporting_evidence' fields.
""",
            "enhance_reasoning_prompt": """
I'll help you improve your reasoning process for scoring this developmental milestone.

Milestone: {milestone}
Domain: {domain}
Age Range: {age_range}
Parent's Response: {response}

Your current reasoning:
{current_reasoning}

Let me enhance this reasoning by considering:
1. Additional developmental context for this age range
2. Potential alternative interpretations of the parent's response
3. More detailed behavioral observations
4. Specific developmental markers relevant to this milestone

Provide an improved, step-by-step reasoning chain leading to a final score.
"""
        }
    
    def _load_reasoning_patterns(self) -> Dict[str, Any]:
        """Load stored reasoning patterns from disk."""
        patterns_file = self.reasoning_patterns_dir / "patterns.json"
        
        if not patterns_file.exists():
            # Initialize with empty patterns
            default_patterns = {
                "domains": {},
                "age_ranges": {},
                "common_steps": [],
                "version": "1.0"
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(default_patterns, f, indent=2)
                
            return default_patterns
        
        try:
            with open(patterns_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading reasoning patterns: {str(e)}")
            return {
                "domains": {},
                "age_ranges": {},
                "common_steps": [],
                "version": "1.0"
            }
    
    def _save_reasoning_patterns(self) -> None:
        """Save updated reasoning patterns to disk."""
        patterns_file = self.reasoning_patterns_dir / "patterns.json"
        
        try:
            with open(patterns_file, 'w') as f:
                json.dump(self.reasoning_patterns, f, indent=2)
            logger.info("Reasoning patterns saved successfully")
        except Exception as e:
            logger.error(f"Error saving reasoning patterns: {str(e)}")
    
    def extract_reasoning_steps(self, text: str) -> List[ReasoningStep]:
        """
        Extract reasoning steps from model output text.
        
        Args:
            text: Text output from a scoring model
            
        Returns:
            List of structured reasoning steps
        """
        steps = []
        
        # Try to use R2R for extraction if available
        if self.r2r_available:
            try:
                # Prepare the prompt for extracting reasoning steps
                prompt = self.config["extract_reasoning_prompt"].format(text=text)
                
                # Use R2R to generate a structured analysis
                result = self.r2r_client.generate(
                    query=prompt,
                    collection_key=None,  # No need for retrieval here
                    max_tokens=1024,
                    temperature=0.1
                )
                
                # Try to parse JSON from the response
                json_match = re.search(r'```json\s*(.*?)\s*```', result.get("generated_text", ""), re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result.get("generated_text", "")
                
                # Clean up the string to help with JSON parsing
                json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
                
                # Extract JSON object if present
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                
                try:
                    data = json.loads(json_str)
                    
                    # Process the steps
                    if "steps" in data and isinstance(data["steps"], list):
                        for i, step_data in enumerate(data["steps"]):
                            step = ReasoningStep(
                                step_number=step_data.get("step_number", i+1),
                                description=step_data.get("description", ""),
                                conclusion=step_data.get("conclusion", ""),
                                confidence=step_data.get("confidence", 0.8),
                                supporting_evidence=step_data.get("supporting_evidence", [])
                            )
                            steps.append(step)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from extraction result: {json_str[:100]}...")
            
            except Exception as e:
                logger.warning(f"R2R reasoning extraction failed: {str(e)}")
        
        # If R2R failed or is not available, use regex-based extraction
        if not steps:
            logger.info("Using fallback regex-based reasoning extraction")
            
            # Look for numbered steps or paragraphs
            step_patterns = [
                r'Step\s+(\d+)[:\.\)]\s*(.*?)(?=Step\s+\d+[:\.\)]|$)',
                r'(\d+)[:\.\)]\s*(.*?)(?=\d+[:\.\)]|$)',
                r'([A-Z][^.!?]*(?:[.!?][^.!?]*)*[.!?])',  # Extract sentences
            ]
            
            for pattern in step_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                
                if matches and pattern == step_patterns[0]:  # Numbered steps with "Step" prefix
                    for i, (num, content) in enumerate(matches):
                        # Try to separate description from conclusion
                        parts = content.split(".")
                        description = parts[0].strip() if parts else "Analysis"
                        conclusion = ".".join(parts[1:]).strip() if len(parts) > 1 else content.strip()
                        
                        step = ReasoningStep(
                            step_number=int(num),
                            description=description,
                            conclusion=conclusion,
                            confidence=0.7,
                            supporting_evidence=[]
                        )
                        steps.append(step)
                    break
                    
                elif matches and pattern == step_patterns[1]:  # Simple numbered points
                    for i, (num, content) in enumerate(matches):
                        # Simple splitting of content
                        parts = content.split(".")
                        description = parts[0].strip() if parts else "Analysis"
                        conclusion = ".".join(parts[1:]).strip() if len(parts) > 1 else content.strip()
                        
                        step = ReasoningStep(
                            step_number=int(num),
                            description=description,
                            conclusion=conclusion,
                            confidence=0.6,
                            supporting_evidence=[]
                        )
                        steps.append(step)
                    break
                    
                elif matches and pattern == step_patterns[2]:  # Extract sentences as steps
                    # If we've resorted to sentence extraction, grab up to 5 sentences
                    for i, sentence in enumerate(matches[:5]):
                        step = ReasoningStep(
                            step_number=i+1,
                            description=f"Analysis point {i+1}",
                            conclusion=sentence.strip(),
                            confidence=0.5,
                            supporting_evidence=[]
                        )
                        steps.append(step)
                    break
        
        # If we still have no steps, create a single step with the entire text
        if not steps:
            steps = [
                ReasoningStep(
                    step_number=1,
                    description="Overall assessment",
                    conclusion=text.strip(),
                    confidence=0.4,
                    supporting_evidence=[]
                )
            ]
            
        return steps
    
    def enhance_reasoning(self, 
                         original_text: str, 
                         milestone_context: Dict[str, Any]) -> str:
        """
        Enhance the reasoning in the original text using R2R and reasoning patterns.
        
        Args:
            original_text: Original reasoning text
            milestone_context: Context about the milestone being assessed
            
        Returns:
            Enhanced reasoning text
        """
        if not self.r2r_available:
            logger.warning("R2R not available for reasoning enhancement")
            return original_text
            
        try:
            # Prepare the prompt for enhancing reasoning
            prompt = self.config["enhance_reasoning_prompt"].format(
                milestone=milestone_context.get("milestone", ""),
                domain=milestone_context.get("domain", ""),
                age_range=milestone_context.get("age_range", ""),
                response=milestone_context.get("response", ""),
                current_reasoning=original_text
            )
            
            # Retrieve relevant information for this milestone domain
            search_results = self.r2r_client.search(
                query=f"{milestone_context.get('domain', '')} {milestone_context.get('milestone', '')} development",
                collection_key="developmental_research",
                limit=3
            )
            
            # Extract context from search results
            contexts = []
            if "results" in search_results and search_results["results"]:
                for result in search_results["results"]:
                    contexts.append(result.get("text", ""))
            
            # Generate enhanced reasoning with retrieval augmentation
            result = self.r2r_client.generate(
                query=prompt,
                collection_key="developmental_research",
                max_tokens=1024,
                temperature=0.3,
                contexts=contexts
            )
            
            enhanced_text = result.get("generated_text", original_text)
            
            # If the enhancement failed or is too short, return the original
            if not enhanced_text or len(enhanced_text) < len(original_text) / 2:
                return original_text
                
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Reasoning enhancement failed: {str(e)}")
            return original_text
    
    def analyze_score_with_reasoning(self, 
                                   score_result: ScoringResult, 
                                   milestone_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a scoring result and add detailed reasoning.
        
        Args:
            score_result: The original scoring result
            milestone_context: Context about the milestone
            
        Returns:
            Enhanced result with detailed reasoning chains
        """
        enhanced_result = score_result.to_dict()
        
        # Extract reasoning text from the result
        reasoning_text = score_result.reasoning or ""
        
        # If there's no reasoning provided, try to extract it from other fields
        if not reasoning_text and score_result.explanation:
            reasoning_text = score_result.explanation
            
        # Extract structured reasoning steps
        reasoning_steps = self.extract_reasoning_steps(reasoning_text)
        
        # Enhance reasoning if possible and if we have steps
        if reasoning_steps and len(reasoning_text) > 0:
            enhanced_reasoning = self.enhance_reasoning(reasoning_text, milestone_context)
            
            # Extract steps from enhanced reasoning
            if enhanced_reasoning != reasoning_text:
                reasoning_steps = self.extract_reasoning_steps(enhanced_reasoning)
                
                # Update the result with enhanced reasoning
                enhanced_result["reasoning"] = enhanced_reasoning
        
        # Create a reasoning chain
        chain = ReasoningChain(
            steps=reasoning_steps,
            final_conclusion=reasoning_steps[-1].conclusion if reasoning_steps else "",
            score=score_result.score,
            confidence=score_result.confidence,
            reasoning_quality=self._calculate_reasoning_quality(reasoning_steps)
        )
        
        # Add reasoning chain to the result
        enhanced_result["reasoning_chain"] = chain.to_dict()
        enhanced_result["reasoning_markdown"] = chain.to_markdown()
        
        # Update reasoning patterns
        self._update_reasoning_patterns(chain, milestone_context)
        
        return enhanced_result
    
    def _calculate_reasoning_quality(self, steps: List[ReasoningStep]) -> float:
        """Calculate a quality score for a reasoning chain."""
        if not steps:
            return 0.0
            
        # Factors in quality assessment
        num_steps = len(steps)
        avg_confidence = sum(step.confidence for step in steps) / num_steps if num_steps > 0 else 0
        has_evidence = any(step.supporting_evidence for step in steps)
        
        # More steps (up to a point) and evidence suggest better reasoning
        steps_score = min(num_steps / self.config.get("max_reasoning_steps", 5), 1.0)
        evidence_score = 0.3 if has_evidence else 0.0
        
        # Calculate overall quality score
        quality = (steps_score * 0.4) + (avg_confidence * 0.3) + evidence_score
        
        # Cap at 1.0
        return min(quality, 1.0)
    
    def _update_reasoning_patterns(self, 
                                chain: ReasoningChain, 
                                milestone_context: Dict[str, Any]) -> None:
        """Update stored reasoning patterns based on new examples."""
        # Only update if we have a good quality chain
        if chain.reasoning_quality < 0.6:
            return
            
        # Get domain and age range
        domain = milestone_context.get("domain", "").lower()
        age_range = milestone_context.get("age_range", "").lower()
        
        # Update domain-specific patterns
        if domain:
            if domain not in self.reasoning_patterns["domains"]:
                self.reasoning_patterns["domains"][domain] = {
                    "common_steps": [],
                    "examples": []
                }
                
            domain_data = self.reasoning_patterns["domains"][domain]
            
            # Extract step descriptions for pattern matching
            step_descriptions = [step.description for step in chain.steps]
            
            # Update common steps for this domain
            for desc in step_descriptions:
                if desc not in domain_data["common_steps"]:
                    domain_data["common_steps"].append(desc)
                    
            # Add this as an example (limit to 10 examples per domain)
            domain_data["examples"] = domain_data["examples"][-9:] + [{
                "milestone": milestone_context.get("milestone", ""),
                "steps": [vars(step) for step in chain.steps],
                "score": chain.score.value if isinstance(chain.score, Score) else chain.score
            }]
        
        # Update age range patterns similarly
        if age_range:
            if age_range not in self.reasoning_patterns["age_ranges"]:
                self.reasoning_patterns["age_ranges"][age_range] = {
                    "common_steps": [],
                    "examples": []
                }
                
            age_data = self.reasoning_patterns["age_ranges"][age_range]
            
            # Extract step descriptions
            step_descriptions = [step.description for step in chain.steps]
            
            # Update common steps for this age range
            for desc in step_descriptions:
                if desc not in age_data["common_steps"]:
                    age_data["common_steps"].append(desc)
                    
            # Add as example (limit to 10)
            age_data["examples"] = age_data["examples"][-9:] + [{
                "milestone": milestone_context.get("milestone", ""),
                "steps": [vars(step) for step in chain.steps],
                "score": chain.score.value if isinstance(chain.score, Score) else chain.score
            }]
            
        # Save updated patterns
        self._save_reasoning_patterns()
    
    def get_similar_reasoning_examples(self, 
                                     milestone_context: Dict[str, Any], 
                                     limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve similar reasoning examples to help with assessment.
        
        Args:
            milestone_context: Context about the milestone being assessed
            limit: Maximum number of examples to return
            
        Returns:
            List of similar reasoning examples
        """
        examples = []
        
        # Get examples from domain patterns
        domain = milestone_context.get("domain", "").lower()
        if domain in self.reasoning_patterns["domains"]:
            domain_examples = self.reasoning_patterns["domains"][domain].get("examples", [])
            examples.extend(domain_examples)
            
        # Get examples from age range patterns
        age_range = milestone_context.get("age_range", "").lower()
        if age_range in self.reasoning_patterns["age_ranges"]:
            age_examples = self.reasoning_patterns["age_ranges"][age_range].get("examples", [])
            examples.extend(age_examples)
            
        # If we have R2R, try to find more similar examples
        if self.r2r_available and self.r2r_client:
            try:
                # Search for similar examples in the R2R system
                query = f"{domain} {milestone_context.get('milestone', '')} assessment"
                search_results = self.r2r_client.search(
                    query=query,
                    collection_key="scoring_examples",
                    limit=limit
                )
                
                # Add these to our examples
                if "results" in search_results and search_results["results"]:
                    for result in search_results["results"]:
                        # Try to extract reasoning steps from the result
                        reasoning_text = result.get("text", "")
                        if reasoning_text:
                            examples.append({
                                "milestone": result.get("metadata", {}).get("milestone", "Similar milestone"),
                                "text": reasoning_text,
                                "source": "r2r"
                            })
            except Exception as e:
                logger.warning(f"Error retrieving similar examples from R2R: {str(e)}")
        
        # Limit and return examples
        return examples[:limit]
    
    def collect_reasoning_feedback(self, 
                                 reasoning_chain: ReasoningChain, 
                                 feedback: Dict[str, Any]) -> bool:
        """
        Collect feedback on reasoning to improve future assessments.
        
        Args:
            reasoning_chain: The reasoning chain being rated
            feedback: Feedback data including ratings and comments
            
        Returns:
            Success status
        """
        if not self.r2r_available:
            logger.warning("Cannot collect feedback without R2R")
            return False
            
        try:
            # Prepare the feedback document
            document = {
                "title": f"Reasoning Feedback - {feedback.get('milestone', 'Unknown')}",
                "content": json.dumps({
                    "reasoning_chain": reasoning_chain.to_dict(),
                    "feedback": feedback,
                    "timestamp": time.time()
                }),
                "metadata": {
                    "milestone": feedback.get("milestone", ""),
                    "domain": feedback.get("domain", ""),
                    "age_range": feedback.get("age_range", ""),
                    "rating": feedback.get("rating", 0),
                    "feedback_type": "reasoning"
                }
            }
            
            # Store in R2R
            result = self.r2r_client.ingest(
                document=document,
                collection_key=self.feedback_collection
            )
            
            logger.info(f"Feedback collected with ID: {result.get('document_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting reasoning feedback: {str(e)}")
            return False
    
    def get_reasoning_suggestions(self, 
                               milestone_context: Dict[str, Any]) -> List[str]:
        """
        Get reasoning step suggestions for a specific milestone context.
        
        Args:
            milestone_context: Context about the milestone
            
        Returns:
            List of suggested reasoning steps
        """
        suggestions = []
        
        # Get domain-specific step suggestions
        domain = milestone_context.get("domain", "").lower()
        if domain in self.reasoning_patterns["domains"]:
            domain_steps = self.reasoning_patterns["domains"][domain].get("common_steps", [])
            suggestions.extend(domain_steps[:3])  # Add top 3 steps for this domain
        
        # Get age-specific step suggestions
        age_range = milestone_context.get("age_range", "").lower()
        if age_range in self.reasoning_patterns["age_ranges"]:
            age_steps = self.reasoning_patterns["age_ranges"][age_range].get("common_steps", [])
            for step in age_steps[:2]:  # Add top 2 steps for this age range
                if step not in suggestions:
                    suggestions.append(step)
        
        # Add generic steps if we don't have enough
        if len(suggestions) < 3:
            generic_steps = [
                "Analyze the specific behaviors described in the parent's response",
                "Consider the developmental expectations for this age range",
                "Evaluate the frequency and consistency of the behavior",
                "Assess the level of support or assistance needed",
                "Compare to typical developmental progression"
            ]
            
            for step in generic_steps:
                if step not in suggestions and len(suggestions) < 5:
                    suggestions.append(step)
        
        return suggestions

# Function for easy use in scoring modules
def enhance_scoring_with_reasoning(
    score_result: ScoringResult,
    milestone_context: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhance a scoring result with detailed reasoning analysis.
    
    Args:
        score_result: Original scoring result
        milestone_context: Context about the milestone
        config: Optional configuration
        
    Returns:
        Enhanced scoring result with reasoning chains
    """
    enhancer = ReasoningEnhancer(config)
    return enhancer.analyze_score_with_reasoning(score_result, milestone_context) 
 