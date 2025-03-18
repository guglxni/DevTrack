#!/usr/bin/env python3
"""
Advanced patch for LLM scorer with more direct, forceful modifications.
"""

import re
import sys
import os
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced_llm_patch")

# Define score enum for reference
class Score(Enum):
    NOT_RATED = -1
    CANNOT_DO = 0
    LOST_SKILL = 1
    EMERGING = 2
    WITH_SUPPORT = 3
    INDEPENDENT = 4

def directly_patch_file():
    """Apply patch directly to the LLM scorer code file"""
    project_root = os.getcwd()
    llm_scorer_path = os.path.join(project_root, "src", "core", "scoring", "llm_scorer.py")
    
    if not os.path.exists(llm_scorer_path):
        logger.error(f"Could not find LLM scorer file at {llm_scorer_path}")
        return False
    
    # Read the file
    with open(llm_scorer_path, 'r') as f:
        content = f.read()
    
    # Create a backup
    backup_path = llm_scorer_path + ".bak"
    with open(backup_path, 'w') as f:
        f.write(content)
    logger.info(f"Created backup at {backup_path}")
    
    # Define the new parse method
    new_parse_method = """
    def _parse_llm_response(self, response_text: str) -> Tuple[Score, float, str]:
        \"\"\"
        Enhanced parser for LLM response to extract score, confidence, and reasoning
        \"\"\"
        logger.info(f"Parsing LLM response: {response_text[:100]}...")
        
        # Try to extract score from text category labels first
        category_patterns = [
            r"(?:milestone status|score|assessment)(?:\s+is)?\s+([A-Z_]+)",  # "milestone status is INDEPENDENT"
            r"([A-Z_]+)\s*\(\d+\)",  # "INDEPENDENT(7)" or "INDEPENDENT (7)"
            r"([A-Z_]+)\s+category",  # "INDEPENDENT category"
            r"classified\s+as\s+([A-Z_]+)",  # "classified as INDEPENDENT"
            r"falls\s+into\s+(?:the\s+)?([A-Z_]+)",  # "falls into the INDEPENDENT"
            r"score\s+of\s+([A-Z_]+)",  # "score of INDEPENDENT"
            r"child\s+(?:is|can|has)\s+([A-Z_]+)",  # "child is INDEPENDENT"
            r"child(?:'s)?\s+milestone\s+(?:status)?\s+is\s+([A-Z_]+)",  # "child's milestone status is INDEPENDENT"
        ]
        
        for pattern in category_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                category_text = match.group(1).strip().upper()
                logger.info(f"Found category text: {category_text}")
                
                # Map the category text to Score enum
                score_map = {
                    "INDEPENDENT": Score.INDEPENDENT,
                    "EMERGING": Score.EMERGING,
                    "WITH_SUPPORT": Score.WITH_SUPPORT,
                    "SUPPORT": Score.WITH_SUPPORT, 
                    "LOST_SKILL": Score.LOST_SKILL,
                    "LOST": Score.LOST_SKILL,
                    "CANNOT_DO": Score.CANNOT_DO,
                    "CANNOT": Score.CANNOT_DO
                }
                
                for key, value in score_map.items():
                    if key in category_text:
                        score = value
                        logger.info(f"Mapped to score: {score}")
                        
                        # Extract reasoning
                        reasoning_text = response_text[match.end():].strip()
                        if not reasoning_text:
                            reasoning_text = "Score extracted from category label"
                        
                        # Use a high confidence for explicit category labels
                        confidence = 0.85
                        
                        return score, confidence, reasoning_text
        
        # If we couldn't extract from category text, try numbers with new patterns
        number_patterns = [
            r"[Ss]core.*?(\d+)",  # "Score: 7" or "score is 7"
            r"^\s*(\d+)\s*$",  # Just a number on a line
            r"(?:score|rating|assessment).*?(\d+)/\d+",  # "Score: 7/7"
            r"(?:score|rating|assessment).*?(\d+) out of \d+",  # "Score: 7 out of 7"
            r"(?<=status is )[1-7]",  # "status is 7"
            r"(?<=score: )[1-7]",  # "score: 7"
            r"\((\d+)\)",  # "(7)" - extract just the number
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    score_value = int(match.group(1))
                    logger.info(f"Found numeric score: {score_value}")
                    
                    # Interpret numeric scores
                    score_map = {
                        0: Score.CANNOT_DO,
                        1: Score.CANNOT_DO,
                        2: Score.LOST_SKILL,
                        3: Score.WITH_SUPPORT,
                        4: Score.EMERGING,
                        5: Score.EMERGING,
                        6: Score.INDEPENDENT,
                        7: Score.INDEPENDENT
                    }
                    
                    if score_value in score_map:
                        score = score_map[score_value]
                        logger.info(f"Mapped numeric {score_value} to score: {score}")
                        
                        # Extract reasoning
                        reasoning_text = response_text[match.end():].strip()
                        if not reasoning_text:
                            reasoning_text = f"Score {score_value} mapped to {score.name}"
                        
                        confidence = 0.75
                        return score, confidence, reasoning_text
                except ValueError:
                    pass  # Not a valid number, continue to next pattern
        
        # If all else fails, do simple text analysis
        text_lower = response_text.lower()
        
        # Look for clear indicators
        if any(word in text_lower for word in ["independently", "consistently", "without help", "independently and consistently"]):
            logger.info("Text analysis found INDEPENDENT indicators")
            return Score.INDEPENDENT, 0.7, "Found indicators of independent performance"
            
        if any(word in text_lower for word in ["emerging", "beginning", "starting to", "sometimes", "occasionally"]):
            logger.info("Text analysis found EMERGING indicators")
            return Score.EMERGING, 0.7, "Found indicators of emerging skill"
            
        if any(phrase in text_lower for phrase in ["with support", "with help", "with assistance", "when assisted"]):
            logger.info("Text analysis found WITH_SUPPORT indicators")
            return Score.WITH_SUPPORT, 0.7, "Found indicators of requiring support"
            
        if any(phrase in text_lower for phrase in ["lost skill", "used to", "regressed", "no longer"]):
            logger.info("Text analysis found LOST_SKILL indicators")
            return Score.LOST_SKILL, 0.7, "Found indicators of skill regression"
            
        if any(phrase in text_lower for phrase in ["cannot", "does not", "doesn't", "unable", "not able"]):
            logger.info("Text analysis found CANNOT_DO indicators")
            return Score.CANNOT_DO, 0.7, "Found indicators of inability to perform skill"
        
        # Last resort - "does it frequently" strongly suggests independent skill
        if "frequently" in text_lower or "regularly" in text_lower or "often" in text_lower:
            logger.info("Text analysis found frequency indicators suggesting INDEPENDENT")
            return Score.INDEPENDENT, 0.8, "Frequency indicators suggest independent skill mastery"
            
        # Failed to extract anything meaningful
        logger.warning(f"Failed to extract score from response: {response_text}")
        return Score.NOT_RATED, 0.5, "Could not determine score from model response"
    """
    
    # Find the existing method
    pattern = r"def _parse_llm_response\(self, response_text: str\).*?def "
    replacement = new_parse_method + "\n    def "
    
    # Apply replacement with regex
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back to file
    with open(llm_scorer_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Successfully replaced _parse_llm_response method!")
    return True

# Also modify ensemble_scorer to prioritize LLM scoring
def patch_ensemble_scorer():
    """Modify the ensemble scorer to prioritize LLM results"""
    project_root = os.getcwd()
    ensemble_path = os.path.join(project_root, "src", "core", "scoring", "ensemble_scorer.py")
    
    if not os.path.exists(ensemble_path):
        logger.error(f"Could not find ensemble scorer file at {ensemble_path}")
        return False
    
    # Back it up
    os.system(f"cp {ensemble_path} {ensemble_path}.bak")
    
    # Read content
    with open(ensemble_path, 'r') as f:
        content = f.read()
    
    # Replace the weight config with more emphasis on LLM
    weight_pattern = r"weights = {(.*?)}"
    new_weights = """weights = {
            "llm": self.config.get("score_weight_llm", 0.7),  # Higher weight for LLM
            "keyword": self.config.get("score_weight_keyword", 0.1),  # Lower weight for keywords
            "embedding": self.config.get("score_weight_embedding", 0.2),
            "transformer": self.config.get("score_weight_transformer", 0.2),
        }"""
    
    new_content = re.sub(weight_pattern, new_weights, content, flags=re.DOTALL)
    
    # Add code to prioritize LLM in combine_results
    combine_pattern = r"def _combine_results\(self, results: List\[ScoringResult\]\)"
    combine_replacement = """def _combine_results(self, results: List[ScoringResult]):
        \"\"\"
        Combine results from different scorers using a weighted approach.
        LLM scorer gets priority if it produces a valid result.
        \"\"\"
        # Check if LLM produced a valid result
        for result in results:
            if result.method == "llm_based" and result.score != Score.NOT_RATED and result.confidence > 0.7:
                # LLM has a strong opinion, use it
                logger.info("Using high-confidence LLM result directly")
                return result
        """
    
    new_content = new_content.replace(combine_pattern, combine_replacement)
    
    # Write back
    with open(ensemble_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Successfully modified ensemble scorer to prioritize LLM!")
    return True

if __name__ == "__main__":
    success = directly_patch_file()
    ensemble_success = patch_ensemble_scorer()
    
    if success and ensemble_success:
        logger.info("All patches applied successfully!")
        sys.exit(0)
    else:
        logger.error("Failed to apply all patches")
        sys.exit(1) 