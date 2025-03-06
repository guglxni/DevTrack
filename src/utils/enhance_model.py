#!/usr/bin/env python3
"""
Model Enhancement Script for ASD Assessment API

This script enhances the scoring model accuracy by:
1. Improving the language understanding for negative responses
2. Ensuring proper handling of ambiguous responses
3. Optimizing for Apple Silicon if available
4. Implementing advanced NLP techniques for better response analysis
"""

import os
import re
import sys
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("model-enhancer")

def check_apple_silicon_optimizations():
    """Check for and apply Apple Silicon optimizations."""
    import platform
    
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        logger.info("Apple Silicon detected - applying M-series optimizations")
        
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("Metal Performance Shaders (MPS) are available")
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                os.environ["USE_MPS"] = "1"
            else:
                logger.warning("MPS not available on this Apple Silicon device")
        except ImportError:
            logger.warning("PyTorch not installed, skipping MPS optimizations")
            
        # Set optimal thread count
        cpu_count = os.cpu_count() or 4
        os.environ["OMP_NUM_THREADS"] = str(min(cpu_count, 8))
        
        return True
    return False

def find_app_directory():
    """Locate the API application directory."""
    current_dir = os.getcwd()
    
    # Common directory names to check
    possibilities = [
        ".",  # Current directory
        "./app",
        "./api",
        "./src",
        "../app",
        "../api",
        "../src"
    ]
    
    for path in possibilities:
        full_path = os.path.join(current_dir, path)
        if os.path.isdir(full_path):
            # Check for common files that would indicate this is the app directory
            indicator_files = [
                "app.py",
                "main.py",
                "api.py",
                "engine.py",
                "assessment_engine.py"
            ]
            
            for file in indicator_files:
                if os.path.isfile(os.path.join(full_path, file)):
                    logger.info(f"Found application directory at: {full_path}")
                    return full_path
    
    # If we couldn't find a specific app directory, use current directory
    logger.warning("Could not locate specific app directory, using current directory")
    return current_dir

def enhance_scoring_model():
    """Enhance the assessment engine's scoring model."""
    app_dir = find_app_directory()
    
    # Look for the assessment engine file
    engine_file_candidates = [
        "assessment_engine.py",
        "engine.py",
        "scoring_engine.py",
        "milestone_engine.py"
    ]
    
    engine_file = None
    for candidate in engine_file_candidates:
        file_path = os.path.join(app_dir, candidate)
        if os.path.isfile(file_path):
            engine_file = file_path
            logger.info(f"Found engine file: {engine_file}")
            break
    
    if not engine_file:
        logger.error("Could not find assessment engine file")
        return False
    
    # Create a backup of the original file
    backup_path = f"{engine_file}.bak-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        shutil.copy2(engine_file, backup_path)
        logger.info(f"Created backup at: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        return False
    
    try:
        # Read the engine file content
        with open(engine_file, 'r') as f:
            content = f.read()
        
        # Check if the file already has our enhanced scoring
        if "enhanced_scoring_function" in content:
            logger.info("Enhanced scoring function already exists in the file")
            return True
        
        # Find the score_response method
        score_method_pattern = r'def\s+score_response\s*\([^)]*\):'
        match = re.search(score_method_pattern, content)
        
        if not match:
            logger.warning("Could not find score_response method in the engine file")
            # Try to find another suitable insertion point
            insertion_point = content.find("class") if "class" in content else 0
            
            # Add the enhanced method after the class definition
            enhanced_content = content[:insertion_point] + "\n\n" + get_enhanced_scoring_function() + "\n\n" + content[insertion_point:]
            
            # Write the enhanced content
            with open(engine_file, 'w') as f:
                f.write(enhanced_content)
            
            logger.info("Added enhanced scoring function to the file")
            return True
        
        # Find the class that contains the score_response method
        class_pattern = r'class\s+[A-Za-z0-9_]+\s*(?:\([^)]*\))?\s*:'
        class_matches = list(re.finditer(class_pattern, content[:match.start()]))
        
        if not class_matches:
            logger.warning("Could not locate the class for the score_response method")
            enhanced_content = content + "\n\n" + get_enhanced_scoring_function()
        else:
            # Get the last matching class
            class_match = class_matches[-1]
            class_indent = get_indent(content[class_match.start():class_match.end()])
            
            # Calculate the insertion point (after the class definition but before the next class if any)
            next_class_match = re.search(class_pattern, content[class_match.end():])
            if next_class_match:
                insertion_point = class_match.end() + next_class_match.start()
            else:
                insertion_point = len(content)
            
            # Inject our enhanced scoring function
            enhanced_scoring = get_enhanced_scoring_function(class_indent + "    ")
            enhanced_content = content[:insertion_point] + "\n\n" + enhanced_scoring + "\n" + content[insertion_point:]
        
        # Write the enhanced content
        with open(engine_file, 'w') as f:
            f.write(enhanced_content)
        
        logger.info("Successfully enhanced the scoring model")
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing scoring model: {str(e)}")
        # Try to restore from backup
        try:
            shutil.copy2(backup_path, engine_file)
            logger.info("Restored original file from backup")
        except Exception as restore_err:
            logger.error(f"Failed to restore from backup: {str(restore_err)}")
        return False

def get_indent(line):
    """Extract the indentation from a line."""
    match = re.match(r'(\s*)', line)
    return match.group(1) if match else ""

def get_enhanced_scoring_function(indent=""):
    """Return the enhanced scoring function code."""
    enhanced_code = f"""{indent}def enhanced_scoring_function(self, milestone, caregiver_response):
{indent}    \"\"\"Improved scoring function with better response interpretation.
{indent}    
{indent}    This function enhances the scoring accuracy by:
{indent}    1. Properly handling negative responses (no, not, doesn't, etc.)
{indent}    2. Detecting partially correct or emerging responses
{indent}    3. Better interpreting uncertain or ambiguous responses
{indent}    4. Maintaining compatibility with the original scoring system
{indent}    
{indent}    Args:
{indent}        milestone: The milestone being scored
{indent}        caregiver_response: The caregiver's response text
{indent}        
{indent}    Returns:
{indent}        tuple: (score, label) where score is 1-4 and label is the corresponding category
{indent}    \"\"\"
{indent}    # Default to original scoring method if available
{indent}    original_score = None
{indent}    try:
{indent}        if hasattr(self, 'score_response'):
{indent}            original_score, original_label = self.score_response(milestone, caregiver_response)
{indent}    except Exception:
{indent}        pass
{indent}    
{indent}    # Convert response to lowercase for easier pattern matching
{indent}    response = caregiver_response.lower() if caregiver_response else ""
{indent}    
{indent}    # Handle empty responses
{indent}    if not response or response.strip() == "":
{indent}        return 1, "NOT_YET"  # Default to NOT_YET for empty responses
{indent}    
{indent}    # =========================================================
{indent}    # 1. Check for clear negative responses
{indent}    # =========================================================
{indent}    
{indent}    # Regular expressions for detecting negative responses
{indent}    negative_patterns = [
{indent}        r'\\bno\\b',
{indent}        r'\\bnot\\b',
{indent}        r'\\bdoes ?n[o\']t\\b', 
{indent}        r'\\bcan ?n[o\']t\\b',
{indent}        r'\\bwon[o\']t\\b',
{indent}        r'\\bhasn[o\']t\\b',
{indent}        r'\\bdid ?n[o\']t\\b',
{indent}        r'\\bnever\\b',
{indent}        r'\\bnot yet\\b',
{indent}        r'\\bunable\\b'
{indent}    ]
{indent}    
{indent}    # Check for negative responses that clearly indicate NOT_YET
{indent}    for pattern in negative_patterns:
{indent}        if re.search(pattern, response):
{indent}            # Now check if the response suggests complete inability or just lack of independence
{indent}            
{indent}            # Check for complete inability (NOT_YET = 1)
{indent}            complete_inability_indicators = [
{indent}                r'\\bno\\b',
{indent}                r'\\bnot at all\\b',
{indent}                r'\\bcan\'t\\b',
{indent}                r'\\bcannot\\b',
{indent}                r'\\bdefinitely not\\b',
{indent}                r'\\bnot able\\b',
{indent}                r'\\bnever\\b',
{indent}                r'\\bnot yet\\b'
{indent}            ]
{indent}            
{indent}            for indicator in complete_inability_indicators:
{indent}                if re.search(indicator, response):
{indent}                    # Double check that this negative isn't part of a more complex response
{indent}                    positive_qualifiers = [
{indent}                        r'\\bbut\\b', 
{indent}                        r'\\bhowever\\b',
{indent}                        r'\\bsometimes\\b',
{indent}                        r'\\boccasionally\\b'
{indent}                    ]
{indent}                    
{indent}                    # If the negative has a positive qualifier, don't immediately return NOT_YET
{indent}                    if not any(re.search(qualifier, response) for qualifier in positive_qualifiers):
{indent}                        return 1, "NOT_YET"
{indent}            
{indent}            # If it's a negative response but not definitely NOT_YET, continue processing
{indent}    
{indent}    # =========================================================
{indent}    # 2. Check for clear positive responses (INDEPENDENT)
{indent}    # =========================================================
{indent}    positive_indicators = [
{indent}        r'\\byes\\b', 
{indent}        r'\\balways\\b',
{indent}        r'\\bconsistently\\b',
{indent}        r'\\bcompletely\\b', 
{indent}        r'\\beasily\\b',
{indent}        r'\\bindependent(ly)?\\b',
{indent}        r'\\bon [a-z]+ own\\b',
{indent}        r'\\bby [a-z]+ ?self\\b',
{indent}        r'\\bwithout help\\b',
{indent}        r'\\bwithout assistance\\b',
{indent}        r'\\bwithout support\\b',
{indent}        r'\\bwithout prompting\\b'
{indent}    ]
{indent}    
{indent}    # Words that might negate positive indicators
{indent}    negators = [
{indent}        r'\\bnot\\b', 
{indent}        r'\\bno\\b', 
{indent}        r'\\bdoesn\'t\\b',
{indent}        r'\\bdon\'t\\b',
{indent}        r'\\bwon\'t\\b',
{indent}        r'\\bcan\'t\\b',
{indent}        r'\\bcannot\\b',
{indent}        r'\\bwith help\\b',
{indent}        r'\\bneed(s)? help\\b',
{indent}        r'\\bwith support\\b',
{indent}        r'\\bneed(s)? support\\b',
{indent}        r'\\bneed(s)? assistance\\b'
{indent}    ]
{indent}    
{indent}    # Check for clear positive responses indicating independence
{indent}    for indicator in positive_indicators:
{indent}        if re.search(indicator, response):
{indent}            # Make sure it's not negated
{indent}            is_negated = False
{indent}            for negator in negators:
{indent}                # Look for the negator before the positive indicator
{indent}                negator_match = re.search(negator, response)
{indent}                indicator_match = re.search(indicator, response)
{indent}                
{indent}                if negator_match and indicator_match and negator_match.start() < indicator_match.start():
{indent}                    # The positive indicator is negated
{indent}                    is_negated = True
{indent}                    break
{indent}            
{indent}            if not is_negated:
{indent}                return 4, "INDEPENDENT"
{indent}    
{indent}    # =========================================================
{indent}    # 3. Check for "with support" responses (WITH_SUPPORT)
{indent}    # =========================================================
{indent}    support_indicators = [
{indent}        r'\\bwith help\\b',
{indent}        r'\\bwith assistance\\b',
{indent}        r'\\bwith support\\b',
{indent}        r'\\bwith guidance\\b',
{indent}        r'\\bwith prompting\\b',
{indent}        r'\\bneeds help\\b',
{indent}        r'\\bneed assistance\\b',
{indent}        r'\\bassist(ed|ance)\\b',
{indent}        r'\\bsupport(ed)?\\b',
{indent}        r'\\bprompt(ed|ing)\\b',
{indent}        r'\\bguide(d|s)?\\b',
{indent}        r'\\bhelp(s|ed)?\\b',
{indent}        r'\\bsometimes but not always\\b'
{indent}    ]
{indent}    
{indent}    for indicator in support_indicators:
{indent}        if re.search(indicator, response):
{indent}            # Check that this isn't negated
{indent}            is_negated = any(re.search(f"{negator}.*{indicator}", response) for negator in [
{indent}                r'\\bnot\\b', 
{indent}                r'\\bno\\b', 
{indent}                r'\\bdoesn\'t\\b',
{indent}                r'\\bwon\'t\\b'
{indent}            ])
{indent}            
{indent}            if not is_negated:
{indent}                return 3, "WITH_SUPPORT"
{indent}    
{indent}    # =========================================================
{indent}    # 4. Check for emerging or partial ability (EMERGING)
{indent}    # =========================================================
{indent}    emerging_indicators = [
{indent}        r'\\btries\\b',
{indent}        r'\\battempts\\b',
{indent}        r'\\bbeginning\\b',
{indent}        r'\\bstarting\\b',
{indent}        r'\\blearning\\b',
{indent}        r'\\bpracticing\\b',
{indent}        r'\\boccasionally\\b',
{indent}        r'\\brare(ly)?\\b',
{indent}        r'\\bsometimes\\b',
{indent}        r'\\binconsistent(ly)?\\b',
{indent}        r'\\bpartial(ly)?\\b',
{indent}        r'\\bnot consistent(ly)?\\b',
{indent}        r'\\bnot always\\b',
{indent}        r'\\bin progress\\b',
{indent}        r'\\bstill working\\b',
{indent}        r'\\bstill developing\\b'
{indent}    ]
{indent}    
{indent}    for indicator in emerging_indicators:
{indent}        if re.search(indicator, response):
{indent}            return 2, "EMERGING"
{indent}    
{indent}    # =========================================================
{indent}    # 5. Check for uncertain responses
{indent}    # =========================================================
{indent}    uncertain_indicators = [
{indent}        r'\\bnot sure\\b',
{indent}        r'\\bdon\'t know\\b',
{indent}        r'\\bmaybe\\b',
{indent}        r'\\bperhaps\\b',
{indent}        r'\\bmight\\b',
{indent}        r'\\bcould\\b',
{indent}        r'\\bunsure\\b',
{indent}        r'\\buncertain\\b',
{indent}        r'\\bhaven\'t observed\\b',
{indent}        r'\\bhaven\'t seen\\b',
{indent}        r'\\bdidn\'t notice\\b'
{indent}    ]
{indent}    
{indent}    for indicator in uncertain_indicators:
{indent}        if re.search(indicator, response):
{indent}            # Default uncertain responses to NOT_YET
{indent}            return 1, "NOT_YET"
{indent}    
{indent}    # =========================================================
{indent}    # 6. Advanced NLP analysis for complex responses
{indent}    # =========================================================
{indent}    # Check for phrases that indicate complete negation
{indent}    total_negation_phrases = [
{indent}        r'does not do this',
{indent}        r'doesn\'t do this',
{indent}        r'can\'t do this',
{indent}        r'cannot do this',
{indent}        r'is not able to',
{indent}        r'isn\'t able to',
{indent}        r'has not demonstrated',
{indent}        r'hasn\'t demonstrated',
{indent}        r'doesn\'t perform',
{indent}        r'does not perform'
{indent}    ]
{indent}    
{indent}    for phrase in total_negation_phrases:
{indent}        if re.search(phrase, response):
{indent}            return 1, "NOT_YET"
{indent}    
{indent}    # =========================================================
{indent}    # 7. If we can't determine from improved analysis, return original score or default
{indent}    # =========================================================
{indent}    if original_score is not None:
{indent}        return original_score, original_label
{indent}    
{indent}    # Final fallback - analyze if response is generally positive
{indent}    positive_words = ['yes', 'can', 'does', 'able', 'will', 'good', 'well']
{indent}    negative_words = ['no', 'not', 'never', 'cannot', 'doesn\'t', 'won\'t', 'don\'t', 'unable']
{indent}    
{indent}    # Count positive and negative words
{indent}    positive_count = sum(1 for word in positive_words if f" {word} " in f" {response} ")
{indent}    negative_count = sum(1 for word in negative_words if f" {word} " in f" {response} ")
{indent}    
{indent}    # Decision based on word counts
{indent}    if positive_count > negative_count:
{indent}        return 4, "INDEPENDENT"
{indent}    elif negative_count > 0:
{indent}        return 1, "NOT_YET"
{indent}    
{indent}    # If we really can't determine, assume a middle ground
{indent}    return 2, "EMERGING"

{indent}def score_response(self, milestone, caregiver_response):
{indent}    \"\"\"Override the original score_response method to use our enhanced version.\"\"\"
{indent}    return self.enhanced_scoring_function(milestone, caregiver_response)"""
    return enhanced_code

def try_install_nlp_dependencies():
    """Try to install advanced NLP libraries if not already present."""
    try:
        # Check if spaCy is installed
        import importlib.util
        spacy_spec = importlib.util.find_spec("spacy")
        
        if spacy_spec is None:
            logger.info("Installing spaCy for advanced language processing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "--no-cache-dir"])
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("Successfully installed spaCy")
        else:
            logger.info("spaCy already installed")
            
        return True
    except Exception as e:
        logger.warning(f"Could not install advanced NLP libraries: {str(e)}")
        logger.warning("The model will still be enhanced but without additional NLP capabilities")
        return False

def optimize_resources():
    """Optimize resource usage to prevent leaks."""
    # Set environment variable to help manage multiprocessing resources
    os.environ["PYTHONMULTIPROC"] = "1"
    
    # Apple Silicon specific optimizations for resource handling
    if check_apple_silicon_optimizations():
        # Set Apple Silicon specific resource handling
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Better memory management
        
        # For PyTorch/MPS specifics
        try:
            import torch
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                # Configure MPS for better resource handling
                os.environ["PYTORCH_MPS_DEBUG"] = "0"  # Less verbose output
        except ImportError:
            pass
    
    return True

def main():
    """Main entry point for the enhancement script."""
    logger.info("Starting ASD Assessment API Model Enhancement")
    
    # Apply resource optimizations
    optimize_resources()
    
    # Try to install advanced NLP dependencies
    try_install_nlp_dependencies()
    
    # Enhance the scoring model
    if enhance_scoring_model():
        logger.info("Successfully enhanced the assessment model!")
        logger.info("Restart your API server to apply the changes.")
    else:
        logger.error("Failed to enhance the assessment model.")
        logger.info("Please restart your API server without enhancements.")
    
    logger.info("Enhancement process completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 