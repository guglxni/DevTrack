#!/usr/bin/env python3

"""
Advanced NLP Module for ASD Developmental Milestone Assessment

This module provides specialized NLP techniques for analyzing caregiver responses
to developmental milestone questions. It enhances the accuracy of scoring by implementing:

1. Context-aware negation detection
2. Phrase-level semantic analysis
3. Advanced pattern matching for developmental language
4. Optimized processing for Apple Silicon hardware
"""

import os
import re
import sys
import logging
import importlib.util
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("advanced-nlp")

# Check for Apple Silicon and setup optimizations
def setup_optimizations():
    """Configure system for optimal performance on the current hardware."""
    import platform
    
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        logger.info("Apple Silicon detected - applying M-series optimizations")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTHONMULTIPROC"] = "1"
        
        # Set optimal thread count for M-series chips
        cpu_count = os.cpu_count() or 4
        os.environ["OMP_NUM_THREADS"] = str(min(cpu_count, 8))
        
        # Try to use Metal Performance Shaders if available
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("Metal Performance Shaders (MPS) are available")
                os.environ["USE_MPS"] = "1"
        except ImportError:
            logger.info("PyTorch not available, skipping MPS optimizations")
    else:
        logger.info("Non-Apple Silicon platform detected")
    
    return is_apple_silicon

class AdvancedResponseAnalyzer:
    """Analyzes caregiver responses to developmental milestone questions with context awareness."""
    
    def __init__(self):
        """Initialize the response analyzer with vocabulary sets and pattern matching."""
        # Check for Apple Silicon and apply optimizations
        self.apply_m_series_optimizations()
        
        # Initialize the specialized vocabulary for each scoring category
        self.vocab = {
            "CANNOT_DO": [
                "no", "not", "never", "doesn't", "does not", "cannot", "can't", "unable", 
                "hasn't", "has not", "not able", "not at all", "not yet started", "not capable"
            ],
            "LOST_SKILL": [
                "used to", "did before", "previously", "no longer", "stopped", "regressed", 
                "lost", "forgotten", "disappeared", "faded", "decreased", "not anymore", 
                "used to be able", "had this skill"
            ],
            "EMERGING": [
                "sometimes", "occasionally", "starting to", "beginning to", "tries to", 
                "attempts to", "inconsistent", "not consistent", "inconsistently", "varies", 
                "variable", "some days", "on occasion", "now and then", "hit or miss", "maybe",
                "kind of", "sort of", "partially"
            ],
            "WITH_SUPPORT": [
                "with help", "needs help", "when prompted", "with assistance", "with support", 
                "when reminded", "if I", "needs me to", "when I", "guided", "coaching", 
                "depends on", "only in certain", "specific situation", "particular setting",
                "needs direction", "only when", "specific context"
            ],
            "INDEPENDENT": [
                "yes", "always", "consistently", "independently", "by himself", "by herself", 
                "on own", "without help", "without support", "without assistance", "without prompting", 
                "all the time", "every time", "in all situations", "mastered", "completely"
            ]
        }
        
        # Compile regex patterns for more accurate detection
        self.patterns = {
            "CANNOT_DO": [
                r"\b(no|not|never|doesn'?t|cannot|can'?t)\b(?!.*\b(sometimes|occasionally|with help)\b)",
                r"\b(unable|hasn'?t|not able|not at all)\b",
                r"^no$", 
                r"^not yet$",
                r"hasn'?t (started|begun|tried|attempted)",
                r"not (demonstrated|shown|exhibited|displayed)"
            ],
            "LOST_SKILL": [
                r"(used to|did before|previously|no longer)\b",
                r"\b(stopped|regressed|lost|forgotten)\b",
                r"(disappeared|faded|decreased)\b",
                r"not (any ?more|any ?longer)",
                r"had this skill (but|and) (lost|forgot|stopped)",
                r"(was|used to be) able (to|but) (now|not)"
            ],
            "EMERGING": [
                r"\b(sometimes|occasionally|starting to|beginning to)\b",
                r"\b(tries|attempts|trying|attempting) to\b",
                r"\b(inconsistent|not consistent|varies|variable)\b",
                r"(some days|on occasion|now and then|hit or miss)",
                r"\b(maybe|kind of|sort of|partially)\b",
                r"not (always|consistently|regularly)"
            ],
            "WITH_SUPPORT": [
                r"\b(with (help|support|assistance|prompting|guidance))\b",
                r"\b(needs (help|support|assistance|prompting|guidance))\b",
                r"(when prompted|when reminded|when assisted|when supported)",
                r"(if I|needs me to|when I|with my|with our)",
                r"(only in (certain|specific|particular))",
                r"(depends on|needs direction|only when)"
            ],
            "INDEPENDENT": [
                r"^yes$",
                r"\b(always|consistently|independently)\b(?!.*\b(not|except|but)\b)",
                r"\b(by (him|her)self|on (his|her|their) own)\b",
                r"(without (help|support|assistance|prompting))",
                r"(all the time|every time|in all situations|mastered|completely)",
                r"^(does|can|is able)"
            ]
        }
        
        # Initialize the spaCy model for linguistic analysis if available
        self.nlp = self.load_spacy_model()
        
    def apply_m_series_optimizations(self):
        """Apply optimizations for Apple Silicon M-series chips."""
        try:
            import platform
            is_mac = platform.system() == 'Darwin'
            processor = platform.processor()
            
            if is_mac and ('M1' in processor or 'M2' in processor or 'M3' in processor or 'M4' in processor):
                logger.info(f"Detected Apple Silicon: {processor}")
                
                # Set environment variables for optimal performance
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                
                # Determine optimal thread settings based on chip
                if 'M4' in processor:
                    os.environ["MKL_NUM_THREADS"] = "12"
                    os.environ["NUMEXPR_NUM_THREADS"] = "12"
                    os.environ["OMP_NUM_THREADS"] = "12"
                elif 'M3' in processor:
                    os.environ["MKL_NUM_THREADS"] = "10"
                    os.environ["NUMEXPR_NUM_THREADS"] = "10"
                    os.environ["OMP_NUM_THREADS"] = "10"
                elif 'M2' in processor:
                    os.environ["MKL_NUM_THREADS"] = "8"
                    os.environ["NUMEXPR_NUM_THREADS"] = "8"
                    os.environ["OMP_NUM_THREADS"] = "8"
                else:  # M1
                    os.environ["MKL_NUM_THREADS"] = "8"
                    os.environ["NUMEXPR_NUM_THREADS"] = "8"
                    os.environ["OMP_NUM_THREADS"] = "8"
                
                logger.info("Applied Apple Silicon M-series optimizations")
                return True
            return False
        except Exception as e:
            logger.warning(f"Could not apply M-series optimizations: {str(e)}")
            return False
    
    def load_spacy_model(self):
        """Load the spaCy NLP model for advanced linguistic processing."""
        try:
            import spacy
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                logger.info("Downloading spaCy model...")
                spacy.cli.download("en_core_web_sm")
                return spacy.load("en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not installed. Using basic text analysis instead.")
            return None
    
    def detect_negation(self, text):
        """Detect negation in the response with linguistic context."""
        if not text:
            return 0.0
            
        # Check for direct negations
        negation_words = ["no", "not", "never", "doesn't", "does not", "cannot", "can't"]
        for word in negation_words:
            if re.search(rf"\b{word}\b", text.lower()):
                # Check if negation is reversed by double negative
                if len(re.findall(r"\b(no|not|never|doesn'?t|cannot|can'?t)\b", text.lower())) % 2 == 0:
                    return 0.0
                return 1.0
                
        # Check negation patterns
        for pattern in self.patterns["CANNOT_DO"]:
            if re.search(pattern, text.lower()):
                return 1.0
                
        return 0.0
    
    def detect_lost_skill(self, text):
        """Detect indications that a skill was acquired but then lost."""
        if not text:
            return 0.0
            
        for word in self.vocab["LOST_SKILL"]:
            if word in text.lower():
                return 1.0
                
        for pattern in self.patterns["LOST_SKILL"]:
            if re.search(pattern, text.lower()):
                return 1.0
                
        return 0.0
        
    def detect_uncertainty(self, text):
        """Detect uncertainty or inconsistency in the response."""
        if not text:
            return 0.0
            
        # Check for uncertainty indicators
        uncertainty_score = 0.0
        uncertainty_triggers = {
            "sometimes": 0.7,
            "occasionally": 0.6,
            "maybe": 0.5,
            "tries to": 0.8,
            "attempting": 0.7,
            "inconsistent": 0.9,
            "varies": 0.6,
            "beginning to": 0.8,
            "starting to": 0.8
        }
        
        for trigger, score in uncertainty_triggers.items():
            if trigger in text.lower():
                uncertainty_score = max(uncertainty_score, score)
                
        # Check pattern match
        for pattern in self.patterns["EMERGING"]:
            if re.search(pattern, text.lower()):
                uncertainty_score = max(uncertainty_score, 0.8)
                
        return uncertainty_score
    
    def detect_support(self, text):
        """Detect if the child requires support to perform the skill."""
        if not text:
            return 0.0
            
        # Check for support indicators
        support_score = 0.0
        support_triggers = {
            "with help": 0.9,
            "needs help": 0.9,
            "with support": 0.9,
            "with assistance": 0.9,
            "when prompted": 0.8,
            "when reminded": 0.8,
            "if I": 0.7,
            "when I": 0.7,
            "needs me to": 0.8,
            "only in specific": 0.8,
            "specific situations": 0.8
        }
        
        for trigger, score in support_triggers.items():
            if trigger in text.lower():
                support_score = max(support_score, score)
                
        # Check pattern match
        for pattern in self.patterns["WITH_SUPPORT"]:
            if re.search(pattern, text.lower()):
                support_score = max(support_score, 0.8)
                
        return support_score
    
    def detect_independence(self, text):
        """Detect indications that child performs skill independently."""
        if not text:
            return 0.0
            
        # If text is just "yes", look more carefully at context
        if text.lower().strip() == "yes":
            return 0.9
            
        independence_score = 0.0
        independence_triggers = {
            "always": 0.9,
            "consistently": 0.9,
            "independently": 1.0,
            "by himself": 0.9,
            "by herself": 0.9,
            "on own": 0.9,
            "without help": 1.0,
            "without support": 1.0,
            "all the time": 0.9,
            "every time": 0.9,
            "mastered": 1.0
        }
        
        for trigger, score in independence_triggers.items():
            if trigger in text.lower():
                independence_score = max(independence_score, score)
                
        # Check for negation that would invalidate independence
        if self.detect_negation(text) > 0.5:
            independence_score = 0.0
                
        # Check pattern match
        for pattern in self.patterns["INDEPENDENT"]:
            if re.search(pattern, text.lower()) and not re.search(r"\b(not|no|never)\b", text.lower()):
                independence_score = max(independence_score, 0.8)
                
        return independence_score
    
    def detect_progress_indicators(self, text):
        """Detect words or phrases that indicate progress or development."""
        if not text:
            return 0.0
            
        progress_score = 0.0
        progress_indicators = {
            "beginning": 0.6,
            "starting": 0.6,
            "learning": 0.5,
            "improving": 0.7,
            "getting better": 0.7,
            "working on": 0.5,
            "developing": 0.6,
            "progressing": 0.7,
            "emerging": 0.8
        }
        
        for indicator, score in progress_indicators.items():
            if indicator in text.lower():
                progress_score = max(progress_score, score)
                
        return progress_score
        
    def analyze_response(self, milestone, response, domain=None):
        """
        Analyze a caregiver's response to a developmental milestone question.
        
        Args:
            milestone (str): The developmental milestone being assessed
            response (str): The caregiver's response text
            domain (str, optional): The developmental domain (e.g., 'RL', 'GM', 'FM')
            
        Returns:
            dict: Analysis results including score, confidence, and explanation
        """
        if not response:
            return {
                "score_label": "CANNOT_DO",
                "score": 0,
                "confidence": 1.0,
                "explanation": "No response provided."
            }
            
        # Clean the response text
        response_clean = response.strip().lower()
        
        # Apply special handling for complex responses with explanations
        complex_response_result = self.handle_complex_response(response_clean)
        if complex_response_result:
            return complex_response_result
            
        # Apply linguistic analysis if spaCy is available
        if self.nlp:
            doc = self.nlp(response_clean)
            # TODO: Add more sophisticated linguistic analysis here
        
        # Detect indicators for each scoring category
        negation_score = self.detect_negation(response_clean)
        lost_skill_score = self.detect_lost_skill(response_clean)
        uncertainty_score = self.detect_uncertainty(response_clean)
        support_score = self.detect_support(response_clean)
        independence_score = self.detect_independence(response_clean)
        progress_score = self.detect_progress_indicators(response_clean)
        
        # Pattern matching scores for each category
        pattern_scores = {
            "CANNOT_DO": 0,
            "LOST_SKILL": 0,
            "EMERGING": 0,
            "WITH_SUPPORT": 0,
            "INDEPENDENT": 0
        }
        
        # Check matches against compiled patterns
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_clean):
                    pattern_scores[category] = 1.0
                    break
        
        # Linguistic scores based on feature detection
        linguistic_scores = {
            "negation": negation_score,
            "lost_skill": lost_skill_score,
            "uncertainty": uncertainty_score,
            "support": support_score,
            "independence": independence_score,
            "progress": progress_score
        }
        
        # Determine the most likely score based on evidence
        if negation_score > 0.7:
            # Strong negation indicates skill not acquired
            score_label = "CANNOT_DO"
            score = 0
            confidence = negation_score
        elif lost_skill_score > 0.7:
            # Strong indication of lost skill
            score_label = "LOST_SKILL"
            score = 1
            confidence = lost_skill_score
        elif uncertainty_score > 0.5 and independence_score < 0.5:
            # Signs of emerging but inconsistent skill
            score_label = "EMERGING"
            score = 2
            confidence = uncertainty_score
        elif support_score > 0.7:
            # Strong indication of requiring support
            score_label = "WITH_SUPPORT"
            score = 3
            confidence = support_score
        elif independence_score > 0.7:
            # Strong indication of independence
            score_label = "INDEPENDENT"
            score = 4
            confidence = independence_score
        elif progress_score > 0.5:
            # Signs of skill development
            score_label = "EMERGING"
            score = 2
            confidence = progress_score
        else:
            # Default case - analyze response length and content
            words = response_clean.split()
            if len(words) <= 1:
                if response_clean == "yes":
                    score_label = "INDEPENDENT"
                    score = 4
                    confidence = 0.8
                elif response_clean == "no":
                    score_label = "CANNOT_DO"
                    score = 0
                    confidence = 0.8
                elif response_clean in ["sometimes", "maybe"]:
                    score_label = "EMERGING"
                    score = 2
                    confidence = 0.8
                else:
                    # Insufficient information
                    score_label = "CANNOT_DO"
                    score = 0
                    confidence = 0.6
            else:
                # Prioritize negation over single "yes"
                if negation_score > 0 and independence_score < 0.9:
                    score_label = "CANNOT_DO"
                    score = 0
                    confidence = 0.7
                else:
                    # Default to WITH_SUPPORT with lower confidence
                    score_label = "WITH_SUPPORT"
                    score = 3
                    confidence = 0.5
        
        # Generate explanation
        explanation = f"Analysis of '{response_clean}': "
        if score_label == "CANNOT_DO":
            explanation += "Response indicates skill not acquired."
        elif score_label == "LOST_SKILL":
            explanation += "Response indicates skill was acquired but lost."
        elif score_label == "EMERGING":
            explanation += "Response indicates emerging or inconsistent skill."
        elif score_label == "WITH_SUPPORT":
            explanation += "Response indicates skill present with support."
        else:  # INDEPENDENT
            explanation += "Response indicates independent skill."
            
        # Return analysis results
        return {
            "milestone": milestone,
            "domain": domain,
            "score_label": score_label,
            "score": score,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "pattern_scores": pattern_scores,
            "linguistic_analysis": linguistic_scores
        }

    def handle_complex_response(self, response):
        """
        Special handler for complex responses with explanations that might confuse the model.
        
        This function specifically targets responses with patterns that indicate specific scoring:
        - CANNOT_DO (0): "no, not yet", "not at all", "never", etc.
        - LOST_SKILL (1): "used to", "previously", "lost", "regressed", etc.  
        - EMERGING (2): "sometimes", "occasionally", "not consistently", etc.
        - WITH_SUPPORT (3): "with help", "with support", "when prompted", etc.
        - INDEPENDENT (4): "yes", "always", "consistently", etc.
        
        Args:
            response (str): The cleaned response text
            
        Returns:
            dict or None: Analysis results if pattern matched, None otherwise
        """
        # Handle "no, not yet" and similar patterns - CANNOT_DO (0)
        cannot_do_patterns = [
            r"\b(no|not),?\s+(yet|yet started|started yet)",
            r"not at all",
            r"\bnever\b",
            r"doesn't ([a-z]+\s){0,3}(do|show|perform|demonstrate)",
            r"does not ([a-z]+\s){0,3}(do|show|perform|demonstrate)",
            r"doesn't ([a-z]+\s){0,3}at all", 
            r"unable to",
            r"can'?t\s+([a-z]+\s){0,3}(do|perform|demonstrate)",
            r"hasn'?t ([a-z]+\s){0,3}(done|shown|performed|demonstrated)",
            r"hasn'?t ([a-z]+\s){0,3}(started|begun|developed)",
            r"no ability",
            r"^no,\s+",  # Starts with "no," followed by any explanation
            r"^no\s+\w+\s+does not",  # Patterns like "no he does not"
            r"^no\s+\w+\s+doesn't"    # Patterns like "no he doesn't"
        ]
        
        for pattern in cannot_do_patterns:
            if re.search(pattern, response):
                print(f"Enhanced NLP detected CANNOT_DO pattern: '{pattern}'")
                return {
                    "score_label": "CANNOT_DO",
                    "score": 0,
                    "confidence": 0.95,
                    "explanation": "Response indicates skill has not been acquired."
                }
        
        # Special case: if response starts with "no" and contains negation, it's CANNOT_DO
        if response.startswith("no") and any(neg in response for neg in ["not", "doesn't", "does not", "never", "can't", "cannot"]):
            print(f"Enhanced NLP detected complex negation starting with 'no'")
            return {
                "score_label": "CANNOT_DO",
                "score": 0,
                "confidence": 0.95,
                "explanation": "Response starts with 'no' and contains negation, indicating skill has not been acquired."
            }
        
        # Handle "used to" and similar patterns - LOST_SKILL (1)
        lost_skill_patterns = [
            r"used to",
            r"was able to",
            r"could (before|previously)",
            r"previously",
            r"before but",
            r"(lost|regressed|stopped)",
            r"no longer",
            r"not anymore",
            r"did (before|earlier|previously)",
            r"has (lost|regressed)",
            r"used to be able"
        ]
        
        for pattern in lost_skill_patterns:
            if re.search(pattern, response):
                print(f"Enhanced NLP detected LOST_SKILL pattern: '{pattern}'")
                return {
                    "score_label": "LOST_SKILL",
                    "score": 1, 
                    "confidence": 0.95,
                    "explanation": "Response indicates skill was acquired but has been lost."
                }
                
        # Handle "sometimes" and similar patterns - EMERGING (2)
        emerging_patterns = [
            r"\bsometimes\b",
            r"\boccasionally\b",
            r"not (always|consistently)",
            r"inconsistent",
            r"(trying|beginning|starting) to",
            r"learning to",
            r"working on",
            r"developing",
            r"in progress",
            r"some days",
            r"hit(s)? and miss(es)?",
            r"(once|twice) in a while",
            r"not (steady|stable|reliable)",
            r"depends on"
        ]
        
        for pattern in emerging_patterns:
            if re.search(pattern, response):
                print(f"Enhanced NLP detected EMERGING pattern: '{pattern}'")
                return {
                    "score_label": "EMERGING",
                    "score": 2,
                    "confidence": 0.95,
                    "explanation": "Response indicates skill is emerging and inconsistent."
                }
                
        # Handle "with help" and similar patterns - WITH_SUPPORT (3)
        with_support_patterns = [
            r"with (help|support|assistance)",
            r"when (prompted|reminded|guided|helped|assisted)",
            r"needs (help|support|assistance|prompting|reminding)",
            r"if (i|we|someone) help",
            r"with (guidance|coaching)",
            r"can do it if",
            r"requires (help|support|assistance)",
            r"only with",
            r"hand[- ]?over[- ]?hand",
            r"physical (prompt|guidance|support|assistance)",
            r"verbal (prompt|cue|reminder)",
            r"when (shown|demonstrated)"
        ]
        
        for pattern in with_support_patterns:
            if re.search(pattern, response):
                print(f"Enhanced NLP detected WITH_SUPPORT pattern: '{pattern}'")
                return {
                    "score_label": "WITH_SUPPORT", 
                    "score": 3,
                    "confidence": 0.95,
                    "explanation": "Response indicates skill is acquired but only with support."
                }
                
        # INDEPENDENT will be handled by the default scoring logic
        # This let's us fall back to the regular analysis if none of the specific patterns match
            
        return None

def install_dependencies():
    """Install required dependencies if they are not already installed."""
    try:
        import pkg_resources
        
        # Define required packages
        required = {
            "spacy": "3.6.1",
            "numpy": "1.24.3"
        }
        
        # Check and install missing packages
        missing = []
        for package, version in required.items():
            try:
                pkg_resources.get_distribution(f"{package}>={version}")
            except pkg_resources.VersionConflict:
                missing.append(f"{package}>={version}")
            except pkg_resources.DistributionNotFound:
                missing.append(f"{package}>={version}")
        
        # Install missing packages
        if missing:
            print(f"Installing missing packages: {', '.join(missing)}")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            
            # Install spaCy model if needed
            if "spacy" in ''.join(missing):
                print("Installing spaCy model...")
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                
        return True
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        return False

def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced NLP for developmental milestone responses")
    parser.add_argument("--milestone", type=str, required=True, help="The developmental milestone to assess")
    parser.add_argument("--response", type=str, required=True, help="The caregiver's response")
    parser.add_argument("--domain", type=str, help="The developmental domain (optional)")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies if missing")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    
    # Create analyzer and process response
    analyzer = AdvancedResponseAnalyzer()
    result = analyzer.analyze_response(args.milestone, args.response, args.domain)
    
    # Print results
    print(f"Milestone: {args.milestone}")
    print(f"Response: {args.response}")
    print("\nAnalysis Results:")
    print(f"Score: {result['score']} ({result['score_label']})")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Explanation: {result['explanation']}")
    print("\nDetailed Analysis:")
    
    print("Pattern Matching:")
    for category, score in result['pattern_scores'].items():
        print(f"  {category}: {score:.2f}")
        
    print("\nLinguistic Features:")
    for feature, score in result['linguistic_analysis'].items():
        print(f"  {feature}: {score:.2f}")
    
    return result

if __name__ == "__main__":
    main() 