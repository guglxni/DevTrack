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

class HybridScorer:
    """
    Advanced hybrid scoring system that combines multiple NLP approaches:
    1. Word boundary-aware keyword matching
    2. Negation detection with context windows
    3. Semantic similarity using sentence embeddings
    4. Weighted ensemble for final decision
    """
    def __init__(self, use_transformer=False, use_spacy=False, use_sentence_embeddings=False):
        """
        Initialize the hybrid scorer with configurable components.
        
        Args:
            use_transformer: Whether to use transformer models for classification
            use_spacy: Whether to use spaCy for NER and linguistic analysis
            use_sentence_embeddings: Whether to use sentence embeddings
        """
        self.use_transformer = use_transformer
        self.use_spacy = use_spacy
        self.use_sentence_embeddings = use_sentence_embeddings
        self.negation_regex = re.compile(r'\b(no|not|never|doesn\'t|does not|cannot|can\'t|unable|hasn\'t|has not)\b')
        self.positive_regex = re.compile(r'\b(yes|always|consistently|definitely|very well)\b')
        
        # Word boundary regex for each score category
        self.score_regexes = {
            "CANNOT_DO": re.compile(r'\b(no|not|never|doesn\'t|does not|cannot|can\'t|unable|hasn\'t|has not|not able|not at all|not yet started|not capable)\b'),
            "LOST_SKILL": re.compile(r'\b(used to|previously|before|no longer|stopped|regressed|lost ability|could before|forgotten how)\b'),
            "EMERGING": re.compile(r'\b(sometimes|occasionally|beginning to|starting to|trying to|inconsistently|might|rarely|not consistent|learning to)\b'),
            "WITH_SUPPORT": re.compile(r'\b(with help|when assisted|with support|with guidance|needs help|when prompted|specific situations|only when|if guided|with assistance)\b'),
            "INDEPENDENT": re.compile(r'\b(yes|always|consistently|definitely|independently|without help|on own|mastered|very good at|excellent|regularly|all situations)\b')
        }
        
        # Try to load advanced components if requested
        if use_transformer:
            try:
                from transformers import pipeline
                self.transformer = pipeline("zero-shot-classification")
                logger.info("Transformer model loaded successfully")
            except ImportError:
                logger.warning("Transformers not available - disabling transformer model")
                self.use_transformer = False
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except ImportError:
                logger.warning("spaCy not available - disabling spaCy model")
                self.use_spacy = False
        
        if use_sentence_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            except ImportError:
                logger.warning("Sentence transformers not available - disabling sentence embeddings")
                self.use_sentence_embeddings = False
    
    def score_with_word_boundaries(self, response_text, milestone=None, keywords=None):
        """
        Score response text using regex patterns with proper word boundaries.
        
        Args:
            response_text: The text response to analyze
            milestone: Optional milestone for context
            keywords: Optional custom keywords for each score category
        
        Returns:
            Dict with score category and count of matched patterns
        """
        response_lower = response_text.lower()
        logger.info(f"Analyzing with word boundaries: '{response_lower}'")
        
        # Initialize counters for each score category
        score_counts = {"CANNOT_DO": 0, "LOST_SKILL": 0, "EMERGING": 0, 
                        "WITH_SUPPORT": 0, "INDEPENDENT": 0}
        
        # Track matched keywords for debugging
        matched_keywords = {category: [] for category in score_counts.keys()}
        
        # Use custom keywords if provided
        if keywords:
            for category, keyword_list in keywords.items():
                if category in score_counts:
                    # Create a regex pattern with word boundaries for each keyword
                    for keyword in keyword_list:
                        keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = re.findall(keyword_pattern, response_lower)
                        if matches:
                            score_counts[category] += len(matches)
                            matched_keywords[category].extend(matches)
        # Otherwise use the default regexes
        else:
            for category, regex in self.score_regexes.items():
                matches = regex.findall(response_lower)
                score_counts[category] += len(matches)
                matched_keywords[category].extend(matches)
        
        # Log the matches for debugging
        for category, keywords in matched_keywords.items():
            if keywords:
                logger.info(f"Matched {category} keywords: {', '.join(keywords)}")
        
        # Find the category with the highest score
        if all(count == 0 for count in score_counts.values()):
            best_category = None
            confidence = 0.0
        else:
            best_category = max(score_counts.items(), key=lambda x: x[1])[0]
            total_matches = sum(score_counts.values())
            confidence = score_counts[best_category] / total_matches if total_matches > 0 else 0.0
        
        # Convert category name to score value
        score_values = {"CANNOT_DO": 0, "LOST_SKILL": 1, "EMERGING": 2, 
                        "WITH_SUPPORT": 3, "INDEPENDENT": 4}
        
        score = score_values.get(best_category, -1) if best_category else -1
        
        return {
            "score": score,
            "score_label": best_category or "NOT_RATED",
            "confidence": confidence,
            "matches": matched_keywords
        }
    
    def check_for_negations(self, response_text):
        """
        Check for negations with context window analysis.
        
        Args:
            response_text: The text response to analyze
        
        Returns:
            Dict with negation score
        """
        response_lower = response_text.lower()
        
        # Check for clear negative indicators
        negation_matches = self.negation_regex.findall(response_lower)
        positive_matches = self.positive_regex.findall(response_lower)
        
        negation_count = len(negation_matches)
        positive_count = len(positive_matches)
        
        # For each negation, check the surrounding context (5 words before and after)
        sentences = re.split(r'[.!?]', response_lower)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if self.negation_regex.search(word):
                    # Check if this negation applies to the milestone capability
                    # Look for positive terms that might be negated
                    context_start = max(0, i - 5)
                    context_end = min(len(words), i + 6)
                    context = words[context_start:context_end]
                    
                    # If we find positive capability terms in this context, increase negation weight
                    capability_terms = ['can', 'able', 'does', 'is', 'has', 'shows', 'demonstrates']
                    for term in capability_terms:
                        if term in context:
                            negation_count += 1
                            break
        
        # Determine result based on counts
        if negation_count > 0 and negation_count > positive_count:
            score = 0  # CANNOT_DO
            confidence = min(1.0, negation_count / (negation_count + positive_count + 1))
        elif positive_count > 0:
            score = 4  # INDEPENDENT
            confidence = min(1.0, positive_count / (negation_count + positive_count + 1))
        else:
            score = -1  # NOT_RATED
            confidence = 0.0
        
        return {
            "score": score,
            "score_label": "CANNOT_DO" if score == 0 else "INDEPENDENT" if score == 4 else "NOT_RATED",
            "confidence": confidence
        }
    
    def score_with_transformer(self, response_text, milestone=None):
        """
        Use a transformer model for zero-shot classification.
        
        Args:
            response_text: The text response to analyze
            milestone: Optional milestone for context
        
        Returns:
            Dict with score results
        """
        if not self.use_transformer:
            return {"score": -1, "score_label": "NOT_RATED", "confidence": 0.0, "method": "transformer_unavailable"}
        
        try:
            # Define label descriptions for zero-shot classification
            candidate_labels = [
                "Child cannot perform this skill at all",
                "Child used to have this skill but lost it",
                "Child is beginning to develop this skill",
                "Child can do this with help or support",
                "Child can do this independently"
            ]
            
            # Incorporate milestone in query if available
            if milestone:
                query = f"Milestone: {milestone}. Response: {response_text}"
            else:
                query = response_text
            
            # Run classification
            result = self.transformer(query, candidate_labels)
            
            # Map to score
            score_mapping = {
                "Child cannot perform this skill at all": 0,         # CANNOT_DO
                "Child used to have this skill but lost it": 1,      # LOST_SKILL
                "Child is beginning to develop this skill": 2,       # EMERGING
                "Child can do this with help or support": 3,         # WITH_SUPPORT
                "Child can do this independently": 4                 # INDEPENDENT
            }
            
            best_label = result["labels"][0]
            score = score_mapping[best_label]
            confidence = result["scores"][0]
            
            return {
                "score": score,
                "score_label": list(score_mapping.keys())[list(score_mapping.values()).index(score)],
                "confidence": confidence,
                "method": "transformer"
            }
        
        except Exception as e:
            logger.error(f"Error in transformer scoring: {str(e)}")
            return {"score": -1, "score_label": "NOT_RATED", "confidence": 0.0, "method": "transformer_error"}
    
    def semantic_scoring(self, response_text, milestone=None):
        """
        Use sentence embeddings to compare with canonical examples.
        
        Args:
            response_text: The text response to analyze
            milestone: Optional milestone for context
        
        Returns:
            Dict with score results
        """
        if not self.use_sentence_embeddings:
            return {"score": -1, "score_label": "NOT_RATED", "confidence": 0.0, "method": "embeddings_unavailable"}
        
        try:
            import torch
            
            # Create embeddings for canonical examples of each category
            examples = {
                "INDEPENDENT": [
                    "Child always recognizes familiar people without any help",
                    "Child consistently identifies family members and distinguishes them from strangers",
                    "Yes, my child does this very well all the time",
                    "Completely independent with this skill"
                ],
                "WITH_SUPPORT": [
                    "Child recognizes people with some help",
                    "Needs prompting to recognize family members",
                    "Can do this with assistance",
                    "Does this when supported by an adult"
                ],
                "EMERGING": [
                    "Child is starting to recognize some people",
                    "Sometimes recognizes family members",
                    "Beginning to show this skill",
                    "Occasionally demonstrates this ability"
                ],
                "LOST_SKILL": [
                    "Child used to recognize people but doesn't anymore",
                    "Previously could do this but has regressed",
                    "Had this skill before but lost it",
                    "Used to be able to do this"
                ],
                "CANNOT_DO": [
                    "Child never recognizes anyone, even parents",
                    "Does not show any recognition of familiar people",
                    "Unable to do this at all",
                    "No, my child cannot do this"
                ]
            }
            
            # Add milestone context if available
            if milestone:
                for category in examples:
                    examples[category] = [f"For milestone '{milestone}': {ex}" for ex in examples[category]]
            
            # Create embeddings
            response_embedding = self.sentence_model.encode(response_text, convert_to_tensor=True)
            
            # Compare with each category
            scores = {}
            for category, category_examples in examples.items():
                category_embeddings = self.sentence_model.encode(category_examples, convert_to_tensor=True)
                # Get max similarity score for this category
                similarities = torch.nn.functional.cosine_similarity(
                    response_embedding.unsqueeze(0), 
                    category_embeddings, 
                    dim=1
                )
                scores[category] = torch.max(similarities).item()
            
            # Get highest scoring category
            best_category = max(scores.items(), key=lambda x: x[1])
            
            # Convert category to score
            score_values = {
                "CANNOT_DO": 0, 
                "LOST_SKILL": 1, 
                "EMERGING": 2, 
                "WITH_SUPPORT": 3, 
                "INDEPENDENT": 4
            }
            
            score = score_values.get(best_category[0], -1)
            
            return {
                "score": score,
                "score_label": best_category[0],
                "confidence": best_category[1],
                "method": "semantic_embeddings"
            }
            
        except Exception as e:
            logger.error(f"Error in semantic scoring: {str(e)}")
            return {"score": -1, "score_label": "NOT_RATED", "confidence": 0.0, "method": "embeddings_error"}
    
    def weighted_ensemble(self, scores):
        """
        Combine multiple scoring approaches using a weighted ensemble.
        
        Args:
            scores: List of (result_dict, weight) tuples
        
        Returns:
            Dict with final score results
        """
        # Filter out scores with negative values (NOT_RATED)
        valid_scores = [(result, weight) for result, weight in scores if result["score"] >= 0]
        
        if not valid_scores:
            return {
                "score": -1,
                "score_label": "NOT_RATED",
                "confidence": 0.0,
                "methods": [s[0]["method"] for s in scores if "method" in s[0]]
            }
        
        # Calculate weighted scores for each category
        weighted_scores = defaultdict(float)
        total_weight = 0
        
        for result, weight in valid_scores:
            weighted_scores[result["score"]] += result["confidence"] * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for score in weighted_scores:
                weighted_scores[score] /= total_weight
        
        # Select best score
        best_score = max(weighted_scores.items(), key=lambda x: x[1])
        
        # Map score value to label
        score_labels = {
            0: "CANNOT_DO",
            1: "LOST_SKILL",
            2: "EMERGING",
            3: "WITH_SUPPORT",
            4: "INDEPENDENT"
        }
        
        return {
            "score": best_score[0],
            "score_label": score_labels[best_score[0]],
            "confidence": best_score[1],
            "methods": [s[0].get("method", "unknown") for s in valid_scores]
        }
    
    def score(self, response_text, milestone=None, keywords=None):
        """
        Score a response using the hybrid approach.
        
        Args:
            response_text: The text response to analyze
            milestone: Optional milestone text for context
            keywords: Optional custom keywords for each score category
        
        Returns:
            Dict with final score results
        """
        # Step 1: Word boundary-aware keyword matching
        boundary_result = self.score_with_word_boundaries(response_text, milestone, keywords)
        
        # Step 2: Check for negations
        negation_result = self.check_for_negations(response_text)
        
        # Step 3: Transformer-based classification (if available)
        transformer_result = self.score_with_transformer(response_text, milestone) if self.use_transformer else {
            "score": -1, 
            "score_label": "NOT_RATED", 
            "confidence": 0.0,
            "method": "transformer_disabled"
        }
        
        # Step 4: Semantic similarity (if available)
        semantic_result = self.semantic_scoring(response_text, milestone) if self.use_sentence_embeddings else {
            "score": -1, 
            "score_label": "NOT_RATED", 
            "confidence": 0.0,
            "method": "embeddings_disabled"
        }
        
        # Step 5: Weighted ensemble
        ensemble_weights = [
            (boundary_result, 0.5),    # Word boundary matching has highest weight
            (negation_result, 0.3),    # Negation detection is important
            (transformer_result, 0.1), # Lower weight as it may not be available
            (semantic_result, 0.1)     # Lower weight as it may not be available
        ]
        
        final_result = self.weighted_ensemble(ensemble_weights)
        
        # Add detailed info for debugging
        final_result["detail"] = {
            "boundary": boundary_result,
            "negation": negation_result,
            "transformer": transformer_result,
            "semantic": semantic_result
        }
        
        return final_result

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