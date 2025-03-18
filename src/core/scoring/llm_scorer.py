#!/usr/bin/env python3
"""
LLM-Based Scorer Module

This module provides a scorer implementation using a local language model (LLM)
for developmental milestone assessment.
"""

import os
import sys
import json
import time
import logging
import platform
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("llama_cpp not available. LLMBasedScorer will not function without it.")
    LLAMA_CPP_AVAILABLE = False

from .base import BaseScorer, Score, ScoringResult
try:
    from ..knowledge import get_domain_by_name, format_prompt_with_context, load_prompt, get_age_specific_prompt, format_with_age_domain_context
    KNOWLEDGE_MODULE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_MODULE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_scorer")

class LLMBasedScorer(BaseScorer):
    """
    Scorer that uses a local Language Model (LLM) for milestone assessment.
    
    This class encapsulates the functionality needed to use local LLMs
    (like Mistral 7B) for developmental milestone assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM-based scorer
        
        Args:
            config: Configuration dictionary with LLM and prompt settings
        """
        super().__init__(config)
        
        self.model = None
        self.use_remote_api = False
        self._initialize_model()
        
        # Prompt-related config
        self.use_domain_prompts = self.config.get("use_domain_specific_prompts", True)
        self.use_age_prompts = self.config.get("use_age_specific_prompts", True)
        self.custom_templates_dir = self.config.get("custom_templates_dir", None)
        
        # Remote API config
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MISTRAL_API_KEY")
        self.api_base = os.environ.get("LLM_API_BASE", "https://api.mistral.ai/v1")
        self.api_model = os.environ.get("LLM_API_MODEL", "mistral-small")
        
        if self.use_remote_api and not self.api_key:
            logger.warning("Remote API enabled but no API key found. LLM scoring will likely fail.")
        
        logger.info(f"LLMBasedScorer initialized with model: {self.config.get('model_path', 'Remote API')}")
        logger.info(f"Domain-specific prompts: {self.use_domain_prompts}, Age-specific prompts: {self.use_age_prompts}")
    
    @staticmethod
    def _default_config():
        """
        Get default configuration for the LLM scorer
        
        Returns:
            Dict: Default configuration dictionary
        """
        # Find the models directory relative to the project root
        project_root = Path(__file__).parent.parent.parent.parent
        model_dir = project_root / "models"
        
        return {
            # Model settings
            "model_path": str(model_dir / "mistral-7b-instruct-v0.2.Q3_K_S.gguf"),
            "n_ctx": 2048,        # Context size
            "n_batch": 512,       # Batch size for prompt processing
            "n_gpu_layers": 0,    # Number of layers to offload to GPU (0 = CPU only)
            "n_threads": 4,       # Number of threads to use
            "f16_kv": True,       # Use half-precision for key/value cache
            
            # Inference settings
            "temperature": 0.1,   # Low temperature for more deterministic outputs
            "top_p": 0.9,         # Nucleus sampling parameter
            "top_k": 40,          # Top-k sampling parameter
            "max_tokens": 256,    # Maximum number of tokens to generate
            
            # Prompt settings
            "use_domain_specific_prompts": True,
            "use_age_specific_prompts": True,
            "custom_templates_dir": None,  # Custom directory for templates
            
            # Scoring settings
            "score_pattern": r"Score.*?(\d+)",  # Regex to extract score from response
            "confidence_default": 0.7,         # Default confidence if not extracted
            "retry_on_failure": True,          # Whether to retry on parsing failures
            "max_retries": 2                   # Maximum number of retries
        }
    
    def _initialize_model(self) -> None:
        """
        Initialize the LLM model for inference
        
        Raises:
            RuntimeError: If llama_cpp is not available or model loading fails
        """
        if not LLAMA_CPP_AVAILABLE:
            logger.error("Cannot initialize LLM: llama_cpp package not available")
            raise RuntimeError("llama_cpp package is required for LLMBasedScorer")
        
        # Check for model path in environment variables first
        env_model_path = os.environ.get("LLM_MODEL_PATH")
        if env_model_path:
            self.config["model_path"] = env_model_path
            logger.info(f"Using model path from environment variable: {env_model_path}")
        
        # Check if model file exists
        model_path = self.config["model_path"]
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at primary path: {model_path}")
            
            # Try alternative paths
            alt_paths = [
                os.path.join("models", os.path.basename(model_path)),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", os.path.basename(model_path)),
                os.path.join("/app/models", os.path.basename(model_path)),
                os.environ.get("MODELS_DIR", "")
            ]
            
            for alt_path in alt_paths:
                if alt_path and os.path.exists(alt_path):
                    logger.info(f"Found model at alternative path: {alt_path}")
                    model_path = alt_path
                    self.config["model_path"] = model_path
                    break
            
            # If still not found, try to use a remote API instead
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at any path. Will attempt to use remote API if configured.")
                self.use_remote_api = True
                self.model = None
                return
        
        try:
            logger.info(f"Loading LLM from {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.config["n_ctx"],
                n_batch=self.config["n_batch"],
                n_gpu_layers=self.config["n_gpu_layers"],
                n_threads=self.config["n_threads"],
                f16_kv=self.config["f16_kv"]
            )
            logger.info("LLM loaded successfully")
            self.use_remote_api = False
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            logger.warning("Will attempt to use remote API if configured")
            self.use_remote_api = True
            self.model = None
    
    def _format_prompt(self, response: str, milestone_context: Dict[str, Any]) -> str:
        """
        Format a prompt for the LLM based on the response and milestone context.
        
        Args:
            response: The parent's response about their child
            milestone_context: Dictionary with milestone information
            
        Returns:
            Formatted prompt string
        """
        # Try to get a domain-specific template
        domain = milestone_context.get("domain")
        age_months = None
        
        logger.info(f"Formatting prompt with domain: {domain}, milestone_context: {milestone_context}")
        
        if "age_range" in milestone_context:
            age_range = milestone_context.get("age_range", "")
            # Try to extract the lower bound of the age range
            if isinstance(age_range, str) and "-" in age_range:
                try:
                    age_months = int(age_range.split("-")[0])
                except (ValueError, IndexError):
                    pass
        
        prompt = None
        
        # Try domain-specific template first
        if domain:
            template = self._get_domain_specific_template(domain)
            logger.info(f"Domain template: {template}")
            if template:
                try:
                    # Try to import the format function
                    try:
                        from src.core.knowledge import format_prompt_with_context
                        logger.info(f"Template structure: {template.keys()}")
                        if "template" in template:
                            logger.info(f"Template content: {template['template'][:100]}...")
                            # The template is nested inside a 'template' key
                            # Create a properly formatted template dictionary
                            formatted_template = {
                                "base_template": template["template"]
                            }
                            try:
                                prompt = format_prompt_with_context(formatted_template, response, milestone_context)
                                logger.info(f"Formatted prompt with format_prompt_with_context: {prompt[:100]}...")
                            except Exception as e:
                                logger.warning(f"Error formatting with format_prompt_with_context: {e}")
                                # Direct string formatting as fallback
                                behavior = milestone_context.get("behavior", "Unknown milestone")
                                age_range = milestone_context.get("age_range", "")
                                prompt = template["template"].format(
                                    behavior=behavior,
                                    response=response,
                                    age_range=age_range
                                )
                                logger.info(f"Formatted prompt with direct formatting: {prompt[:100]}...")
                        else:
                            prompt = format_prompt_with_context(template, response, milestone_context)
                            logger.info(f"Formatted prompt with format_prompt_with_context: {prompt[:100]}...")
                    except (ImportError, AttributeError) as e:
                        # Fall back to direct string formatting
                        logger.warning(f"Could not import format_prompt_with_context: {e}, using direct formatting")
                        behavior = milestone_context.get("behavior", "Unknown milestone")
                        criteria = milestone_context.get("criteria", "")
                        
                        prompt = f"""
                        You are a developmental assessment expert analyzing a parent's response about their child's developmental milestone.
                        
                        Domain: {domain}
                        Milestone: {behavior}
                        Criteria: {criteria}
                        
                        Parent's Response: "{response}"
                        
                        Based solely on the parent's response, please assess the child's ability with this milestone.
                        Rate the child's ability on a scale of 0-4, where:
                        0 = CANNOT_DO (Child cannot perform this skill at all)
                        1 = LOST_SKILL (Child previously had the skill but has lost it)
                        2 = EMERGING (Skill is emerging but inconsistent)
                        3 = WITH_SUPPORT (Child can perform the skill with support or prompting)
                        4 = INDEPENDENT (Child can perform the skill independently and consistently)
                        
                        Provide your assessment in this format:
                        Score: [0-4]
                        Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
                        Reasoning: [Your explanation]
                        """
                except Exception as e:
                    logger.warning(f"Error formatting domain-specific prompt: {e}")
                    prompt = None
        
        # Try age-specific template if domain-specific failed
        if prompt is None and age_months is not None:
            template = self._get_age_specific_template(age_months)
            if template:
                try:
                    # Try to import the format function
                    try:
                        from src.core.knowledge import format_prompt_with_context
                        prompt = format_prompt_with_context(template, response, milestone_context)
                    except (ImportError, AttributeError):
                        # Fall back to direct string formatting
                        logger.warning("Could not import format_prompt_with_context, using direct formatting")
                        behavior = milestone_context.get("behavior", "Unknown milestone")
                        
                        prompt = f"""
                        You are a developmental assessment expert analyzing a parent's response about their child's developmental milestone.
                        
                        Age: {age_months} months
                        Milestone: {behavior}
                        
                        Parent's Response: "{response}"
                        
                        Based solely on the parent's response, please assess the child's ability with this milestone.
                        Rate the child's ability on a scale of 0-4, where:
                        0 = CANNOT_DO (Child cannot perform this skill at all)
                        1 = LOST_SKILL (Child previously had the skill but has lost it)
                        2 = EMERGING (Skill is emerging but inconsistent)
                        3 = WITH_SUPPORT (Child can perform the skill with support or prompting)
                        4 = INDEPENDENT (Child can perform the skill independently and consistently)
                        
                        Provide your assessment in this format:
                        Score: [0-4]
                        Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
                        Reasoning: [Your explanation]
                        """
                except Exception as e:
                    logger.warning(f"Error formatting age-specific prompt: {e}")
                    prompt = None
        
        # Use generic prompt as fallback
        if prompt is None:
            logger.warning(f"No suitable template found, using generic prompt")
            
            # Try to use our new formatting function
            try:
                from src.core.knowledge import format_with_age_domain_context
                logger.info("Using format_with_age_domain_context for fallback prompt")
                prompt = format_with_age_domain_context(response, milestone_context)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import format_with_age_domain_context: {e}")
                # Fall back to original generic prompt
                behavior = milestone_context.get("behavior", "Unknown milestone")
                domain_str = f"Domain: {domain}" if domain else ""
                age_str = f"Age: {age_months} months" if age_months else ""
                
                prompt = f"""
                You are a developmental assessment expert analyzing a parent's response about their child's developmental milestone.
                
                {domain_str}
                {age_str}
                Milestone: {behavior}
                
                Parent's Response: "{response}"
                
                Based solely on the parent's response, please assess the child's ability with this milestone.
                Rate the child's ability on a scale of 0-4, where:
                0 = CANNOT_DO (Child cannot perform this skill at all)
                1 = LOST_SKILL (Child previously had the skill but has lost it)
                2 = EMERGING (Skill is emerging but inconsistent)
                3 = WITH_SUPPORT (Child can perform the skill with support or prompting)
                4 = INDEPENDENT (Child can perform the skill independently and consistently)
                
                Provide your assessment in this format:
                Score: [0-4]
                Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
                Reasoning: [Your explanation]
                """
        
        return prompt
    
    def _get_domain_specific_template(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get a domain-specific prompt template
        
        Args:
            domain: The developmental domain
            
        Returns:
            Dict or None: Template dictionary or None if not found
        """
        # First try to use our new domain_templates module
        try:
            from src.core.knowledge import get_template_for_domain, DOMAIN_TEMPLATES
            template_dict = get_template_for_domain(domain)
            if template_dict:
                logger.info(f"Using domain template from domain_templates module for {domain}")
                return {"template": template_dict.get("template", "")}
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import from domain_templates module: {e}")
        
        # Check custom directory if provided
        if self.custom_templates_dir:
            template_path = os.path.join(self.custom_templates_dir, f"{domain}_template.json")
            if os.path.exists(template_path):
                try:
                    with open(template_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading custom template: {str(e)}")
        
        # Try loading from the standard location
        try:
            return load_prompt(f"{domain}_template")
        except Exception as e:
            logger.warning(f"Error loading standard template: {str(e)}")
            return None
    
    def _get_age_specific_template(self, age_months: int) -> Optional[Dict[str, Any]]:
        """
        Get an age-specific prompt template
        
        Args:
            age_months: Child's age in months
            
        Returns:
            Dict or None: Template dictionary or None if not found
        """
        # Check custom directory first if provided
        if self.custom_templates_dir:
            # Determine age bracket
            age_bracket = None
            if 0 <= age_months <= 12:
                age_bracket = "infant"
            elif 13 <= age_months <= 24:
                age_bracket = "toddler"
            elif 25 <= age_months <= 36:
                age_bracket = "preschooler"
            
            if age_bracket:
                template_path = os.path.join(self.custom_templates_dir, f"{age_bracket}_template.json")
                if os.path.exists(template_path):
                    try:
                        with open(template_path, 'r') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading custom age template: {str(e)}")
        
        # Create a default age-specific template
        age_bracket = None
        if 0 <= age_months <= 12:
            age_bracket = "infant (0-12 months)"
        elif 13 <= age_months <= 24:
            age_bracket = "toddler (13-24 months)"
        elif 25 <= age_months <= 36:
            age_bracket = "preschooler (25-36 months)"
        else:
            age_bracket = f"{age_months} months"
            
        # Create a basic template
        template = {
            "base_template": f"""
You are a developmental assessment expert specializing in child development for {age_bracket}.

Milestone: {{behavior}}
Age: {age_months} months
Age Range: {{age_range}}

Parent's Response: "{{response}}"

Based on the parent's response, assess if the child can complete this developmental milestone.
Consider what is age-appropriate for a child of {age_months} months.

Rate the child's ability on a scale of 0-4, where:
0 = CANNOT_DO (Child cannot perform this skill at all)
1 = LOST_SKILL (Child previously had the skill but has lost it)
2 = EMERGING (Skill is emerging but inconsistent)
3 = WITH_SUPPORT (Child can perform the skill with support or prompting)
4 = INDEPENDENT (Child can perform the skill independently and consistently)

Provide your assessment in this format:
Score: [0-4]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
        }
        
        logger.info(f"Created default age-specific template for {age_months} months")
        return template
    
    def _parse_llm_response(self, response_text: str) -> Tuple[Score, float, str]:
        """
        Parse the LLM response to extract score, confidence, and reasoning
        
        Args:
            response_text: Raw text response from the LLM
            
        Returns:
            Tuple: (score category, confidence, reasoning text)
        """
        # First, try to extract category labels directly
        category_patterns = [
            r"Category.*?([A-Z_]+)",  # Standard format: "Category: INDEPENDENT"
            r"([A-Z_]+)\s*\(\d+\)",   # Format like "INDEPENDENT (7)"
            r"([A-Z_]+)\s+category",  # Format like "INDEPENDENT category"
            r"classified\s+as\s+([A-Z_]+)",  # "classified as INDEPENDENT"
            r"falls\s+into\s+the\s+([A-Z_]+)",  # "falls into the INDEPENDENT"
            r"score\s+of\s+([A-Z_]+)"  # "score of INDEPENDENT"
        ]
        
        # Try to extract category directly
        for pattern in category_patterns:
            match = re.search(pattern, response_text)
            if match:
                category_str = match.group(1).strip()
                try:
                    # Try to convert the string to a Score enum
                    score = Score[category_str]
                    
                    # Extract reasoning (everything after the category)
                    reasoning_text = response_text[match.end():].strip()
                    if not reasoning_text:
                        # If no reasoning after category, use everything before it
                        reasoning_text = response_text[:match.start()].strip()
                    
                    if not reasoning_text:
                        reasoning_text = "Category extracted directly from model response"
                    
                    # Higher confidence for explicit category labels
                    return score, 0.85, reasoning_text
                except (KeyError, ValueError):
                    # Not a valid category name, continue to next pattern
                    pass
        
        # Extract the numeric score using regex if category extraction failed
        score_patterns = [
            r"Score.*?(\d+)",  # Standard format: "Score: 7"
            r"^\s*(\d+)\s*$",  # Just a number on a line by itself
            r"^\s*(\d+)\s*\n",  # Number at the beginning followed by newline
            r"(\d+)/\d+",      # Format like "7/7"
            r"[Ss]core\s*(?:is|:)?\s*(\d+)",  # Various forms of "score is X" or "score: X"
            r"rating\s*(?:is|:)?\s*(\d+)",    # "rating is X" or "rating: X"
            r"(\d+)\s*out of\s*7"  # Format like "6 out of 7"
        ]
        
        score_match = None
        for pattern in score_patterns:
            match = re.search(pattern, response_text)
            if match:
                score_match = match
                break
        
        if not score_match:
            logger.warning(f"Failed to extract score from response: {response_text}")
            # Try to extract any reasoning even if score is missing
            reasoning_text = response_text.strip()
            if not reasoning_text:
                reasoning_text = "Could not determine score from model response"
            return Score.NOT_RATED, 0.5, reasoning_text
        
        try:
            # Convert numeric score to Score enum
            score_value = int(score_match.group(1))
            
            # Map old 1-7 scale to new 0-4 scale if needed
            if score_value > 4:
                # Map 5-7 to 2-4
                if score_value == 5:
                    mapped_score_value = 2  # EMERGING
                elif score_value == 6:
                    mapped_score_value = 3  # WITH_SUPPORT
                elif score_value == 7:
                    mapped_score_value = 4  # INDEPENDENT
                else:
                    mapped_score_value = score_value
                
                logger.info(f"Mapped old scale score {score_value} to new scale score {mapped_score_value}")
                score_value = mapped_score_value
            
            score = self._numeric_to_score(score_value)
            
            # Extract reasoning (everything after the score)
            reasoning_text = response_text[score_match.end():].strip()
            if not reasoning_text:
                # If no reasoning after score, use everything before it
                reasoning_text = response_text[:score_match.start()].strip()
            
            if not reasoning_text:
                reasoning_text = "Score extracted from model response without explicit reasoning"
            
            # Determine confidence based on response clarity
            confidence_default = self.config.get("confidence_default", 0.7)
            confidence = confidence_default
            
            # Adjust confidence based on reasoning clarity
            if len(reasoning_text) > 100:  # Detailed reasoning
                confidence += 0.1
            if "although" in reasoning_text.lower() or "however" in reasoning_text.lower():
                confidence -= 0.1  # Some uncertainty
            
            # Cap confidence between 0.5 and 0.9
            confidence = max(0.5, min(0.9, confidence))
            
            # Fix inconsistencies in the reasoning text
            actual_score = score.value
            actual_category = score.name
            
            # Check for incorrect score mentions in the reasoning
            incorrect_score_pattern = r"[Aa]\s+score\s+of\s+(\d+)"
            incorrect_score_match = re.search(incorrect_score_pattern, reasoning_text)
            if incorrect_score_match:
                mentioned_score = int(incorrect_score_match.group(1))
                if mentioned_score != actual_score:
                    # Replace the incorrect score with the correct one
                    reasoning_text = re.sub(
                        incorrect_score_pattern,
                        f"A score of {actual_score}",
                        reasoning_text
                    )
                    logger.info(f"Corrected inconsistent score in reasoning: {mentioned_score} -> {actual_score}")
            
            # Check for any other numeric score mentions that might be inconsistent
            score_mentions_pattern = r"score\s+(?:of|is|:)?\s+(\d+)"
            for match in re.finditer(score_mentions_pattern, reasoning_text, re.IGNORECASE):
                mentioned_score = int(match.group(1))
                if mentioned_score != actual_score:
                    # Replace the incorrect score with the correct one
                    reasoning_text = reasoning_text[:match.start(1)] + str(actual_score) + reasoning_text[match.end(1):]
                    logger.info(f"Corrected another inconsistent score in reasoning: {mentioned_score} -> {actual_score}")
            
            # Check for category mentions that might be inconsistent with the actual score
            category_mapping = {
                0: "CANNOT_DO",
                1: "LOST_SKILL",
                2: "EMERGING",
                3: "WITH_SUPPORT",
                4: "INDEPENDENT",
                5: "EMERGING",  # Common mistake: using 5 for EMERGING
                6: "WITH_SUPPORT",  # Common mistake: using 6 for WITH_SUPPORT
                7: "INDEPENDENT"  # Common mistake: using 7 for INDEPENDENT
            }
            
            # If the reasoning mentions a category that doesn't match the actual score
            for score_val, category in category_mapping.items():
                if score_val != actual_score and category in reasoning_text:
                    # Only replace if it's a clear category reference
                    category_pattern = r"\b" + category + r"\b"
                    if re.search(category_pattern, reasoning_text):
                        reasoning_text = re.sub(
                            category_pattern,
                            actual_category,
                            reasoning_text
                        )
                        logger.info(f"Corrected inconsistent category in reasoning: {category} -> {actual_category}")
            
            # Final check for the specific pattern in the screenshot
            final_score_pattern = r"A score of (\d+) and the category of ([A-Z_]+)"
            final_match = re.search(final_score_pattern, reasoning_text)
            if final_match:
                mentioned_score = int(final_match.group(1))
                mentioned_category = final_match.group(2)
                if mentioned_score != actual_score or mentioned_category != actual_category:
                    # Replace with the correct score and category
                    reasoning_text = re.sub(
                        final_score_pattern,
                        f"A score of {actual_score} and the category of {actual_category}",
                        reasoning_text
                    )
                    logger.info(f"Corrected final score and category: {mentioned_score}/{mentioned_category} -> {actual_score}/{actual_category}")
            
            return score, confidence, reasoning_text
            
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return Score.NOT_RATED, 0.5, f"Error processing model response: {str(e)}"
    
    def _numeric_to_score(self, score_value: int) -> Score:
        """
        Convert numeric score to Score enum
        
        Args:
            score_value: Numeric score from LLM (0-4 scale)
            
        Returns:
            Score: Corresponding Score enum value
        """
        # Map directly from 0-4 scale to Score enum
        if score_value == 0:
            return Score.CANNOT_DO
        elif score_value == 1:
            return Score.LOST_SKILL
        elif score_value == 2:
            return Score.EMERGING
        elif score_value == 3:
            return Score.WITH_SUPPORT
        elif score_value == 4:
            return Score.INDEPENDENT
        else:
            logger.warning(f"Unexpected score value: {score_value}, defaulting to NOT_RATED")
            return Score.NOT_RATED
    
    def _call_remote_api(self, prompt: str) -> str:
        """
        Call a remote LLM API when local model is not available
        
        Args:
            prompt: The formatted prompt to send to the API
            
        Returns:
            str: The generated text response
            
        Raises:
            RuntimeError: If the API call fails
        """
        if not self.api_key:
            raise RuntimeError("No API key available for remote LLM API")
        
        try:
            import requests
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Determine which API format to use based on the API base URL
            if "openai" in self.api_base:
                # OpenAI-compatible format
                data = {
                    "model": self.api_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in child development assessment."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.get("temperature", 0.1),
                    "max_tokens": self.config.get("max_tokens", 256)
                }
            else:
                # Mistral AI format
                data = {
                    "model": self.api_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert in child development assessment."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.get("temperature", 0.1),
                    "max_tokens": self.config.get("max_tokens", 256)
                }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"API error: {response.status_code}")
            
            result = response.json()
            
            # Extract the generated text based on API format
            if "openai" in self.api_base:
                return result["choices"][0]["message"]["content"]
            else:
                return result["choices"][0]["message"]["content"]
                
        except ImportError:
            logger.error("requests package not available for API calls")
            raise RuntimeError("requests package is required for remote API calls")
        except Exception as e:
            logger.error(f"Error calling remote API: {str(e)}")
            raise RuntimeError(f"Error calling remote API: {str(e)}")

    def score(self, response: str, milestone_context: Dict[str, Any]) -> ScoringResult:
        """
        Score a response using LLM-based analysis
        
        Args:
            response: The text response to score
            milestone_context: Context about the milestone
            
        Returns:
            ScoringResult: The scoring result with score and confidence
        """
        # Format the prompt for the LLM
        prompt = self._format_prompt(response, milestone_context)
        
        # Add debug logging for the prompt
        logger.info(f"Sending prompt to LLM: {prompt}")
        
        # Generate text using either local model or remote API
        if self.use_remote_api:
            try:
                generated_text = self._call_remote_api(prompt)
            except Exception as e:
                logger.error(f"Remote API call failed: {str(e)}")
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=0.0,
                    method="llm_api_failed",
                    reasoning="LLM API call failed: " + str(e)
                )
        else:
            # Use local model
            if not self.model:
                logger.error("No local model available")
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=0.0,
                    method="llm_not_available",
                    reasoning="Local LLM model is not available"
                )
            
            try:
                # Generate text with the local model
                output = self.model(
                    prompt,
                    max_tokens=self.config.get("max_tokens", 256),
                    temperature=self.config.get("temperature", 0.1),
                    top_p=self.config.get("top_p", 0.9),
                    top_k=self.config.get("top_k", 40),
                    stop=["</s>", "User:", "Human:", "Question:"],
                    echo=False
                )
                
                generated_text = output["choices"][0]["text"]
                # Add debug logging for the generated text
                logger.info(f"LLM generated text: {generated_text}")
            except Exception as e:
                logger.error(f"Error generating text with local model: {str(e)}")
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=0.0,
                    method="llm_generation_failed",
                    reasoning="LLM text generation failed: " + str(e)
                )
        
        # Parse the generated text to extract the score
        try:
            # Add debug logging before parsing
            logger.info(f"Attempting to parse LLM response: {generated_text}")
            score, confidence, reasoning = self._parse_llm_response(generated_text)
            
            # Ensure reasoning is never None
            if reasoning is None or reasoning.strip() == "":
                reasoning = "No explicit reasoning provided by the LLM model."
            
            return ScoringResult(
                score=score,
                confidence=confidence,
                method="llm",
                reasoning=reasoning,
                details={"generated_text": generated_text}
            )
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="llm_parsing_failed",
                reasoning=f"Failed to parse LLM response: {str(e)}\nRaw text: {generated_text[:100]}..."
            )
    
    def batch_score(self, items: List[Dict[str, Any]]) -> List[ScoringResult]:
        """
        Score multiple responses in batch
        
        Args:
            items: List of dictionaries with 'response' and 'milestone_context' keys
            
        Returns:
            List[ScoringResult]: Scoring results for each item
        """
        results = []
        
        for item in items:
            response = item["response"]
            milestone_context = item.get("milestone_context", {})
            
            result = self.score(response, milestone_context)
            results.append(result)
            
            # Small delay to avoid overloading the model
            time.sleep(0.5)
        
        return results


# Test the scorer if run directly
if __name__ == "__main__":
    # Simple test case
    scorer = LLMBasedScorer()
    
    test_response = "Yes, my child can do this completely independently and has been doing it consistently for months."
    test_milestone = {
        "behavior": "Walks independently",
        "criteria": "Child walks without support for at least 10 steps",
        "age_range": "12-18 months"
    }
    
    result = scorer.score(test_response, test_milestone)
    
    print(f"Score: {result.score.name}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}") 