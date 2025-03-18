"""
Prompt Templates Module

This module provides functions for creating, validating, and managing domain-specific 
prompt templates for milestone assessment using language models.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import random

# Set up logging
logger = logging.getLogger("knowledge_engineering")

# Directory for prompt templates
PROMPT_TEMPLATES_DIR = Path("config/prompt_templates")


def _get_base_prompt_template() -> str:
    """
    Get the base prompt template shared across all domains
    
    Returns:
        str: Base prompt template with placeholders
    """
    return """You are an expert in child development assessment, specializing in {domain_name}.

Your task is to evaluate a parent's response about their child's developmental milestone.

MILESTONE: {milestone}
CRITERIA: {criteria}
AGE RANGE: {age_range}
DOMAIN: {domain_name}

PARENT'S RESPONSE: "{response}"

{domain_guidance}

Based on the parent's response, determine if the child's milestone status is:

1. CANNOT_DO (1): {cannot_do_desc}
2. WITH_SUPPORT (3): {with_support_desc}
3. EMERGING (5): {emerging_desc}
4. INDEPENDENT (7): {independent_desc}
5. LOST_SKILL (2): {lost_skill_desc}

IMPORTANT: Analyze the response carefully for specific indicators. Consider the milestone criteria, child's age, and domain-specific factors.

Provide your assessment as a single score (1-7) and a brief explanation.

Score (1-7):"""


def create_domain_specific_prompt(domain_name: str) -> Optional[Dict[str, Any]]:
    """
    Create a domain-specific prompt template
    
    Args:
        domain_name: Name or code of the developmental domain
        
    Returns:
        Dict or None: The created template or None if domain not found
    """
    # This is a placeholder - we'll need to implement the actual domain lookup
    # and template creation once we integrate with developmental_domains.py
    
    # For now, return a sample template
    sample_template = {
        "template_id": f"{domain_name.lower()}_template",
        "domain_code": domain_name,
        "domain_name": domain_name.title(),
        "base_template": _get_base_prompt_template(),
        "domain_guidance": "Sample domain guidance for " + domain_name,
        "category_descriptions": {
            "cannot_do_desc": "Child cannot perform the skill at all",
            "with_support_desc": "Child can do with help or inconsistently", 
            "emerging_desc": "Child is beginning to show the skill",
            "independent_desc": "Child can do consistently without help",
            "lost_skill_desc": "Child could do this before but has lost the skill"
        },
        "version": "1.0.0"
    }
    
    return sample_template


def format_prompt_with_context(template: Dict[str, Any], 
                              response: str, 
                              milestone_context: Dict[str, Any]) -> str:
    """
    Format a prompt template with milestone context
    
    Args:
        template: The prompt template dictionary
        response: The parent's response text
        milestone_context: Context about the milestone being assessed
        
    Returns:
        str: The formatted prompt
    """
    # Extract milestone information
    behavior = milestone_context.get("behavior", "")
    criteria = milestone_context.get("criteria", "")
    age_range = milestone_context.get("age_range", "")
    
    # Get the base template
    base_template = template.get("base_template", "")
    
    # Get domain guidance
    domain_guidance = template.get("domain_guidance", "")
    
    # Get category descriptions
    category_descriptions = template.get("category_descriptions", {})
    cannot_do_desc = category_descriptions.get("cannot_do_desc", "Child cannot perform the skill at all")
    with_support_desc = category_descriptions.get("with_support_desc", "Child can do with help or inconsistently")
    emerging_desc = category_descriptions.get("emerging_desc", "Child is beginning to show the skill")
    independent_desc = category_descriptions.get("independent_desc", "Child can do consistently without help")
    lost_skill_desc = category_descriptions.get("lost_skill_desc", "Child could do this before but has lost the skill")
    
    # Format the prompt
    prompt = base_template.format(
        milestone=behavior,
        criteria=criteria,
        age_range=age_range,
        response=response,
        domain_guidance=domain_guidance,
        cannot_do_desc=cannot_do_desc,
        with_support_desc=with_support_desc,
        emerging_desc=emerging_desc,
        independent_desc=independent_desc,
        lost_skill_desc=lost_skill_desc
    )
    
    return prompt


def validate_prompt(prompt: str, expected_tokens: List[str]) -> Tuple[bool, str]:
    """
    Validate a prompt for completeness and correctness
    
    Args:
        prompt: The formatted prompt to validate
        expected_tokens: List of expected tokens/phrases that should be in the prompt
        
    Returns:
        Tuple: (is_valid, validation_message)
    """
    missing_tokens = []
    
    for token in expected_tokens:
        if token not in prompt:
            missing_tokens.append(token)
    
    if missing_tokens:
        return False, f"Prompt missing expected elements: {', '.join(missing_tokens)}"
    
    # Check for reasonable length (< 2048 tokens ~= 8000 chars)
    if len(prompt) > 8000:
        return False, f"Prompt too long: {len(prompt)} chars (max ~8000 recommended)"
    
    return True, "Prompt validation successful"


def save_prompt(template: Dict[str, Any], validate: bool = True) -> bool:
    """
    Save a prompt template to the configuration directory
    
    Args:
        template: The template dictionary to save
        validate: Whether to validate the template before saving
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(PROMPT_TEMPLATES_DIR, exist_ok=True)
        
        # Validate template format
        if validate:
            required_fields = ["template_id", "domain_code", "domain_name", "base_template", 
                              "domain_guidance", "category_descriptions", "version"]
            
            missing_fields = [field for field in required_fields if field not in template]
            if missing_fields:
                logger.error(f"Template missing required fields: {', '.join(missing_fields)}")
                return False
        
        # Save template to file
        template_path = PROMPT_TEMPLATES_DIR / f"{template['template_id']}.json"
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
            
        logger.info(f"Saved prompt template to {template_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving prompt template: {str(e)}")
        return False


def load_prompt(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a prompt template from the configuration directory
    
    Args:
        template_id: ID of the template to load
        
    Returns:
        Dict or None: The loaded template or None if not found
    """
    try:
        # Try with exact template ID
        template_path = PROMPT_TEMPLATES_DIR / f"{template_id}.json"
        
        # If not found, try with domain name format
        if not template_path.exists():
            domain_template_path = PROMPT_TEMPLATES_DIR / f"{template_id.lower()}_template.json"
            if domain_template_path.exists():
                template_path = domain_template_path
            else:
                logger.error(f"Template not found: {template_id}")
                return None
            
        with open(template_path, 'r') as f:
            template = json.load(f)
            
        logger.info(f"Loaded prompt template from {template_path}")
        return template
        
    except Exception as e:
        logger.error(f"Error loading prompt template: {str(e)}")
        return None


def create_milestone_specific_prompts():
    """Generate specialized prompts for different developmental domains"""
    domains = ["motor", "communication", "social", "cognitive"]
    
    created_templates = []
    for domain in domains:
        # Create template
        template = create_domain_specific_prompt(domain)
        if not template:
            logger.error(f"Failed to create template for domain: {domain}")
            continue
            
        # Validate template with expected tokens
        sample_prompt = format_prompt_with_context(
            template,
            "Sample response for validation",
            {
                "behavior": "Sample milestone",
                "criteria": "Sample criteria",
                "age_range": "12-24 months"
            }
        )
        
        expected_tokens = [
            "expert in child development",
            "MILESTONE:",
            "CRITERIA:",
            "AGE RANGE:",
            "PARENT'S RESPONSE:",
            "CANNOT_DO",
            "WITH_SUPPORT",
            "EMERGING",
            "INDEPENDENT",
            "LOST_SKILL"
        ]
        
        is_valid, validation_message = validate_prompt(sample_prompt, expected_tokens)
        if not is_valid:
            logger.error(f"Template validation failed for {domain}: {validation_message}")
            continue
            
        # Save template
        if save_prompt(template):
            created_templates.append(template["template_id"])
        
    return created_templates


def format_with_age_domain_context(response: str, milestone_context: Dict[str, Any]) -> str:
    """
    Format a prompt with both age and domain-specific context
    
    Args:
        response: The parent's response text
        milestone_context: Context about the milestone being assessed
        
    Returns:
        str: The formatted prompt with age and domain context
    """
    # Extract milestone information
    behavior = milestone_context.get("behavior", "")
    criteria = milestone_context.get("criteria", "")
    age_range = milestone_context.get("age_range", "")
    domain = milestone_context.get("domain", "").lower()
    
    # Create a generic template with age and domain considerations
    template = {
        "base_template": """You are an expert in child development assessment, specializing in {domain_name}.

Your task is to evaluate a parent's response about their child's developmental milestone.

MILESTONE: {milestone}
CRITERIA: {criteria}
AGE RANGE: {age_range}
DOMAIN: {domain_name}

PARENT'S RESPONSE: "{response}"

Based on the parent's response, determine if the child's milestone status is:

1. CANNOT_DO (0): Child cannot perform the skill at all
2. LOST_SKILL (1): Child could do this before but has lost the skill
3. EMERGING (2): Child is beginning to show the skill
4. WITH_SUPPORT (3): Child can do with help or inconsistently
5. INDEPENDENT (4): Child can do consistently without help

IMPORTANT: Analyze the response carefully for specific indicators. Consider the milestone criteria, child's age, and domain-specific factors.

Provide your assessment as a single score (0-4) and a brief explanation.

Score (0-4):"""
    }
    
    # Format the prompt
    prompt = template["base_template"].format(
        milestone=behavior,
        criteria=criteria,
        age_range=age_range,
        domain_name=domain.upper() if domain else "Child Development",
        response=response
    )
    
    return prompt


if __name__ == "__main__":
    # Create all prompt templates if run directly
    created = create_milestone_specific_prompts()
    print(f"Created {len(created)} prompt templates: {', '.join(created)}") 