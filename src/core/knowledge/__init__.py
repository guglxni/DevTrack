"""
Knowledge Engineering Module

This package contains domain-specific knowledge and enhanced prompt engineering
for developmental milestone assessments.
"""

from typing import Dict, Any, Optional, List

from .developmental_domains import DevelopmentalDomain, get_domain_by_name, get_all_domains
from .prompt_templates import create_domain_specific_prompt, validate_prompt, save_prompt, load_prompt, format_prompt_with_context
from .category_knowledge import (
    get_category_evidence, 
    get_category_boundary, 
    get_domain_specific_evidence,
    get_all_categories,
    get_all_boundaries,
    CategoryEvidence,
    CategoryBoundary
)
from .category_helper import (
    get_research_based_definition,
    get_research_indicators,
    get_domain_indicators,
    get_boundary_criteria,
    get_domain_boundary_criteria,
    get_confidence_threshold,
    get_boundary_threshold,
    analyze_response_for_category,
    determine_category_from_response,
    refine_category_with_research,
    get_citation_for_category
)
from .text_analyzer import (
    analyze_text_for_category,
    get_best_category_match,
    extract_key_details,
    generate_analysis_explanation
)
from .age_specific_knowledge import (
    get_age_expectations,
    get_category_guidance,
    get_expected_skills,
    get_confidence_adjustment,
    get_age_bracket,
    adjust_category_for_age,
    AGE_BRACKETS
)

# Import the new domain templates module
try:
    from .domain_templates import (
        get_template_for_domain,
        save_templates_to_disk,
        load_template_from_disk,
        DOMAIN_TEMPLATES,
        BASE_TEMPLATE
    )
except ImportError:
    import logging
    logging.getLogger(__name__).warning("domain_templates module not found")

# Add the format_with_age_domain_context function
def format_with_age_domain_context(response: str, milestone_context: Dict[str, Any]) -> str:
    """
    Format a prompt with age and domain context.
    
    Args:
        response: The parent's response text
        milestone_context: Context about the milestone
        
    Returns:
        A formatted prompt string
    """
    domain = milestone_context.get("domain", "Unknown")
    age_range = milestone_context.get("age_range", "Unknown")
    behavior = milestone_context.get("behavior", "Unknown milestone")
    
    # Try to get a domain-specific template
    template = None
    try:
        if "get_template_for_domain" in globals():
            if domain:
                template_dict = get_template_for_domain(domain)
                if template_dict:
                    template = template_dict.get("template", None)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error getting domain template: {e}")
    
    # Fall back to base template if no domain template
    if not template:
        try:
            if "BASE_TEMPLATE" in globals():
                template = BASE_TEMPLATE
            else:
                # Fallback template if the import failed
                template = """
                You are a developmental assessment expert analyzing a parent's response about their child's milestone.
                
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
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error getting base template: {e}")
    
    # Format the template with the context
    try:
        return template.format(
            behavior=behavior,
            domain=domain,
            age_range=age_range,
            response=response
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Error formatting template: {e}")
        # Return a very simple template as a last resort
        return f"""
        Analyze this parent response about their child's milestone '{behavior}':
        
        "{response}"
        
        Provide a score from 1-7, a category (CANNOT_DO, LOST_SKILL, EMERGING, WITH_SUPPORT, INDEPENDENT), and your reasoning.
        """

__all__ = [
    # Developmental Domains
    'DevelopmentalDomain',
    'get_domain_by_name',
    'get_all_domains',
    
    # Prompt Templates
    'create_domain_specific_prompt',
    'validate_prompt',
    'save_prompt',
    'load_prompt',
    'format_prompt_with_context',
    'format_with_age_domain_context',
    
    # Domain Templates
    'get_template_for_domain',
    'save_templates_to_disk',
    'load_template_from_disk',
    'DOMAIN_TEMPLATES',
    'BASE_TEMPLATE',
    
    # Category Knowledge
    'get_category_evidence',
    'get_category_boundary',
    'get_domain_specific_evidence',
    'get_all_categories',
    'get_all_boundaries',
    'CategoryEvidence',
    'CategoryBoundary',
    
    # Category Helper Functions
    'get_research_based_definition',
    'get_research_indicators',
    'get_domain_indicators',
    'get_boundary_criteria',
    'get_domain_boundary_criteria',
    'get_confidence_threshold',
    'get_boundary_threshold',
    'analyze_response_for_category',
    'determine_category_from_response',
    'refine_category_with_research',
    'get_citation_for_category',
    
    # Text Analyzer Functions
    'analyze_text_for_category',
    'get_best_category_match',
    'extract_key_details',
    'generate_analysis_explanation',
    
    # Age-Specific Knowledge
    'get_age_expectations',
    'get_category_guidance',
    'get_expected_skills',
    'get_confidence_adjustment',
    'get_age_bracket',
    'adjust_category_for_age',
    'AGE_BRACKETS'
] 