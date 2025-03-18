"""
Domain-specific prompt templates for developmental assessment.

This module provides templates for each developmental domain that can be used
by the LLM scorer to better analyze parent responses.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Base template that all domain-specific templates extend
BASE_TEMPLATE = """
You are a developmental assessment expert analyzing a parent's response about their child's milestone in the {domain} domain.

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

# Domain-specific templates
DOMAIN_TEMPLATES = {
    # Social domain (SOC)
    "soc": {
        "name": "Social Domain",
        "description": "Social skills, interaction with others, forming relationships",
        "template": """
You are a developmental assessment expert specializing in social development in children.
        
Milestone: {behavior}
Domain: Social (SOC)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this social milestone.
Consider key indicators such as:
- Interaction with family members and peers
- Recognition of and response to others
- Participation in social activities
- Demonstration of social understanding and norms

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this social skill at all)
3 = LOST_SKILL (Child used to have this social skill but has lost it)
4-5 = EMERGING (Child is beginning to show this social skill sometimes)
6 = WITH_SUPPORT (Child can perform this social skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this social skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Gross Motor domain (GM)
    "gm": {
        "name": "Gross Motor Domain",
        "description": "Large muscle movements, coordination, and physical activities",
        "template": """
You are a developmental assessment expert specializing in gross motor development in children.
        
Milestone: {behavior}
Domain: Gross Motor (GM)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this gross motor milestone.
Consider key indicators such as:
- Large muscle coordination
- Balance and stability
- Strength and endurance
- Ability to perform physical activities

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this gross motor skill at all)
3 = LOST_SKILL (Child used to have this gross motor skill but has lost it)
4-5 = EMERGING (Child is beginning to show this gross motor skill sometimes)
6 = WITH_SUPPORT (Child can perform this gross motor skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this gross motor skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Fine Motor domain (FM)
    "fm": {
        "name": "Fine Motor Domain",
        "description": "Small muscle movements, precision, and hand-eye coordination",
        "template": """
You are a developmental assessment expert specializing in fine motor development in children.
        
Milestone: {behavior}
Domain: Fine Motor (FM)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this fine motor milestone.
Consider key indicators such as:
- Small muscle coordination
- Hand-eye coordination
- Precision and control
- Ability to manipulate small objects

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this fine motor skill at all)
3 = LOST_SKILL (Child used to have this fine motor skill but has lost it)
4-5 = EMERGING (Child is beginning to show this fine motor skill sometimes)
6 = WITH_SUPPORT (Child can perform this fine motor skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this fine motor skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Expressive Language domain (EL)
    "el": {
        "name": "Expressive Language Domain",
        "description": "Verbal expression, vocabulary, and communication skills",
        "template": """
You are a developmental assessment expert specializing in expressive language development in children.
        
Milestone: {behavior}
Domain: Expressive Language (EL)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this expressive language milestone.
Consider key indicators such as:
- Vocabulary size and use
- Sentence formation and complexity
- Clarity of speech
- Ability to express needs and ideas

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this expressive language skill at all)
3 = LOST_SKILL (Child used to have this expressive language skill but has lost it)
4-5 = EMERGING (Child is beginning to show this expressive language skill sometimes)
6 = WITH_SUPPORT (Child can perform this expressive language skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this expressive language skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Receptive Language domain (RL)
    "rl": {
        "name": "Receptive Language Domain",
        "description": "Understanding language, following directions, listening skills",
        "template": """
You are a developmental assessment expert specializing in receptive language development in children.
        
Milestone: {behavior}
Domain: Receptive Language (RL)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this receptive language milestone.
Consider key indicators such as:
- Understanding words and phrases
- Following directions
- Responding to questions
- Comprehending concepts

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this receptive language skill at all)
3 = LOST_SKILL (Child used to have this receptive language skill but has lost it)
4-5 = EMERGING (Child is beginning to show this receptive language skill sometimes)
6 = WITH_SUPPORT (Child can perform this receptive language skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this receptive language skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Emotional domain (Emo)
    "emo": {
        "name": "Emotional Domain",
        "description": "Emotional regulation, expression, and understanding",
        "template": """
You are a developmental assessment expert specializing in emotional development in children.
        
Milestone: {behavior}
Domain: Emotional (Emo)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this emotional milestone.
Consider key indicators such as:
- Emotional regulation
- Expression of feelings
- Empathy and understanding others' emotions
- Coping with challenges

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this emotional skill at all)
3 = LOST_SKILL (Child used to have this emotional skill but has lost it)
4-5 = EMERGING (Child is beginning to show this emotional skill sometimes)
6 = WITH_SUPPORT (Child can perform this emotional skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this emotional skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Activities of Daily Living domain (ADL)
    "adl": {
        "name": "Activities of Daily Living Domain",
        "description": "Self-care, independence in daily tasks, and practical skills",
        "template": """
You are a developmental assessment expert specializing in activities of daily living development in children.
        
Milestone: {behavior}
Domain: Activities of Daily Living (ADL)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this daily living milestone.
Consider key indicators such as:
- Self-care abilities
- Independence in routine tasks
- Personal responsibility
- Practical life skills

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this daily living skill at all)
3 = LOST_SKILL (Child used to have this daily living skill but has lost it)
4-5 = EMERGING (Child is beginning to show this daily living skill sometimes)
6 = WITH_SUPPORT (Child can perform this daily living skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this daily living skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    },
    
    # Cognitive domain (Cog)
    "cog": {
        "name": "Cognitive Domain",
        "description": "Thinking, learning, problem-solving, and mental processes",
        "template": """
You are a developmental assessment expert specializing in cognitive development in children.
        
Milestone: {behavior}
Domain: Cognitive (Cog)
Age Range: {age_range}

Parent's Response: "{response}"

Based on the parent's response, assess if the child can complete this cognitive milestone.
Consider key indicators such as:
- Problem-solving abilities
- Memory and attention
- Conceptual understanding
- Learning and application of knowledge

Rate the child's ability on a scale of 1-7, where:
1-2 = CANNOT_DO (Child cannot perform this cognitive skill at all)
3 = LOST_SKILL (Child used to have this cognitive skill but has lost it)
4-5 = EMERGING (Child is beginning to show this cognitive skill sometimes)
6 = WITH_SUPPORT (Child can perform this cognitive skill with help or prompting)
7 = INDEPENDENT (Child consistently performs this cognitive skill independently)

Provide your assessment in this format:
Score: [1-7]
Category: [CANNOT_DO|LOST_SKILL|EMERGING|WITH_SUPPORT|INDEPENDENT]
Reasoning: [Your explanation]
"""
    }
}

def get_template_for_domain(domain: str) -> Optional[Dict[str, Any]]:
    """
    Get a template for a specific developmental domain.
    
    Args:
        domain: Domain code (soc, gm, fm, etc.)
        
    Returns:
        Template dictionary or None if domain not found
    """
    if not domain:
        logger.warning("Empty domain provided")
        return None
        
    domain_lower = domain.lower()
    
    # Direct match
    if domain_lower in DOMAIN_TEMPLATES:
        logger.info(f"Found exact match for domain {domain} -> {domain_lower}")
        return DOMAIN_TEMPLATES[domain_lower]
    
    # Try to match partial domain name
    for key, template in DOMAIN_TEMPLATES.items():
        if domain_lower in key or key in domain_lower:
            logger.info(f"Found partial match for domain {domain} -> {key}")
            return template
    
    # No match found
    logger.warning(f"No template found for domain: {domain}")
    return None

def save_templates_to_disk(directory: str = "templates") -> bool:
    """
    Save all templates to disk for easy loading.
    
    Args:
        directory: Directory to save templates to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        
        # Save each domain template
        for domain, template in DOMAIN_TEMPLATES.items():
            filename = os.path.join(directory, f"{domain}_template.json")
            with open(filename, 'w') as f:
                json.dump(template, f, indent=2)
            
        logger.info(f"Saved {len(DOMAIN_TEMPLATES)} templates to {directory}")
        return True
    except Exception as e:
        logger.error(f"Error saving templates: {e}")
        return False

def load_template_from_disk(domain: str, directory: str = "templates") -> Optional[Dict[str, Any]]:
    """
    Load a template from disk.
    
    Args:
        domain: Domain code (soc, gm, fm, etc.)
        directory: Directory to load templates from
        
    Returns:
        Template dictionary or None if not found
    """
    try:
        domain = domain.lower() if domain else ""
        filename = os.path.join(directory, f"{domain}_template.json")
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                template = json.load(f)
            logger.info(f"Loaded template for {domain} from disk")
            return template
        else:
            logger.warning(f"No template file found for {domain}")
            return None
    except Exception as e:
        logger.error(f"Error loading template for {domain}: {e}")
        return None 