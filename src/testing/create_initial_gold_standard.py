#!/usr/bin/env python3
"""
Create Initial Gold Standard Dataset

This script generates an initial gold standard dataset for the developmental milestone
scoring system, combining generated test data with manually curated examples.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.append('.')

from src.testing.gold_standard_manager import GoldStandardManager
from src.testing.enhanced_test_generator import EnhancedTestDataGenerator
from src.testing.test_data_generator import Score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/create_gold_standard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("create_gold_standard")

def generate_base_test_data(count: int = 100) -> List[Dict[str, Any]]:
    """Generate base test data using the enhanced generator."""
    logger.info(f"Generating {count} base test samples...")
    generator = EnhancedTestDataGenerator()
    data = generator.generate_enhanced_test_data(count)
    logger.info(f"Generated {len(data)} base test samples")
    return data

def add_edge_cases() -> List[Dict[str, Any]]:
    """Add manually curated edge cases to challenge the scoring system."""
    logger.info("Adding manually curated edge cases...")
    
    edge_cases = [
        # Ambiguous between EMERGING and WITH_SUPPORT
        {
            "response": "He's starting to take a few steps while holding onto furniture, but still needs support.",
            "milestone_context": {
                "id": "motor_01",
                "domain": "motor",
                "behavior": "Walks independently",
                "criteria": "Child walks without support for at least 10 steps",
                "age_range": "12-18 months"
            },
            "expected_score": "WITH_SUPPORT",
            "expected_score_value": 3,
            "notes": "This is ambiguous because it mentions both 'starting to' (suggesting EMERGING) and 'needs support' (suggesting WITH_SUPPORT)."
        },
        
        # Ambiguous between LOST_SKILL and CANNOT_DO
        {
            "response": "She never really mastered this skill fully. She tried a few times months ago but hasn't done it since.",
            "milestone_context": {
                "id": "cognitive_05",
                "domain": "cognitive",
                "behavior": "Identifies basic colors",
                "criteria": "Child can correctly name at least 3 colors",
                "age_range": "30-36 months"
            },
            "expected_score": "LOST_SKILL",
            "expected_score_value": 1,
            "notes": "This is ambiguous because it's unclear if the child truly acquired the skill before losing it."
        },
        
        # Contradictory information
        {
            "response": "Yes, he can do this independently. I always have to help him though.",
            "milestone_context": {
                "id": "social_03",
                "domain": "social",
                "behavior": "Takes turns in simple games",
                "criteria": "Child waits for their turn in simple games",
                "age_range": "24-30 months"
            },
            "expected_score": "WITH_SUPPORT",
            "expected_score_value": 3,
            "notes": "This contains contradictory information - claims independence but then mentions always needing help."
        },
        
        # Irrelevant information
        {
            "response": "We've been working on potty training lately. He's been doing really well with that. Oh, about stacking blocks? No, he can't do that yet.",
            "milestone_context": {
                "id": "motor_02",
                "domain": "motor",
                "behavior": "Stacks blocks",
                "criteria": "Child can stack at least 3 blocks",
                "age_range": "12-18 months"
            },
            "expected_score": "CANNOT_DO",
            "expected_score_value": 0,
            "notes": "This contains irrelevant information before addressing the actual milestone."
        },
        
        # Multilingual response
        {
            "response": "Sí, ella puede hacerlo muy bien. She does this independently all the time.",
            "milestone_context": {
                "id": "communication_04",
                "domain": "communication",
                "behavior": "Points to named objects",
                "criteria": "Child points to at least 5 objects when named",
                "age_range": "12-18 months"
            },
            "expected_score": "INDEPENDENT",
            "expected_score_value": 4,
            "notes": "This contains text in Spanish followed by English, but clearly indicates independence."
        },
        
        # Complex developmental history
        {
            "response": "He used to do this well until about 3 months ago when he had that ear infection. Then he stopped completely for a while. Now he's starting to do it again occasionally, but not consistently.",
            "milestone_context": {
                "id": "communication_02",
                "domain": "communication",
                "behavior": "Responds to their name",
                "criteria": "Child looks or turns when their name is called",
                "age_range": "6-12 months"
            },
            "expected_score": "EMERGING",
            "expected_score_value": 2,
            "notes": "This describes a complex history: had skill, lost it, now re-emerging."
        },
        
        # Conditional performance
        {
            "response": "It depends on who's asking. With me, she always responds. With strangers or even her dad, she often ignores them completely.",
            "milestone_context": {
                "id": "social_04",
                "domain": "social",
                "behavior": "Recognizes familiar people",
                "criteria": "Child shows recognition of familiar caregivers",
                "age_range": "6-12 months"
            },
            "expected_score": "WITH_SUPPORT",
            "expected_score_value": 3,
            "notes": "This describes conditional performance based on the person involved."
        },
        
        # Spelling and grammar issues
        {
            "response": "yeh he do this alot. independant. no help needed frum me or dad.",
            "milestone_context": {
                "id": "cognitive_03",
                "domain": "cognitive",
                "behavior": "Completes simple puzzles",
                "criteria": "Child can complete simple 3-4 piece puzzles",
                "age_range": "24-30 months"
            },
            "expected_score": "INDEPENDENT",
            "expected_score_value": 4,
            "notes": "This has spelling and grammar issues but clearly indicates independence."
        },
        
        # Very detailed response
        {
            "response": "I've been tracking this carefully. On Monday (May 3rd), she stacked 2 blocks for the first time. Then on Wednesday, she managed 3 blocks but they fell immediately. Yesterday, she stacked 4 blocks and they stayed up for about 10 seconds before she knocked them down deliberately. She seems to be improving rapidly with this skill.",
            "milestone_context": {
                "id": "motor_02",
                "domain": "motor",
                "behavior": "Stacks blocks",
                "criteria": "Child can stack at least 3 blocks",
                "age_range": "12-18 months"
            },
            "expected_score": "INDEPENDENT",
            "expected_score_value": 4,
            "notes": "This provides very detailed information about the child's progress."
        },
        
        # Extremely short response
        {
            "response": "No.",
            "milestone_context": {
                "id": "cognitive_01",
                "domain": "cognitive",
                "behavior": "Sorts objects by color or shape",
                "criteria": "Child can sort objects into at least 2 categories",
                "age_range": "24-30 months"
            },
            "expected_score": "CANNOT_DO",
            "expected_score_value": 0,
            "notes": "This is an extremely short response but clearly indicates the child cannot do the task."
        }
    ]
    
    logger.info(f"Added {len(edge_cases)} manually curated edge cases")
    return edge_cases

def add_domain_specific_examples() -> List[Dict[str, Any]]:
    """Add domain-specific examples for each developmental domain."""
    logger.info("Adding domain-specific examples...")
    
    examples = [
        # Motor domain - fine motor skills
        {
            "response": "She can hold a crayon and make marks on paper, but can't draw recognizable shapes yet.",
            "milestone_context": {
                "id": "motor_05",
                "domain": "motor",
                "behavior": "Draws simple shapes",
                "criteria": "Child can draw at least one recognizable shape",
                "age_range": "24-36 months"
            },
            "expected_score": "EMERGING",
            "expected_score_value": 2
        },
        
        # Motor domain - gross motor skills
        {
            "response": "He can climb up the stairs if I hold his hand, but needs to be carried down.",
            "milestone_context": {
                "id": "motor_03",
                "domain": "motor",
                "behavior": "Climbs stairs with support",
                "criteria": "Child can climb stairs with hand support",
                "age_range": "18-24 months"
            },
            "expected_score": "WITH_SUPPORT",
            "expected_score_value": 3
        },
        
        # Communication domain - receptive language
        {
            "response": "She understands when I ask her to get her shoes or her teddy bear. She can follow simple instructions like that.",
            "milestone_context": {
                "id": "communication_05",
                "domain": "communication",
                "behavior": "Follows two-step instructions",
                "criteria": "Child can follow instructions with two actions",
                "age_range": "24-30 months"
            },
            "expected_score": "INDEPENDENT",
            "expected_score_value": 4
        },
        
        # Communication domain - expressive language
        {
            "response": "He used to say 'mama' and 'dada' but hasn't said any words in the last month.",
            "milestone_context": {
                "id": "communication_01",
                "domain": "communication",
                "behavior": "Uses two-word sentences",
                "criteria": "Child combines two words to express ideas",
                "age_range": "18-24 months"
            },
            "expected_score": "LOST_SKILL",
            "expected_score_value": 1
        },
        
        # Social domain - peer interaction
        {
            "response": "She watches other children play and sometimes sits near them, but doesn't interact with them yet.",
            "milestone_context": {
                "id": "social_01",
                "domain": "social",
                "behavior": "Plays alongside other children",
                "criteria": "Child engages in parallel play near peers",
                "age_range": "18-24 months"
            },
            "expected_score": "EMERGING",
            "expected_score_value": 2
        },
        
        # Social domain - emotional development
        {
            "response": "When his friend fell down at playgroup, he went over and patted him on the back. He even brought him his favorite toy.",
            "milestone_context": {
                "id": "social_03",
                "domain": "social",
                "behavior": "Shows empathy when others are upset",
                "criteria": "Child responds to others' emotions with caring behavior",
                "age_range": "24-36 months"
            },
            "expected_score": "INDEPENDENT",
            "expected_score_value": 4
        },
        
        # Cognitive domain - problem solving
        {
            "response": "If his toy rolls under the couch, he'll try to reach it. If he can't reach it, he'll come get me to help.",
            "milestone_context": {
                "id": "cognitive_02",
                "domain": "cognitive",
                "behavior": "Understands object permanence",
                "criteria": "Child looks for hidden objects",
                "age_range": "8-12 months"
            },
            "expected_score": "INDEPENDENT",
            "expected_score_value": 4
        },
        
        # Cognitive domain - academic skills
        {
            "response": "She can count '1, 2, 3' if I start counting with her, but she skips numbers or gets confused if trying on her own.",
            "milestone_context": {
                "id": "cognitive_04",
                "domain": "cognitive",
                "behavior": "Counts to five",
                "criteria": "Child can count to five in correct sequence",
                "age_range": "30-36 months"
            },
            "expected_score": "WITH_SUPPORT",
            "expected_score_value": 3
        }
    ]
    
    logger.info(f"Added {len(examples)} domain-specific examples")
    return examples

def create_gold_standard_dataset():
    """Create a comprehensive gold standard dataset."""
    # Generate base test data
    data = generate_base_test_data(100)
    
    # Add edge cases
    edge_cases = add_edge_cases()
    data.extend(edge_cases)
    
    # Add domain-specific examples
    domain_examples = add_domain_specific_examples()
    data.extend(domain_examples)
    
    # Create metadata
    metadata = {
        "description": "Initial gold standard dataset for developmental milestone scoring",
        "creation_date": datetime.now().isoformat(),
        "version": "1.0.0",
        "source": "mixed",
        "sample_count": len(data),
        "composition": {
            "generated": 100,
            "edge_cases": len(edge_cases),
            "domain_examples": len(domain_examples)
        },
        "creator": "Developmental Milestone Scoring System Team"
    }
    
    # Save to gold standard manager
    manager = GoldStandardManager()
    version = manager.save_dataset(data, "1.0.0", metadata)
    
    logger.info(f"Created gold standard dataset version {version} with {len(data)} samples")
    print(f"Created gold standard dataset version {version} with {len(data)} samples")
    
    # Validate the dataset
    valid, issues = manager.validate_dataset(data)
    if valid:
        logger.info("Dataset validation passed")
        print("Dataset validation passed")
    else:
        logger.warning("Dataset validation found issues:")
        print("Dataset validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
            print(f"  - {issue}")
    
    # Analyze the dataset
    analysis = manager.analyze_dataset(data, version)
    
    # Print score distribution
    print("\nScore Distribution:")
    for score, count in analysis["score_distribution"].items():
        percentage = (count / len(data)) * 100
        print(f"  {score}: {count} ({percentage:.1f}%)")
    
    # Print domain distribution
    print("\nDomain Distribution:")
    for domain, count in analysis["domain_distribution"].items():
        percentage = (count / len(data)) * 100
        print(f"  {domain}: {count} ({percentage:.1f}%)")
    
    return version

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Create the gold standard dataset
    create_gold_standard_dataset() 