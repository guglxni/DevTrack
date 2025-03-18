"""
Age-Specific Knowledge for Developmental Assessment

This module extends domain-specific knowledge with age-specific considerations
for more accurate developmental assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Define age brackets for developmental assessment
AGE_BRACKETS = {
    "infant": (0, 12),       # 0-12 months
    "toddler": (13, 24),     # 13-24 months
    "preschooler": (25, 60)  # 25-60 months
}

@dataclass
class AgeSpecificExpectations:
    """Age-specific expectations for developmental milestones"""
    
    age_range: Tuple[int, int]  # (min_months, max_months)
    description: str 
    expected_skills: Dict[str, List[str]]  # Domain -> List of expected skills
    variation_notes: str
    assessment_considerations: List[str]
    common_misconceptions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "age_range": self.age_range,
            "age_range_str": f"{self.age_range[0]}-{self.age_range[1]} months",
            "description": self.description,
            "expected_skills": self.expected_skills,
            "variation_notes": self.variation_notes,
            "assessment_considerations": self.assessment_considerations,
            "common_misconceptions": self.common_misconceptions
        }

@dataclass
class AgeCategoryGuidance:
    """Age-specific guidance for scoring categories"""
    
    category: str
    age_range: Tuple[int, int]
    description: str
    typical_indicators: List[str]
    confidence_adjustment: float  # Adjustment to base confidence for this age
    boundary_considerations: Dict[str, str]
    domain_specific_notes: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "category": self.category,
            "age_range": self.age_range,
            "age_range_str": f"{self.age_range[0]}-{self.age_range[1]} months",
            "description": self.description,
            "typical_indicators": self.typical_indicators,
            "confidence_adjustment": self.confidence_adjustment,
            "boundary_considerations": self.boundary_considerations,
            "domain_specific_notes": self.domain_specific_notes
        }

class AgeKnowledgeRepository:
    """Repository of age-specific developmental knowledge"""
    
    def __init__(self):
        """Initialize the age knowledge repository"""
        self._age_expectations = self._initialize_age_expectations()
        self._category_guidance = self._initialize_category_guidance()
        logger.info(f"Initialized age-specific knowledge for {len(self._age_expectations)} age brackets")
    
    def get_age_bracket(self, age_months: int) -> str:
        """Get the appropriate age bracket label for a given age"""
        for bracket, (min_age, max_age) in AGE_BRACKETS.items():
            if min_age <= age_months <= max_age:
                return bracket
        return "unknown"
    
    def get_age_expectations(self, age_months: int) -> Optional[AgeSpecificExpectations]:
        """Get expectations for a specific age"""
        bracket = self.get_age_bracket(age_months)
        return self._age_expectations.get(bracket)
    
    def get_category_guidance(self, category: str, age_bracket_or_months: Union[str, int]) -> Optional[AgeCategoryGuidance]:
        """Get guidance for a category at a specific age
        
        Args:
            category: The category name
            age_bracket_or_months: Either an age bracket string (e.g., "infant") or age in months
            
        Returns:
            Guidance for the category at the specified age
        """
        # If age_bracket_or_months is an integer, convert to bracket
        if isinstance(age_bracket_or_months, int):
            age_bracket = self.get_age_bracket(age_bracket_or_months)
        else:
            age_bracket = age_bracket_or_months
        
        return self._category_guidance.get((category, age_bracket))
    
    def get_confidence_adjustment(self, category: str, age_months: int) -> float:
        """Get confidence adjustment for a category at a specific age"""
        guidance = self.get_category_guidance(category, age_months)
        if guidance:
            return guidance.confidence_adjustment
        return 0.0
    
    def get_domain_skills(self, domain: str, age_months: int) -> List[str]:
        """Get list of expected skills for a domain at a specific age"""
        domain = domain.lower()
        expectations = self.get_age_expectations(age_months)
        if expectations and domain in expectations.expected_skills:
            return expectations.expected_skills[domain]
        return []
    
    def _initialize_age_expectations(self) -> Dict[str, AgeSpecificExpectations]:
        """Initialize age-specific developmental expectations"""
        expectations = {}
        
        # Infant (0-12 months)
        expectations["infant"] = AgeSpecificExpectations(
            age_range=(0, 12),
            description="Rapid developmental period with focus on foundational skills",
            expected_skills={
                "motor": [
                    "Head control (1-4 months)",
                    "Rolling over (4-6 months)",
                    "Sitting without support (6-8 months)",
                    "Crawling (8-10 months)",
                    "Pulling to stand (9-12 months)"
                ],
                "communication": [
                    "Cooing (1-4 months)",
                    "Babbling (4-8 months)",
                    "Gesturing (8-12 months)",
                    "First words (10-12 months)"
                ],
                "social": [
                    "Social smile (1-3 months)",
                    "Stranger anxiety (7-10 months)",
                    "Social games like peek-a-boo (8-12 months)"
                ],
                "cognitive": [
                    "Visual tracking (1-3 months)",
                    "Object permanence beginning (4-8 months)",
                    "Simple problem solving (8-12 months)",
                    "Imitation of actions (9-12 months)"
                ]
            },
            variation_notes="Wide range of normal variation in timing but sequence is relatively consistent",
            assessment_considerations=[
                "Assessment must account for premature birth",
                "Physical positioning significantly impacts skill demonstration",
                "Brief observation windows may miss emerging skills",
                "Parent report essential for comprehensive picture"
            ],
            common_misconceptions=[
                "Early milestones always predict later development",
                "All skills develop at the same rate",
                "Missing a milestone indicates a problem",
                "Early walking correlates with higher intelligence"
            ]
        )
        
        # Toddler (13-24 months)
        expectations["toddler"] = AgeSpecificExpectations(
            age_range=(13, 24),
            description="Period of rapidly increasing independence and language development",
            expected_skills={
                "motor": [
                    "Walking independently (12-15 months)",
                    "Walking backward (15-18 months)",
                    "Running (18-24 months)",
                    "Kicking a ball (18-24 months)",
                    "Climbing on furniture (18-24 months)"
                ],
                "communication": [
                    "First words (12-18 months)",
                    "Two-word phrases (18-24 months)",
                    "Following simple directions (12-18 months)",
                    "Identifying body parts (18-24 months)",
                    "50+ word vocabulary (21-24 months)"
                ],
                "social": [
                    "Parallel play (12-18 months)",
                    "Beginning pretend play (18-24 months)",
                    "Increased independence (12-24 months)",
                    "Emerging self-awareness (18-24 months)"
                ],
                "cognitive": [
                    "Object permanence established (12-18 months)",
                    "Simple categorization (18-24 months)",
                    "Basic problem solving (12-24 months)",
                    "Understanding cause-effect (18-24 months)"
                ]
            },
            variation_notes="Significant variation in language acquisition timing; motor sequence more consistent",
            assessment_considerations=[
                "Separation anxiety may affect performance",
                "Toddler negativism may interfere with compliance",
                "Attention span still limited to 2-5 minutes",
                "Skill demonstration varies greatly with context and motivation"
            ],
            common_misconceptions=[
                "Language delay always indicates a disorder",
                "Terrible twos behavior is abnormal",
                "Uneven development across domains is problematic",
                "Lack of sharing indicates social problems"
            ]
        )
        
        # Preschooler (25-60 months)
        expectations["preschooler"] = AgeSpecificExpectations(
            age_range=(25, 60),
            description="Period of increasing complexity in language, social skills, and cognitive abilities",
            expected_skills={
                "motor": [
                    "Jumping in place (24-30 months)",
                    "Standing on one foot (30-36 months)",
                    "Pedaling tricycle (30-36 months)",
                    "Drawing circles (30-36 months)",
                    "Using scissors (30-36 months)"
                ],
                "communication": [
                    "Three-word sentences (24-30 months)",
                    "250+ word vocabulary (30-36 months)",
                    "Using pronouns (24-30 months)",
                    "Asking questions (24-36 months)",
                    "Conversation with turns (30-36 months)"
                ],
                "social": [
                    "Simple cooperative play (24-30 months)",
                    "Emerging empathy (24-36 months)",
                    "Engaging in pretend play (24-36 months)",
                    "Following simple rules (30-36 months)"
                ],
                "cognitive": [
                    "Matching and sorting (24-30 months)",
                    "Understanding number concepts to 3 (30-36 months)",
                    "Complex pretend play sequences (30-36 months)",
                    "Completing simple puzzles (24-36 months)"
                ]
            },
            variation_notes="Wider range of acceptable variation in complex skills; environmental factors more influential",
            assessment_considerations=[
                "Language can be assessed more directly",
                "Child can participate in structured tasks",
                "Contextual factors significantly impact performance",
                "Direct observation more feasible and informative"
            ],
            common_misconceptions=[
                "All children should speak clearly by 36 months",
                "Parallel play indicates social problems",
                "Inability to share indicates atypical development",
                "Motor skills develop evenly across all activities"
            ]
        )
        
        return expectations
    
    def _initialize_category_guidance(self) -> Dict[Tuple[str, str], AgeCategoryGuidance]:
        """Initialize age-specific category guidance"""
        guidance = {}
        
        # CANNOT_DO category guidance
        guidance[("CANNOT_DO", "infant")] = AgeCategoryGuidance(
            category="CANNOT_DO",
            age_range=(0, 12),
            description="No evidence of the skill despite appropriate opportunities in infants",
            typical_indicators=[
                "Complete absence of precursor movements or attempts",
                "No response to facilitation or demonstration",
                "Caregiver reports no instances of skill or attempts"
            ],
            confidence_adjustment=-0.10,  # Lower confidence due to rapid development
            boundary_considerations={
                "WITH_SUPPORT": "Infants often show subtle signs of emerging skills; careful observation needed"
            },
            domain_specific_notes={
                "motor": "Focus on precursor movements and postural control",
                "communication": "Look for any preverbal communication attempts",
                "social": "Consider temperamental differences in social engagement",
                "cognitive": "Assess interest and attention to environmental stimuli"
            }
        )
        
        guidance[("CANNOT_DO", "toddler")] = AgeCategoryGuidance(
            category="CANNOT_DO",
            age_range=(13, 24),
            description="No evidence of the skill despite appropriate opportunities in toddlers",
            typical_indicators=[
                "No attempts at the skill even with modeling",
                "No interest in activities that would elicit the skill",
                "Caregiver reports consistent absence across contexts"
            ],
            confidence_adjustment=0.0,  # Standard confidence
            boundary_considerations={
                "WITH_SUPPORT": "Toddlers may show resistance rather than inability; motivation is key"
            },
            domain_specific_notes={
                "motor": "Distinguish between skill absence and refusal",
                "communication": "Check comprehension even when expression is limited",
                "social": "Account for stranger anxiety and separation concerns",
                "cognitive": "Assess through play-based activities rather than direct instruction"
            }
        )
        
        guidance[("CANNOT_DO", "preschooler")] = AgeCategoryGuidance(
            category="CANNOT_DO",
            age_range=(25, 60),
            description="No evidence of the skill despite appropriate opportunities in preschoolers",
            typical_indicators=[
                "Verbal refusal or statement of inability",
                "No demonstration despite clear understanding of task",
                "Consistent inability across multiple attempts and contexts"
            ],
            confidence_adjustment=0.05,  # Higher confidence due to more reliable assessment
            boundary_considerations={
                "WITH_SUPPORT": "Preschoolers can often verbalize their difficulties; incorporate their perspective"
            },
            domain_specific_notes={
                "motor": "Rule out lack of experience versus actual inability",
                "communication": "Distinguish between articulation issues and true absence",
                "social": "Consider context - home vs. group settings may differ",
                "cognitive": "Assess through multiple modalities before concluding inability"
            }
        )
        
        # WITH_SUPPORT category guidance
        guidance[("WITH_SUPPORT", "infant")] = AgeCategoryGuidance(
            category="WITH_SUPPORT",
            age_range=(0, 12),
            description="Skill demonstrated only with physical assistance or environmental modification",
            typical_indicators=[
                "Completes movement when physically guided",
                "Responds to hand-over-hand assistance",
                "Success only in highly supportive positions"
            ],
            confidence_adjustment=-0.05,  # Slightly lower confidence
            boundary_considerations={
                "EMERGING": "The line between support and emerging is less clear in infants, often confused with EMERGING",
                "CANNOT_DO": "Infants may appear unable when actually needing specific support, often confused with CANNOT_DO"
            },
            domain_specific_notes={
                "motor": "Physical support is the primary form of assistance. If showing partial movements without support, consider as EMERGING.",
                "communication": "Environmental cues and prompts constitute support. If responding inconsistently to prompts, consider as EMERGING.",
                "social": "Adult-initiated interaction may mask true social abilities. If showing any spontaneous social initiation, consider as EMERGING.",
                "cognitive": "Simplified environment often needed for skill demonstration. If showing any spontaneous problem-solving, consider as EMERGING."
            }
        )
        
        guidance[("WITH_SUPPORT", "toddler")] = AgeCategoryGuidance(
            category="WITH_SUPPORT",
            age_range=(13, 24),
            description="Skill demonstrated with verbal prompting, modeling, or physical guidance",
            typical_indicators=[
                "Successful with step-by-step verbal direction",
                "Requires demonstration immediately before attempt",
                "Performs with partial physical assistance"
            ],
            confidence_adjustment=0.0,  # Standard confidence
            boundary_considerations={
                "EMERGING": "Look for any successful unprompted attempts as evidence of emergence"
            },
            domain_specific_notes={
                "motor": "Verbal cues becoming as important as physical support",
                "communication": "Imitation and echolalia common forms of supported language",
                "social": "Adult mediation often needed for peer interactions",
                "cognitive": "Visual cues and modeling key forms of support"
            }
        )
        
        guidance[("WITH_SUPPORT", "preschooler")] = AgeCategoryGuidance(
            category="WITH_SUPPORT",
            age_range=(25, 60),
            description="Requires specific support to demonstrate age-appropriate skill",
            typical_indicators=[
                "Needs verbal prompts or reminders",
                "Performs with partial physical guidance",
                "Requires environmental modifications",
                "Succeeds in simplified or structured contexts"
            ],
            confidence_adjustment=-0.1,  # Lower confidence for preschoolers needing support
            boundary_considerations={
                "CANNOT_DO": "Consider increasing support before determining inability"
            },
            domain_specific_notes={
                "motor": "Environmental modifications may mask true support needs",
                "communication": "Ensure distinctions between comprehension and performance issues",
                "social": "Social opportunities may be limited, affecting assessment",
                "cognitive": "Attention issues often confused with skill deficits"
            }
        )
        
        # EMERGING category guidance
        guidance[("EMERGING", "infant")] = AgeCategoryGuidance(
            category="EMERGING",
            age_range=(0, 12),
            description="Inconsistent or partial skill performance without direct assistance",
            typical_indicators=[
                "Occasional success but inconsistent",
                "Partial completion of the expected movement",
                "Performance varies greatly between attempts"
            ],
            confidence_adjustment=-0.10,  # Lower confidence due to rapid changes
            boundary_considerations={
                "INDEPENDENT": "Even single instances of clear success may indicate readiness for independence, often confused with INDEPENDENT",
                "WITH_SUPPORT": "Difficult to distinguish between emerging skills and those requiring support in infants"
            },
            domain_specific_notes={
                "motor": "Look for quality of movement in addition to achievement. If movement is fluid, consider as INDEPENDENT.",
                "communication": "Inconsistent use of sounds or gestures typical in emergence. If consistently responding to verbal cues, consider as WITH_SUPPORT.",
                "social": "Fluctuating social interest normal in infancy. If consistently engaging with familiar caregivers, consider as INDEPENDENT.",
                "cognitive": "Brief demonstrations of understanding sufficient. If requiring significant prompting, consider as WITH_SUPPORT."
            }
        )
        
        guidance[("EMERGING", "toddler")] = AgeCategoryGuidance(
            category="EMERGING",
            age_range=(13, 24),
            description="Inconsistent skill performance with increasing frequency of success",
            typical_indicators=[
                "Successful in some contexts but not others",
                "Requires multiple attempts before success",
                "Performance improving over short time periods"
            ],
            confidence_adjustment=-0.05,  # Slightly lower confidence
            boundary_considerations={
                "INDEPENDENT": "Toddlers may appear independent in familiar contexts only, often confused with INDEPENDENT",
                "WITH_SUPPORT": "Distinguish between inconsistent performance and need for support, often confused with WITH_SUPPORT"
            },
            domain_specific_notes={
                "motor": "Look for increasing coordination and intentionality. If movements are coordinated and purposeful, consider as INDEPENDENT.",
                "communication": "Vocabulary explosion common; inconsistency expected. If using words consistently in appropriate contexts, consider as INDEPENDENT.",
                "social": "Parallel play transitions to interactive play. If consistently engaging in interactive play, consider as INDEPENDENT.",
                "cognitive": "Trial and error problem-solving typical. If requiring adult modeling, consider as WITH_SUPPORT."
            }
        )
        
        guidance[("EMERGING", "preschooler")] = AgeCategoryGuidance(
            category="EMERGING",
            age_range=(25, 60),
            description="Increasingly consistent performance with occasional lapses",
            typical_indicators=[
                "Success in familiar contexts but struggles in new situations",
                "Can verbalize understanding but execution is inconsistent",
                "Performance improves noticeably with practice"
            ],
            confidence_adjustment=0.05,  # Higher confidence
            boundary_considerations={
                "INDEPENDENT": "Should see clear progress toward generalization and automaticity, often confused with INDEPENDENT",
                "WITH_SUPPORT": "Distinguish between inconsistent performance and need for support, often confused with WITH_SUPPORT"
            },
            domain_specific_notes={
                "motor": "Motor planning rather than execution often the limiting factor. If consistently planning and executing movements, consider as INDEPENDENT.",
                "communication": "Grammatical inconsistencies typical during emergence. If requiring modeling to use correct grammar, consider as WITH_SUPPORT.",
                "social": "Group size can impact emerging social skill performance. If consistently successful in small groups, consider as INDEPENDENT.",
                "cognitive": "Abstract thinking emerging but concrete examples still needed. If requiring concrete examples for all new concepts, consider as WITH_SUPPORT."
            }
        )
        
        # INDEPENDENT category guidance
        guidance[("INDEPENDENT", "infant")] = AgeCategoryGuidance(
            category="INDEPENDENT",
            age_range=(0, 12),
            description="Consistent skill performance without assistance across contexts",
            typical_indicators=[
                "Repeats skill spontaneously",
                "Performs in different environments",
                "Initiates the skill without prompting"
            ],
            confidence_adjustment=0.10,  # Higher confidence
            boundary_considerations={
                "EMERGING": "Infants may appear independent in highly familiar contexts only, often confused with EMERGING"
            },
            domain_specific_notes={
                "motor": "Look for spontaneous repetition of movements. If only performed once or twice, consider as EMERGING.",
                "communication": "Consistent use of sounds/gestures for communication. If inconsistent or only with familiar people, consider as EMERGING.",
                "social": "Consistent social engagement with familiar caregivers. If requiring significant encouragement, consider as WITH_SUPPORT.",
                "cognitive": "Consistent demonstration of understanding. If requiring significant environmental setup, consider as WITH_SUPPORT."
            }
        )
        
        guidance[("INDEPENDENT", "toddler")] = AgeCategoryGuidance(
            category="INDEPENDENT",
            age_range=(13, 24),
            description="Consistent skill performance without assistance in familiar contexts",
            typical_indicators=[
                "Initiates skill spontaneously",
                "Performs consistently across familiar settings",
                "Adapts skill to different situations"
            ],
            confidence_adjustment=0.10,  # Higher confidence
            boundary_considerations={
                "EMERGING": "Toddlers may appear independent in preferred activities only, often confused with EMERGING"
            },
            domain_specific_notes={
                "motor": "Look for generalization across environments. If only performed in one setting, consider as EMERGING.",
                "communication": "Consistent word use in appropriate contexts. If vocabulary is limited to specific contexts, consider as EMERGING.",
                "social": "Initiates social interactions with peers and adults. If only engaging with familiar adults, consider as EMERGING.",
                "cognitive": "Applies concepts across different situations. If requiring reminders or prompts, consider as WITH_SUPPORT."
            }
        )
        
        guidance[("INDEPENDENT", "preschooler")] = AgeCategoryGuidance(
            category="INDEPENDENT",
            age_range=(25, 60),
            description="Consistent skill performance across varied contexts without assistance",
            typical_indicators=[
                "Generalizes skill across environments and materials",
                "Adapts skill appropriately to new situations",
                "Teaches or explains skill to others"
            ],
            confidence_adjustment=0.15,  # Higher confidence
            boundary_considerations={
                "EMERGING": "Preschoolers may verbalize understanding but still be inconsistent in performance, often confused with EMERGING"
            },
            domain_specific_notes={
                "motor": "Look for quality, efficiency and adaptability. If requiring setup or specific conditions, consider as EMERGING.",
                "communication": "Uses complex language appropriately across contexts. If grammatical errors persist, consider as EMERGING.",
                "social": "Navigates group dynamics and resolves conflicts. If requiring adult mediation for conflicts, consider as WITH_SUPPORT.",
                "cognitive": "Applies abstract concepts and generalizes learning. If requiring concrete examples, consider as EMERGING."
            }
        )
        
        # LOST_SKILL category guidance
        guidance[("LOST_SKILL", "infant")] = AgeCategoryGuidance(
            category="LOST_SKILL",
            age_range=(0, 12),
            description="Previously consistent skill no longer demonstrated in infants",
            typical_indicators=[
                "Caregiver reports clear previous ability now absent",
                "Documentation of previous demonstration",
                "Complete absence of previously observed skill"
            ],
            confidence_adjustment=-0.15,  # Much lower confidence due to rapid development
            boundary_considerations={
                "EMERGING": "Apparent loss may be transitional as more advanced forms develop"
            },
            domain_specific_notes={
                "motor": "Motor transitions often involve temporary skill regression",
                "communication": "Babbling may decrease as first words emerge",
                "social": "Social engagement may fluctuate with development of attachment",
                "cognitive": "Attention shifts may appear as lost skills"
            }
        )
        
        guidance[("LOST_SKILL", "toddler")] = AgeCategoryGuidance(
            category="LOST_SKILL",
            age_range=(13, 24),
            description="Previously mastered skill shows significant decline or absence",
            typical_indicators=[
                "Consistent inability to perform previously mastered skill",
                "Multiple caregiver reports of skill loss",
                "Frustration when attempting previously easy tasks"
            ],
            confidence_adjustment=-0.10,  # Lower confidence
            boundary_considerations={
                "EMERGING": "Some skills appear cyclical in toddlers as focus shifts"
            },
            domain_specific_notes={
                "motor": "Motor regression less common except during major transitions",
                "communication": "Word loss significant if more than 25% of vocabulary",
                "social": "Changes in social interest require careful evaluation",
                "cognitive": "True cognitive skill loss needs thorough investigation"
            }
        )
        
        guidance[("LOST_SKILL", "preschooler")] = AgeCategoryGuidance(
            category="LOST_SKILL",
            age_range=(25, 60),
            description="Clear regression in previously consistent skill performance",
            typical_indicators=[
                "Inability to perform skills documented in previous assessments",
                "Child may verbalize inability or frustration",
                "Consistent across multiple contexts and observers"
            ],
            confidence_adjustment=-0.05,  # Slightly lower confidence
            boundary_considerations={
                "EMERGING": "Skill use may become more selective rather than truly lost"
            },
            domain_specific_notes={
                "motor": "Motor skill loss less expected and more significant at this age",
                "communication": "Language regression warrants immediate attention",
                "social": "Loss of social skills more likely indicates meaningful change",
                "cognitive": "Cognitive skill loss requires comprehensive evaluation"
            }
        )
        
        return guidance

# Singleton instance
_repository = None

def get_age_repository() -> AgeKnowledgeRepository:
    """Get the age knowledge repository singleton"""
    global _repository
    if _repository is None:
        _repository = AgeKnowledgeRepository()
    return _repository

def get_age_expectations(age_months: int) -> Optional[Dict[str, Any]]:
    """Get age-specific expectations for a given age"""
    expectations = get_age_repository().get_age_expectations(age_months)
    if expectations:
        return expectations.to_dict()
    return None

def get_category_guidance(category: str, age_months: int) -> Optional[Dict[str, Any]]:
    """Get age-specific guidance for a scoring category"""
    guidance = get_age_repository().get_category_guidance(category, age_months)
    if guidance:
        return guidance.to_dict()
    return None

def get_expected_skills(domain: str, age_months: int) -> List[str]:
    """Get expected skills for a domain at a specific age"""
    return get_age_repository().get_domain_skills(domain, age_months)

def get_confidence_adjustment(category: str, age_months: int) -> float:
    """Get age-specific confidence adjustment for a category"""
    return get_age_repository().get_confidence_adjustment(category, age_months)

def get_age_bracket(age_months: int) -> str:
    """Get the age bracket name for a given age"""
    return get_age_repository().get_age_bracket(age_months)

def adjust_category_for_age(category: str, confidence: float, age_months: Union[int, str], domain: str = None) -> Tuple[str, float]:
    """
    Adjust a developmental category based on age-specific expectations
    
    Args:
        category: The original category
        confidence: The confidence level (0-1)
        age_months: The age in months or age bracket name
        domain: Optional domain for more specific adjustments
        
    Returns:
        Tuple of (adjusted_category, adjusted_confidence)
    """
    # Ensure age_months is an integer if it's not a string (age bracket)
    if not isinstance(age_months, (int, str)):
        try:
            age_months = int(age_months)
        except (ValueError, TypeError):
            # Default to 24 months if conversion fails
            age_months = 24  
    
    # Get the bracket if we have age in months
    if isinstance(age_months, int):
        age_bracket = get_age_bracket(age_months)
    else:
        # Assume it's already a bracket name
        age_bracket = age_months
    
    # Get the guidance directly from the repository to get the object, not dict
    repo = get_age_repository()
    guidance = repo.get_category_guidance(category, age_bracket)
    
    # Apply confidence adjustment based on age bracket guidance
    adj_confidence = confidence
    adj_category = category
    
    if guidance:
        # Adjust confidence based on guidance
        adj_confidence += guidance.confidence_adjustment
        
        # Check if we should potentially change the category based on domain-specific notes
        if domain and domain.lower() in guidance.domain_specific_notes:
            domain_note = guidance.domain_specific_notes[domain.lower()]
            
            # Check for category change indicators in the domain notes
            if "consider as EMERGING" in domain_note and confidence < 0.7:
                adj_category = "EMERGING"
                adj_confidence = max(adj_confidence, 0.7)  # Boost confidence for the new category
            elif "consider as WITH_SUPPORT" in domain_note and confidence < 0.7:
                adj_category = "WITH_SUPPORT"
                adj_confidence = max(adj_confidence, 0.7)
            elif "consider as INDEPENDENT" in domain_note and confidence > 0.8:
                adj_category = "INDEPENDENT"
                adj_confidence = max(adj_confidence, 0.85)
            
            # Check boundary considerations for potential category changes
            if category in guidance.boundary_considerations:
                boundary_note = guidance.boundary_considerations[category]
                
                # If confidence is near a boundary, consider changing the category
                if "often confused with" in boundary_note.lower():
                    # Extract the potentially confused category
                    for potential_category in ["EMERGING", "WITH_SUPPORT", "INDEPENDENT", "CANNOT_DO", "LOST_SKILL"]:
                        if potential_category in boundary_note and potential_category != category:
                            # If confidence is low, consider changing to the boundary category
                            if confidence < 0.75:
                                adj_category = potential_category
                                adj_confidence = max(adj_confidence, 0.75)  # Boost confidence for the new category
    
    # Make sure confidence stays in valid range
    adj_confidence = max(0.0, min(1.0, adj_confidence))
    
    # Return adjusted category and confidence
    return adj_category, adj_confidence 