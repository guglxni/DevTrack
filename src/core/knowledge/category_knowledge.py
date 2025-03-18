"""
Evidence-Based Category Knowledge Module

This module provides a structured repository of research-backed category distinctions
for developmental milestone assessment. It centralizes the knowledge derived from
established assessment frameworks and developmental psychology literature.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class CategoryEvidence:
    """Evidence-based definition for a development category"""
    
    name: str
    description: str
    research_based_indicators: List[str]
    framework_mappings: Dict[str, str]
    threshold_indicators: Dict[str, float]
    domain_specific_indicators: Dict[str, List[str]]
    citations: List[str]
    
    def get_domain_indicators(self, domain: str) -> List[str]:
        """Get domain-specific indicators for this category"""
        domain = domain.lower()
        if domain in self.domain_specific_indicators:
            return self.domain_specific_indicators[domain]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "research_based_indicators": self.research_based_indicators,
            "framework_mappings": self.framework_mappings,
            "threshold_indicators": self.threshold_indicators,
            "domain_specific_indicators": self.domain_specific_indicators,
            "citations": self.citations
        }


@dataclass
class CategoryBoundary:
    """Research-based boundary between developmental categories"""
    
    from_category: str
    to_category: str
    boundary_indicators: List[str]
    threshold_value: float
    domain_specific_criteria: Dict[str, List[str]]
    citations: List[str]
    
    def get_domain_criteria(self, domain: str) -> List[str]:
        """Get domain-specific boundary criteria"""
        domain = domain.lower()
        if domain in self.domain_specific_criteria:
            return self.domain_specific_criteria[domain]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "from_category": self.from_category,
            "to_category": self.to_category,
            "boundary_indicators": self.boundary_indicators,
            "threshold_value": self.threshold_value,
            "domain_specific_criteria": self.domain_specific_criteria,
            "citations": self.citations
        }


class CategoryKnowledgeRepository:
    """Central repository for evidence-based category knowledge"""
    
    def __init__(self):
        """Initialize the repository with research-backed category knowledge"""
        self._categories = self._initialize_categories()
        self._boundaries = self._initialize_boundaries()
    
    def get_category_evidence(self, category_name: str) -> Optional[CategoryEvidence]:
        """Get evidence for a specific category"""
        category_name = category_name.upper()
        return self._categories.get(category_name)
    
    def get_category_boundary(self, from_category: str, to_category: str) -> Optional[CategoryBoundary]:
        """Get boundary definition between two categories"""
        key = (from_category.upper(), to_category.upper())
        return self._boundaries.get(key)
    
    def get_all_categories(self) -> List[CategoryEvidence]:
        """Get all category evidence definitions"""
        return list(self._categories.values())
    
    def get_all_boundaries(self) -> List[CategoryBoundary]:
        """Get all boundary definitions"""
        return list(self._boundaries.values())
    
    def get_domain_specific_evidence(self, category_name: str, domain: str) -> Dict[str, Any]:
        """Get domain-specific evidence for a category"""
        category = self.get_category_evidence(category_name)
        if not category:
            return {}
        
        return {
            "name": category.name,
            "description": category.description,
            "indicators": category.get_domain_indicators(domain),
            "citations": category.citations
        }
    
    def _initialize_categories(self) -> Dict[str, CategoryEvidence]:
        """Initialize evidence-based category definitions"""
        categories = {}
        
        # CANNOT_DO Category
        categories["CANNOT_DO"] = CategoryEvidence(
            name="CANNOT_DO",
            description="No evidence of skill emergence despite appropriate opportunities. The child does not demonstrate any components of the target behavior.",
            research_based_indicators=[
                "Multiple opportunities without evidence of skill",
                "No approximations of the target behavior",
                "No response to prompting or support for the skill",
                "Age significantly below typical milestone emergence"
            ],
            framework_mappings={
                "ASQ-3": "Not Yet (0 points)",
                "AEPS": "No Mastery (0)",
                "DAYC-2": "0 points (skill not in repertoire)",
                "Bayley-4": "Below -2 standard deviations on scaled scores"
            },
            threshold_indicators={
                "confidence_minimum": 0.70,
                "support_level": 1.0,
                "consistency": 0.0
            },
            domain_specific_indicators={
                "motor": [
                    "No attempt or approximation of the movement pattern",
                    "No response to physical prompting for the movement",
                    "No engagement with materials needed for the skill"
                ],
                "communication": [
                    "No evidence of targeted communication function",
                    "No attempts to communicate in this mode",
                    "No response to communication models or prompts"
                ],
                "social": [
                    "No social engagement or awareness of social opportunities",
                    "No response to social bids from others",
                    "No interest in social interaction related to the skill"
                ],
                "cognitive": [
                    "No evidence of problem-solving approach or conceptual understanding",
                    "No engagement with cognitive tasks or materials",
                    "No response to demonstration of cognitive skill"
                ]
            },
            citations=[
                "Bayley, N. (2015). Bayley Scales of Infant and Toddler Development (Fourth Edition). San Antonio, TX: Pearson.",
                "Squires, J., & Bricker, D. (2009). Ages & Stages Questionnaires (ASQ-3): A Parent-Completed Child-Monitoring System (3rd ed.). Baltimore, MD: Brookes Publishing.",
                "Bricker, D., et al. (2002). Assessment, Evaluation, and Programming System for Infants and Children (2nd ed.). Baltimore, MD: Brookes Publishing."
            ]
        )
        
        # WITH_SUPPORT Category
        categories["WITH_SUPPORT"] = CategoryEvidence(
            name="WITH_SUPPORT",
            description="Child demonstrates the skill only with specific types of assistance, prompting, or environmental modifications.",
            research_based_indicators=[
                "Performance with physical guidance or hand-over-hand assistance",
                "Performance with verbal prompts or modeling",
                "Performance with environmental adaptations",
                "Performance in limited contexts or with specific materials",
                "Requirement for multiple prompts to initiate or complete"
            ],
            framework_mappings={
                "AEPS": "Assistance (A) or Modification (M)",
                "Bayley-4": "Item passed with examiner support",
                "ASQ-3": "Partial credit toward Sometimes category when support is required"
            },
            threshold_indicators={
                "confidence_minimum": 0.65,
                "support_level": 0.75,
                "consistency": 0.3
            },
            domain_specific_indicators={
                "motor": [
                    "Requires physical positioning, stabilization, or modified environment",
                    "Performs movement with hand-over-hand guidance",
                    "Needs physical support for balance or coordination"
                ],
                "communication": [
                    "Communicates only when structured opportunities are created",
                    "Requires models or prompts to use communication form",
                    "Communicates only with familiar partners using specific prompts"
                ],
                "social": [
                    "Engages socially only in highly structured interactions with significant scaffolding",
                    "Requires adult facilitation to maintain social exchange",
                    "Participates with support but does not initiate social interaction"
                ],
                "cognitive": [
                    "Demonstrates thinking with substantial adult scaffolding or concrete supports",
                    "Requires step-by-step guidance to complete cognitive tasks",
                    "Shows understanding only with extensive prompting or simplified materials"
                ]
            },
            citations=[
                "Bricker, D., et al. (2002). Assessment, Evaluation, and Programming System for Infants and Children (2nd ed.). Baltimore, MD: Brookes Publishing.",
                "McWilliam, R. A., & Casey, A. M. (2008). Engagement of every child in the preschool classroom. Baltimore, MD: Brookes Publishing.",
                "Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes. Cambridge, MA: Harvard University Press."
            ]
        )
        
        # EMERGING Category
        categories["EMERGING"] = CategoryEvidence(
            name="EMERGING",
            description="Skill is beginning to develop; performance is inconsistent or partial across attempts, contexts, or components.",
            research_based_indicators=[
                "Inconsistent performance (sometimes successful, sometimes not)",
                "Partial demonstration of skill components",
                "Successful in some contexts but not others",
                "Variable quality of performance",
                "Recently appearing behavior still being consolidated"
            ],
            framework_mappings={
                "AEPS": "Emerging/Inconsistent (1)",
                "ASQ-3": "Sometimes (5 points)",
                "Bayley-4": "Inconsistent performance on items",
                "DAYC-2": "Partial criteria met for some items"
            },
            threshold_indicators={
                "confidence_minimum": 0.60,
                "support_level": 0.25,
                "consistency": 0.5
            },
            domain_specific_indicators={
                "motor": [
                    "Demonstrates basic movement pattern but with quality issues (e.g., balance, coordination, efficiency)",
                    "Performs skill inconsistently across attempts",
                    "Shows partial components of the movement pattern"
                ],
                "communication": [
                    "Inconsistent use of communication form; limited contexts or partners",
                    "Variable use of the communication skill across situations",
                    "Some components of the communication skill are present, others absent"
                ],
                "social": [
                    "Inconsistent social engagement; may initiate but not maintain or respond but not initiate",
                    "Variable quality of social interaction across partners or settings",
                    "Partial demonstration of social skill components"
                ],
                "cognitive": [
                    "Inconsistent application of concepts; success depends on familiarity and complexity",
                    "Partial understanding of the cognitive concept",
                    "Sometimes uses the thinking skill effectively, sometimes not"
                ]
            },
            citations=[
                "Squires, J., & Bricker, D. (2009). Ages & Stages Questionnaires (ASQ-3): A Parent-Completed Child-Monitoring System (3rd ed.). Baltimore, MD: Brookes Publishing.",
                "Hadders-Algra, M. (2010). Variation and variability: Key words in human motor development. Physical & Occupational Therapy in Pediatrics, 30(4), 345-352.",
                "Thelen, E., & Smith, L. B. (1994). A dynamic systems approach to the development of cognition and action. Cambridge, MA: MIT Press."
            ]
        )
        
        # INDEPENDENT Category
        categories["INDEPENDENT"] = CategoryEvidence(
            name="INDEPENDENT",
            description="Child consistently demonstrates the skill without assistance across relevant contexts and situations.",
            research_based_indicators=[
                "Consistent performance without prompting",
                "Generalization across settings and materials",
                "Fluent, efficient execution of the skill",
                "Self-initiated use of the skill",
                "Appropriate use across relevant situations"
            ],
            framework_mappings={
                "AEPS": "Mastery (2)",
                "ASQ-3": "Yes (10 points)",
                "DAYC-2": "1 point (skill in repertoire)",
                "Bayley-4": "Clear pass on item"
            },
            threshold_indicators={
                "confidence_minimum": 0.80,
                "support_level": 0.0,
                "consistency": 0.8
            },
            domain_specific_indicators={
                "motor": [
                    "Smooth, coordinated execution with appropriate posture and biomechanics",
                    "Performs movement consistently across contexts",
                    "Integrates skill into functional activities spontaneously"
                ],
                "communication": [
                    "Flexible use of communication across partners, contexts, and functions",
                    "Initiates communication independently",
                    "Uses communication skill effectively for multiple purposes"
                ],
                "social": [
                    "Reciprocal social interaction with appropriate initiation and response",
                    "Engages socially across different settings and partners",
                    "Maintains and extends social interaction independently"
                ],
                "cognitive": [
                    "Flexible application of thinking strategies across novel situations",
                    "Generalizes cognitive skill to new problems or contexts",
                    "Integrates cognitive skill with other abilities"
                ]
            },
            citations=[
                "Bricker, D., et al. (2002). Assessment, Evaluation, and Programming System for Infants and Children (2nd ed.). Baltimore, MD: Brookes Publishing.",
                "Squires, J., & Bricker, D. (2009). Ages & Stages Questionnaires (ASQ-3): A Parent-Completed Child-Monitoring System (3rd ed.). Baltimore, MD: Brookes Publishing.",
                "Bayley, N. (2015). Bayley Scales of Infant and Toddler Development (Fourth Edition). San Antonio, TX: Pearson."
            ]
        )
        
        # LOST_SKILL Category
        categories["LOST_SKILL"] = CategoryEvidence(
            name="LOST_SKILL",
            description="A previously mastered skill that the child no longer demonstrates despite appropriate opportunities.",
            research_based_indicators=[
                "Clear documentation/evidence of previous mastery",
                "Current inability to demonstrate the skill",
                "Skill loss not explained by context or motivation",
                "Pattern of regression rather than fluctuation",
                "Temporal association with developmental concerns"
            ],
            framework_mappings={
                "Clinical": "Developmental regression",
                "M-CHAT-R": "Loss of skills item (critical)",
                "ADOS-2": "Developmental regression noted in history",
                "DSM-5": "Loss of previously acquired skills (ASD diagnostic criterion)"
            },
            threshold_indicators={
                "confidence_minimum": 0.75,
                "prior_mastery_evidence": 0.9,
                "current_performance": 0.2
            },
            domain_specific_indicators={
                "motor": [
                    "Previously fluid movement now absent or significantly deteriorated",
                    "Loss of previously established motor pattern",
                    "Regression to earlier motor pattern or quality"
                ],
                "communication": [
                    "Reduction in communication functions previously established",
                    "Loss of words or communicative forms previously used",
                    "Return to pre-linguistic communication after using words"
                ],
                "social": [
                    "Decrease in social engagement behaviors previously established",
                    "Loss of social skills such as joint attention or social referencing",
                    "Reduced interest in social interaction compared to earlier functioning"
                ],
                "cognitive": [
                    "Inability to demonstrate previously established cognitive strategies",
                    "Loss of problem-solving approaches previously mastered",
                    "Return to less mature cognitive patterns"
                ]
            },
            citations=[
                "Ozonoff, S., et al. (2010). A prospective study of the emergence of early behavioral signs of autism. Journal of the American Academy of Child & Adolescent Psychiatry, 49(3), 256-266.",
                "Pearson, N., et al. (2018). Regression in autism spectrum disorder: Reconciling findings from retrospective and prospective research. Autism Research, 11(12), 1602-1620.",
                "American Psychiatric Association. (2013). Diagnostic and statistical manual of mental disorders (5th ed.). Arlington, VA: American Psychiatric Publishing."
            ]
        )
        
        return categories
    
    def _initialize_boundaries(self) -> Dict[Tuple[str, str], CategoryBoundary]:
        """Initialize research-based boundary definitions between categories"""
        boundaries = {}
        
        # CANNOT_DO to WITH_SUPPORT boundary
        boundaries[("CANNOT_DO", "WITH_SUPPORT")] = CategoryBoundary(
            from_category="CANNOT_DO",
            to_category="WITH_SUPPORT",
            boundary_indicators=[
                "First evidence of skill with maximal support (>75% support)",
                "Initial response to intensive prompting",
                "Beginning awareness of skill requirements",
                "First approximations with significant assistance"
            ],
            threshold_value=0.75,
            domain_specific_criteria={
                "motor": [
                    "First movement approximations with physical guidance",
                    "Initial postural responses with maximal support",
                    "Beginning to tolerate positioning for skill"
                ],
                "communication": [
                    "First communicative attempts with intensive prompting",
                    "Initial responses to communication opportunities with support",
                    "Beginning to attend to models of the communication form"
                ],
                "social": [
                    "First social responses with intensive facilitation",
                    "Initial moments of shared attention with support",
                    "Brief social engagement with significant scaffolding"
                ],
                "cognitive": [
                    "First demonstrations of understanding with concrete supports",
                    "Initial problem-solving attempts with step-by-step guidance",
                    "Beginning awareness of cognitive task requirements"
                ]
            },
            citations=[
                "Bricker, D., et al. (2002). Assessment, Evaluation, and Programming System for Infants and Children (2nd ed.). Baltimore, MD: Brookes Publishing.",
                "McWilliam, R. A., & Casey, A. M. (2008). Engagement of every child in the preschool classroom. Baltimore, MD: Brookes Publishing."
            ]
        )
        
        # WITH_SUPPORT to EMERGING boundary
        boundaries[("WITH_SUPPORT", "EMERGING")] = CategoryBoundary(
            from_category="WITH_SUPPORT",
            to_category="EMERGING",
            boundary_indicators=[
                "Reduction in support needed (<75% support) with some independent attempts",
                "Occasional successful performance without prompting",
                "Decreased intensity of prompts needed",
                "Some self-initiated attempts at the skill"
            ],
            threshold_value=0.25,
            domain_specific_criteria={
                "motor": [
                    "Occasional movement success with minimal physical guidance",
                    "Some independent attempts at movement pattern",
                    "Reduced level of physical support needed"
                ],
                "communication": [
                    "Some spontaneous communication attempts",
                    "Reduced prompting required for some instances",
                    "Occasional use of communication form without direct models"
                ],
                "social": [
                    "Some self-initiated social overtures",
                    "Occasional participation without direct facilitation",
                    "Reduced adult involvement in some social exchanges"
                ],
                "cognitive": [
                    "Some problem-solving with reduced supports",
                    "Occasional application of concept without prompting",
                    "Independent attempts at simpler versions of the cognitive task"
                ]
            },
            citations=[
                "Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes. Cambridge, MA: Harvard University Press.",
                "Thelen, E., & Smith, L. B. (1994). A dynamic systems approach to the development of cognition and action. Cambridge, MA: MIT Press."
            ]
        )
        
        # EMERGING to INDEPENDENT boundary
        boundaries[("EMERGING", "INDEPENDENT")] = CategoryBoundary(
            from_category="EMERGING",
            to_category="INDEPENDENT",
            boundary_indicators=[
                "Consistent performance (>80% success) across contexts without support",
                "Consolidated skill components into fluent execution",
                "Reliable generalization to new situations",
                "Regular self-initiated use of the skill"
            ],
            threshold_value=0.80,
            domain_specific_criteria={
                "motor": [
                    "Consistent quality of movement across attempts",
                    "Fluid integration of all movement components",
                    "Reliable performance across different environments"
                ],
                "communication": [
                    "Consistent use of communication form across contexts",
                    "Reliable integration of all communication components",
                    "Effective use with different communication partners"
                ],
                "social": [
                    "Consistent social engagement across settings",
                    "Reliable initiation and maintenance of social interaction",
                    "Effective social participation with various partners"
                ],
                "cognitive": [
                    "Consistent application of thinking strategy",
                    "Reliable problem-solving across different contexts",
                    "Effective generalization to novel situations"
                ]
            },
            citations=[
                "Squires, J., & Bricker, D. (2009). Ages & Stages Questionnaires (ASQ-3): A Parent-Completed Child-Monitoring System (3rd ed.). Baltimore, MD: Brookes Publishing.",
                "Bricker, D., et al. (2002). Assessment, Evaluation, and Programming System for Infants and Children (2nd ed.). Baltimore, MD: Brookes Publishing."
            ]
        )
        
        # INDEPENDENT to LOST_SKILL boundary
        boundaries[("INDEPENDENT", "LOST_SKILL")] = CategoryBoundary(
            from_category="INDEPENDENT",
            to_category="LOST_SKILL",
            boundary_indicators=[
                "Documented regression from previous independent performance",
                "Clear evidence of prior mastery followed by skill absence",
                "Persistent loss across multiple opportunities",
                "Skill absence not explained by motivation or context"
            ],
            threshold_value=0.90,
            domain_specific_criteria={
                "motor": [
                    "Previously mastered movement now consistently absent",
                    "Significant quality deterioration in previously fluid movement",
                    "Return to earlier motor patterns after clear mastery"
                ],
                "communication": [
                    "Loss of previously mastered communication forms",
                    "Significant reduction in previously established communication functions",
                    "Return to pre-linguistic communication after established language"
                ],
                "social": [
                    "Withdrawal from previously enjoyed social interactions",
                    "Loss of social reciprocity previously established",
                    "Significant reduction in social engagement behaviors"
                ],
                "cognitive": [
                    "Inability to solve problems previously mastered",
                    "Loss of conceptual understanding previously demonstrated",
                    "Return to earlier cognitive strategies after clear advancement"
                ]
            },
            citations=[
                "Ozonoff, S., et al. (2010). A prospective study of the emergence of early behavioral signs of autism. Journal of the American Academy of Child & Adolescent Psychiatry, 49(3), 256-266.",
                "Pearson, N., et al. (2018). Regression in autism spectrum disorder: Reconciling findings from retrospective and prospective research. Autism Research, 11(12), 1602-1620.",
                "Thurm, A., et al. (2014). Developmental regression in autism spectrum disorder. Neuropsychology Review, 24(2), 186-194."
            ]
        )
        
        return boundaries


# Module singleton
_repository = CategoryKnowledgeRepository()

def get_category_evidence(category_name: str) -> Optional[CategoryEvidence]:
    """Get evidence for a specific category"""
    return _repository.get_category_evidence(category_name)

def get_category_boundary(from_category: str, to_category: str) -> Optional[CategoryBoundary]:
    """Get boundary definition between two categories"""
    return _repository.get_category_boundary(from_category, to_category)

def get_domain_specific_evidence(category_name: str, domain: str) -> Dict[str, Any]:
    """Get domain-specific evidence for a category"""
    return _repository.get_domain_specific_evidence(category_name, domain)

def get_all_categories() -> List[CategoryEvidence]:
    """Get all category evidence definitions"""
    return _repository.get_all_categories()

def get_all_boundaries() -> List[CategoryBoundary]:
    """Get all boundary definitions"""
    return _repository.get_all_boundaries() 