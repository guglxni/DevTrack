"""
Developmental Domains Module

This module defines domain-specific knowledge for different developmental domains
and provides structured information about milestone categories.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logger = logging.getLogger("knowledge_engineering")

@dataclass
class CategoryRubric:
    """Detailed rubric for a milestone category"""
    name: str  # Category name (e.g., "EMERGING")
    description: str  # General description of the category
    criteria: List[str]  # Specific criteria for this category
    examples: List[str]  # Example responses that fit this category
    keywords: List[str]  # Keywords that suggest this category
    transitions: Dict[str, List[str]]  # Boundary indicators with neighboring categories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "criteria": self.criteria,
            "examples": self.examples,
            "keywords": self.keywords,
            "transitions": self.transitions
        }


@dataclass
class DevelopmentalDomain:
    """Representation of a developmental domain with specific knowledge"""
    code: str  # Short code (e.g., "GM" for Gross Motor)
    name: str  # Full name (e.g., "Gross Motor Skills")
    description: str  # Detailed description of the domain
    milestone_types: List[str]  # Types of milestones in this domain
    assessment_considerations: List[str]  # Special considerations for assessment
    category_rubrics: Dict[str, CategoryRubric]  # Rubrics for each category
    domain_specific_prompts: Optional[Dict[str, str]] = field(default_factory=dict)  # Domain-specific prompt templates
    
    def get_category_rubric(self, category_name: str) -> Optional[CategoryRubric]:
        """Get rubric for a specific category"""
        return self.category_rubrics.get(category_name.upper())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "milestone_types": self.milestone_types,
            "assessment_considerations": self.assessment_considerations,
            "category_rubrics": {k: v.to_dict() for k, v in self.category_rubrics.items()},
            "domain_specific_prompts": self.domain_specific_prompts
        }


# Initialize domain-specific knowledge
_DOMAINS = {}

def _initialize_motor_domain() -> DevelopmentalDomain:
    """Initialize the motor development domain"""
    
    # Create category rubrics for motor domain
    cannot_do = CategoryRubric(
        name="CANNOT_DO",
        description="Child cannot perform the motor skill at all or shows no attempt",
        criteria=[
            "No evidence of the motor skill being attempted",
            "Child shows physical inability to perform the movement",
            "Child does not understand the motor task when demonstrated",
            "Complete absence of prerequisite movements for the skill"
        ],
        examples=[
            "He doesn't even try to walk, even when I hold his hands",
            "She can't hold objects in her hand at all",
            "When I try to get him to throw a ball, he just drops it"
        ],
        keywords=[
            "cannot", "doesn't", "not able", "unable", "never", "no attempt", 
            "won't try", "fails", "can't", "doesn't understand"
        ],
        transitions={
            "EMERGING": [
                "Shows interest but cannot execute",
                "Attempts movement but fails to complete",
                "Watches others do the skill with interest"
            ]
        }
    )
    
    lost_skill = CategoryRubric(
        name="LOST_SKILL",
        description="Child previously demonstrated the motor skill but has lost the ability",
        criteria=[
            "Previously consistent performance that has disappeared",
            "Regression to earlier motor pattern",
            "Loss of motor milestone after illness or significant change",
            "Parent reports clear change from established ability to inability"
        ],
        examples=[
            "She used to walk easily but now refuses to put weight on her legs",
            "He could stack blocks last month, but now he just knocks them down",
            "After his ear infection, he stopped being able to balance on one foot"
        ],
        keywords=[
            "used to", "before", "previously", "stopped", "no longer", "lost",
            "regressed", "went backwards", "doesn't anymore", "now refuses"
        ],
        transitions={
            "CANNOT_DO": [
                "Complete loss without any attempts",
                "Actively avoids the previously mastered skill"
            ],
            "EMERGING": [
                "Occasionally shows remnants of previous ability",
                "Shows frustration when attempting previously mastered skill"
            ]
        }
    )
    
    emerging = CategoryRubric(
        name="EMERGING",
        description="Child is beginning to show the motor skill but is inconsistent",
        criteria=[
            "Occasional successful attempts mixed with failures",
            "Partial completion of the motor sequence",
            "Requires ideal conditions to demonstrate skill",
            "Shows understanding of the goal but execution is immature"
        ],
        examples=[
            "Sometimes he can jump with both feet, but usually he steps down with one foot first",
            "She's tried holding the crayon correctly a few times but can't maintain the grip",
            "He can stand on tiptoes momentarily but then falls back to flat feet"
        ],
        keywords=[
            "sometimes", "trying", "learning", "beginning to", "occasionally", 
            "not consistent", "on occasion", "now and then", "started to"
        ],
        transitions={
            "LOST_SKILL": [
                "Recently showed skill more consistently but now struggling",
                "Shows frustration at inability to perform previously easier task"
            ],
            "WITH_SUPPORT": [
                "Can complete with physical guidance but attempts independently",
                "Seeks minimal assistance after attempting independently",
                "Performs better with verbal encouragement but tries alone first"
            ]
        }
    )
    
    with_support = CategoryRubric(
        name="WITH_SUPPORT",
        description="Child can perform the motor skill with assistance or environmental modification",
        criteria=[
            "Needs physical guidance to complete the movement",
            "Needs environmental modifications (e.g., supportive seating)",
            "Requires verbal cues throughout the movement pattern",
            "Can perform in some contexts with support but not in others"
        ],
        examples=[
            "He can walk upstairs if I hold his hand but not independently",
            "She can kick a ball if I position her body first",
            "When I give him verbal reminders, he can hop on one foot"
        ],
        keywords=[
            "with help", "if I help", "when assisted", "with support", "guided",
            "prompted", "reminded", "needs assistance", "with modification"
        ],
        transitions={
            "EMERGING": [
                "Requires significant support most times",
                "Support needed throughout the entire movement",
                "Cannot initiate the movement without assistance"
            ],
            "INDEPENDENT": [
                "Occasionally performs skill independently",
                "Needs minimal verbal reminders only",
                "Support needed only for complex variations of the skill"
            ]
        }
    )
    
    independent = CategoryRubric(
        name="INDEPENDENT",
        description="Child performs the motor skill consistently without help",
        criteria=[
            "Consistent performance across different environments",
            "No physical assistance or verbal cues needed",
            "Fluid, efficient movement pattern",
            "Can adapt the skill to different contexts"
        ],
        examples=[
            "She can run, jump, and climb without any help at all",
            "He draws circles and lines with good control every time",
            "She easily rides her tricycle around obstacles without assistance"
        ],
        keywords=[
            "independently", "by himself", "on her own", "without help", "mastered",
            "consistently", "easily", "always", "every time", "expertly"
        ],
        transitions={
            "WITH_SUPPORT": [
                "Occasionally needs minimal assistance in challenging situations",
                "Performs independently in familiar settings only"
            ]
        }
    )
    
    # Create the motor domain with all rubrics
    return DevelopmentalDomain(
        code="MOTOR",
        name="Motor Development",
        description="Assessment of fine and gross motor skills including balance, coordination, and manual dexterity",
        milestone_types=[
            "Gross Motor - Large movements using large muscle groups",
            "Fine Motor - Small, precise movements using small muscle groups",
            "Balance and Coordination - Stabilizing body position and coordinating movements",
            "Motor Planning - Ability to conceptualize, plan, and execute novel movements"
        ],
        assessment_considerations=[
            "Consider the physical environment where skills are observed",
            "Note any physical limitations that might affect motor performance",
            "Assess persistence and effort alongside skill acquisition",
            "Consider both accuracy and quality of movement patterns",
            "Evaluate consistency across multiple attempts"
        ],
        category_rubrics={
            "CANNOT_DO": cannot_do,
            "LOST_SKILL": lost_skill,
            "EMERGING": emerging,
            "WITH_SUPPORT": with_support,
            "INDEPENDENT": independent
        }
    )


def _initialize_communication_domain() -> DevelopmentalDomain:
    """Initialize the communication development domain"""
    
    # Create category rubrics for communication domain
    cannot_do = CategoryRubric(
        name="CANNOT_DO",
        description="Child shows no communication abilities for this milestone",
        criteria=[
            "No attempts at the specified communication behavior",
            "No alternative communication strategies to achieve the same goal",
            "Unable to understand or process related communication input",
            "No responsiveness to models or prompts for this communication skill"
        ],
        examples=[
            "He doesn't make any sounds that resemble words",
            "She never points to things she wants",
            "He doesn't respond when I ask him simple questions"
        ],
        keywords=[
            "doesn't communicate", "no words", "never uses", "hasn't started",
            "makes no attempt", "doesn't respond", "no understanding"
        ],
        transitions={
            "EMERGING": [
                "Shows interest in communication but doesn't attempt",
                "Understands the concept but doesn't produce communication",
                "Responds to others' communication but doesn't initiate"
            ]
        }
    )
    
    lost_skill = CategoryRubric(
        name="LOST_SKILL",
        description="Child previously demonstrated the communication skill but has lost it",
        criteria=[
            "Clear history of using the communication skill consistently in the past",
            "Abrupt or gradual loss of previously acquired communication abilities",
            "May show frustration at inability to communicate as before",
            "Parent reports specific examples of previous communication abilities"
        ],
        examples=[
            "He used to say several words clearly but now doesn't speak at all",
            "She used to ask 'what's that?' about everything but stopped completely",
            "He was putting two words together last month but now only uses single words"
        ],
        keywords=[
            "used to", "previously", "before", "lost", "stopped", "disappeared",
            "regressed", "gave up", "no longer", "went back to"
        ],
        transitions={
            "CANNOT_DO": [
                "Complete abandonment of communication method",
                "Shows no remnant of previous communication skill"
            ],
            "EMERGING": [
                "Occasionally shows traces of former communication ability",
                "Attempts communication but gives up quickly"
            ]
        }
    )
    
    emerging = CategoryRubric(
        name="EMERGING",
        description="Child is beginning to demonstrate the communication skill inconsistently",
        criteria=[
            "Occasional successful use of the communication skill",
            "Attempts the communication behavior with partial success",
            "Shows understanding of the communication goal even when execution fails",
            "Communication behavior appears in specific contexts only"
        ],
        examples=[
            "Sometimes he'll say a few two-word phrases, but mostly single words",
            "She's starting to point at things she wants but not consistently",
            "He follows simple directions occasionally but seems to forget frequently"
        ],
        keywords=[
            "sometimes", "occasionally", "beginning to", "starting to", "trying to",
            "inconsistent", "not reliable", "hit or miss", "now and then"
        ],
        transitions={
            "LOST_SKILL": [
                "Recently demonstrated more consistently but now struggling",
                "Shows frustration at communication attempts"
            ],
            "WITH_SUPPORT": [
                "Successfully communicates with significant prompting",
                "Needs models but makes some independent attempts"
            ]
        }
    )
    
    with_support = CategoryRubric(
        name="WITH_SUPPORT",
        description="Child can use the communication skill with assistance or prompting",
        criteria=[
            "Consistently communicates with prompts or models",
            "Needs scaffolding to complete communication sequence",
            "Uses skill when environment is optimally structured",
            "Requires specific cues to initiate communication"
        ],
        examples=[
            "He will repeat new words after me but doesn't use them spontaneously",
            "When I start the sentence for her, she can complete it correctly",
            "If I remind him, he'll use his words instead of pointing"
        ],
        keywords=[
            "with help", "when prompted", "if I remind", "after modeling",
            "needs encouragement", "with support", "given assistance"
        ],
        transitions={
            "EMERGING": [
                "Requires constant prompting throughout communication",
                "Support needed for every instance of communication type",
                "Cannot initiate this type of communication without help"
            ],
            "INDEPENDENT": [
                "Occasionally communicates without support",
                "Needs minimal prompting in some situations only",
                "Self-initiates with verbal reminder only"
            ]
        }
    )
    
    independent = CategoryRubric(
        name="INDEPENDENT",
        description="Child uses the communication skill consistently without assistance",
        criteria=[
            "Spontaneously uses communication skill across contexts",
            "Initiates communication without prompting",
            "Adapts communication to different listeners and situations",
            "Communication is fluent and natural"
        ],
        examples=[
            "She asks questions constantly without any prompting",
            "He always uses complete sentences to tell me what he wants",
            "She consistently uses 4-5 word sentences in everyday conversation"
        ],
        keywords=[
            "always", "consistently", "independently", "without help", "on his own",
            "unprompted", "spontaneously", "regularly", "by herself"
        ],
        transitions={
            "WITH_SUPPORT": [
                "Occasionally needs prompting in stressful situations",
                "Independent in familiar situations only"
            ]
        }
    )
    
    # Create the communication domain with all rubrics
    return DevelopmentalDomain(
        code="COMM",
        name="Communication",
        description="Assessment of receptive and expressive language skills including verbal and non-verbal communication",
        milestone_types=[
            "Receptive Language - Understanding spoken language and following directions",
            "Expressive Language - Using words, phrases and sentences to communicate",
            "Non-verbal Communication - Using gestures, facial expressions and body language",
            "Pragmatic Language - Social use of language in context"
        ],
        assessment_considerations=[
            "Consider both verbal and non-verbal communication attempts",
            "Assess communication across different contexts and listeners",
            "Note any speech sound production or articulation issues",
            "Consider communication intent even when execution is imperfect",
            "Evaluate both communication initiation and response"
        ],
        category_rubrics={
            "CANNOT_DO": cannot_do,
            "LOST_SKILL": lost_skill,
            "EMERGING": emerging,
            "WITH_SUPPORT": with_support,
            "INDEPENDENT": independent
        }
    )


def _initialize_social_domain() -> DevelopmentalDomain:
    """Initialize the social development domain"""
    
    # Create category rubrics for social domain
    cannot_do = CategoryRubric(
        name="CANNOT_DO",
        description="Child shows no evidence of the social skill",
        criteria=[
            "No attempts at social engagement of the specified type",
            "Appears unaware of social expectations in this area",
            "Does not respond to others' social initiations",
            "Shows no interest in this type of social interaction"
        ],
        examples=[
            "He doesn't make eye contact with anyone",
            "She shows no interest in other children at all",
            "He never responds when other kids try to play with him"
        ],
        keywords=[
            "no interest", "doesn't engage", "ignores", "unaware", "avoids",
            "never responds", "doesn't notice", "doesn't understand"
        ],
        transitions={
            "EMERGING": [
                "Shows awareness but actively avoids engagement",
                "Watches social interactions but won't participate",
                "Shows fleeting interest but quickly disengages"
            ]
        }
    )
    
    lost_skill = CategoryRubric(
        name="LOST_SKILL",
        description="Child previously demonstrated the social skill but no longer does",
        criteria=[
            "Clear history of previously demonstrated social abilities now absent",
            "Loss of social engagement skills after period of typical development",
            "Withdrawal from previously enjoyed social interactions",
            "Regression in social reciprocity or social awareness"
        ],
        examples=[
            "She used to enjoy playing with other kids but now just plays alone",
            "He used to greet familiar people enthusiastically but now ignores them",
            "She had several friends before, but now she avoids all social contact"
        ],
        keywords=[
            "used to", "previously", "before", "withdrawn", "regressed", "stopped",
            "no longer", "lost interest", "changed", "became isolated"
        ],
        transitions={
            "CANNOT_DO": [
                "Complete withdrawal from all social engagement",
                "Active rejection of all social overtures"
            ],
            "EMERGING": [
                "Occasionally shows glimpses of former social skills",
                "Shows ambivalence about social engagement"
            ]
        }
    )
    
    emerging = CategoryRubric(
        name="EMERGING",
        description="Child is beginning to show the social skill inconsistently",
        criteria=[
            "Shows social skill in very specific contexts only",
            "Occasional appropriate social responses mixed with inappropriate ones",
            "Brief periods of social engagement followed by disengagement",
            "Unpredictable social responses to similar situations"
        ],
        examples=[
            "Sometimes he'll play alongside other children, but usually plays alone",
            "She's starting to take turns occasionally but still struggles with it",
            "He makes eye contact briefly but then looks away"
        ],
        keywords=[
            "sometimes", "beginning to", "starting to", "inconsistent", "occasionally",
            "brief moments", "unpredictable", "variable", "hit or miss"
        ],
        transitions={
            "LOST_SKILL": [
                "Recently showed more consistent social skills but now regressing",
                "Shows knowledge of social rules but actively rejects them"
            ],
            "WITH_SUPPORT": [
                "Shows social skills when specifically coached",
                "Engages socially with significant scaffolding"
            ]
        }
    )
    
    with_support = CategoryRubric(
        name="WITH_SUPPORT",
        description="Child demonstrates the social skill with assistance or prompting",
        criteria=[
            "Engages socially when guided by adult",
            "Shows appropriate social behavior with reminders",
            "Social interactions improve with preparation or priming",
            "Needs help managing emotions during social exchanges"
        ],
        examples=[
            "He'll share toys if I remind him each time",
            "With coaching, she can join a group of children playing",
            "If I explain what to say first, he can ask other kids to play"
        ],
        keywords=[
            "with help", "when prompted", "with reminders", "if prepared",
            "when coached", "after explanation", "if guided"
        ],
        transitions={
            "EMERGING": [
                "Requires constant adult presence for social success",
                "Social skills fall apart immediately without support",
                "Needs physical prompting for social engagement"
            ],
            "INDEPENDENT": [
                "Needs minimal verbal prompts only",
                "Self-corrects after brief reminders",
                "Initiates social interactions with verbal encouragement only"
            ]
        }
    )
    
    independent = CategoryRubric(
        name="INDEPENDENT",
        description="Child consistently displays the social skill without assistance",
        criteria=[
            "Demonstrates social skills across multiple contexts",
            "Initiates appropriate social interactions without prompting",
            "Adapts social behavior to different people and situations",
            "Maintains social engagement for age-appropriate duration"
        ],
        examples=[
            "He always shares toys with friends without being reminded",
            "She easily makes friends wherever we go",
            "He naturally takes turns in conversations and activities"
        ],
        keywords=[
            "always", "consistently", "independently", "without prompting",
            "naturally", "easily", "skillfully", "automatically"
        ],
        transitions={
            "WITH_SUPPORT": [
                "Occasionally needs support in challenging social situations",
                "Independent with familiar peers but needs help with new people"
            ]
        }
    )
    
    # Create the social domain with all rubrics
    return DevelopmentalDomain(
        code="SOC",
        name="Social Development",
        description="Assessment of social interaction skills including peer relationships, social understanding, and group participation",
        milestone_types=[
            "Social Interaction - Engaging with others appropriately",
            "Social Cognition - Understanding others' perspectives and emotions",
            "Play Skills - Engaging in various types of play with others",
            "Group Participation - Following rules and collaborating in groups"
        ],
        assessment_considerations=[
            "Observe in multiple social contexts (family, peers, unfamiliar adults)",
            "Consider cultural differences in social expectations",
            "Assess both initiation and response to social overtures",
            "Note quality and duration of social interactions",
            "Evaluate social interest separate from social skill"
        ],
        category_rubrics={
            "CANNOT_DO": cannot_do,
            "LOST_SKILL": lost_skill,
            "EMERGING": emerging,
            "WITH_SUPPORT": with_support,
            "INDEPENDENT": independent
        }
    )


def _initialize_cognitive_domain() -> DevelopmentalDomain:
    """Initialize the cognitive development domain"""
    
    # Create category rubrics for cognitive domain
    cannot_do = CategoryRubric(
        name="CANNOT_DO",
        description="Child shows no evidence of the cognitive skill",
        criteria=[
            "No attempts to engage in this type of thinking or problem-solving",
            "Does not demonstrate understanding of the cognitive concept",
            "Shows no recognition of patterns or rules related to this skill",
            "Cannot imitate the cognitive skill when demonstrated"
        ],
        examples=[
            "He doesn't understand the concept of sorting by color at all",
            "She can't remember any details from stories we read",
            "He doesn't recognize shapes no matter how many times we practice"
        ],
        keywords=[
            "doesn't understand", "can't grasp", "no concept of", "unable to",
            "doesn't get it", "completely confused by", "no recognition"
        ],
        transitions={
            "EMERGING": [
                "Shows momentary interest but doesn't attempt",
                "Recognizes when others perform the skill but doesn't try",
                "May attend to demonstrations but doesn't participate"
            ]
        }
    )
    
    lost_skill = CategoryRubric(
        name="LOST_SKILL",
        description="Child previously demonstrated the cognitive skill but has lost it",
        criteria=[
            "Previously demonstrated understanding that is no longer evident",
            "Regression in conceptual understanding or problem-solving ability",
            "Loss of previously consistent memory or attentional skills",
            "Parent reports specific examples of previous cognitive abilities"
        ],
        examples=[
            "She used to know all her numbers to 20 but now doesn't recognize any",
            "He could solve simple puzzles before, but now he just stares at the pieces",
            "She used to remember our routines perfectly but now seems confused"
        ],
        keywords=[
            "used to know", "formerly understood", "previously could", "forgot",
            "lost the ability", "no longer remembers", "stopped understanding"
        ],
        transitions={
            "CANNOT_DO": [
                "No trace of previous understanding remains",
                "Reacts as if concept is completely new"
            ],
            "EMERGING": [
                "Shows fragments of previous understanding",
                "Seems frustrated by inability to perform previously mastered task"
            ]
        }
    )
    
    emerging = CategoryRubric(
        name="EMERGING",
        description="Child is beginning to show the cognitive skill inconsistently",
        criteria=[
            "Demonstrates the cognitive skill occasionally or partially",
            "Shows understanding in highly supportive contexts only",
            "Inconsistent memory or application of learned concepts",
            "Beginning attempts at problem-solving with varying success"
        ],
        examples=[
            "Sometimes she can count to 10, but often skips numbers or gets confused",
            "He occasionally sorts objects by color but gets distracted easily",
            "She's beginning to understand 'bigger' and 'smaller' but not consistently"
        ],
        keywords=[
            "sometimes", "occasionally", "beginning to", "starting to understand",
            "inconsistent", "variable", "can do simple versions", "partially"
        ],
        transitions={
            "LOST_SKILL": [
                "Recently showed more consistent understanding but now regressing",
                "Shows frustration at inability to perform previously easier task"
            ],
            "WITH_SUPPORT": [
                "Can complete with extensive support but attempts independently",
                "Shows understanding when concepts are highly scaffolded"
            ]
        }
    )
    
    with_support = CategoryRubric(
        name="WITH_SUPPORT",
        description="Child demonstrates the cognitive skill with assistance or scaffolding",
        criteria=[
            "Successfully uses the cognitive skill with prompting or cues",
            "Needs examples or models to solve similar problems",
            "Requires simplification or breakdown of complex concepts",
            "Benefits from external organization or structure"
        ],
        examples=[
            "With hints, he can complete the pattern sequence",
            "She can sort objects if I first remind her of the categories",
            "He remembers the steps in order if I give him visual cues"
        ],
        keywords=[
            "with help", "when guided", "if I remind", "with prompting",
            "needs examples", "after demonstration", "with structure"
        ],
        transitions={
            "EMERGING": [
                "Requires continuous step-by-step guidance",
                "Cannot maintain skill when support is removed",
                "Support needed for every aspect of the cognitive task"
            ],
            "INDEPENDENT": [
                "Needs minimal prompting for complex aspects only",
                "Self-corrects after brief reminders",
                "Support needed only for extending the skill to new contexts"
            ]
        }
    )
    
    independent = CategoryRubric(
        name="INDEPENDENT",
        description="Child consistently uses the cognitive skill without assistance",
        criteria=[
            "Applies cognitive concepts across different situations",
            "Solves problems independently using appropriate strategies",
            "Extends learning to new contexts without explicit teaching",
            "Demonstrates fluid thinking and adaptability"
        ],
        examples=[
            "She easily recognizes patterns and creates her own variations",
            "He consistently remembers multi-step instructions without reminders",
            "She can independently sort objects by multiple characteristics"
        ],
        keywords=[
            "independently", "consistently", "without help", "easily",
            "automatically", "comes up with", "figures out", "masters"
        ],
        transitions={
            "WITH_SUPPORT": [
                "Occasionally needs help with very complex examples",
                "Independent with familiar materials but needs support with novel ones"
            ]
        }
    )
    
    # Create the cognitive domain with all rubrics
    return DevelopmentalDomain(
        code="COG",
        name="Cognitive Development",
        description="Assessment of thinking skills including problem-solving, memory, attention, and conceptual understanding",
        milestone_types=[
            "Problem-solving - Finding solutions to challenges",
            "Memory - Storing and retrieving information",
            "Attention - Focusing on relevant stimuli and sustaining concentration",
            "Conceptual Understanding - Grasping abstract ideas and relationships"
        ],
        assessment_considerations=[
            "Consider both the process and outcome of cognitive tasks",
            "Assess generalization of skills across contexts",
            "Note the level of support needed for success",
            "Evaluate both novel problem-solving and applied learning",
            "Consider attention span and distractibility during assessment"
        ],
        category_rubrics={
            "CANNOT_DO": cannot_do,
            "LOST_SKILL": lost_skill,
            "EMERGING": emerging,
            "WITH_SUPPORT": with_support,
            "INDEPENDENT": independent
        }
    )


# Initialize all domains
def _initialize_domains():
    """Initialize all developmental domains"""
    _DOMAINS["MOTOR"] = _initialize_motor_domain()
    _DOMAINS["COMM"] = _initialize_communication_domain()
    _DOMAINS["SOC"] = _initialize_social_domain()
    _DOMAINS["COG"] = _initialize_cognitive_domain()
    logger.info(f"Initialized {len(_DOMAINS)} developmental domains")


# Initialize domains on module import
_initialize_domains()


def get_domain_by_name(domain_name: str) -> Optional[DevelopmentalDomain]:
    """
    Get a developmental domain by name or code
    
    Args:
        domain_name: Domain name or code (e.g., "MOTOR" or "Motor Development")
        
    Returns:
        DevelopmentalDomain or None if not found
    """
    # Try direct lookup
    if domain_name.upper() in _DOMAINS:
        return _DOMAINS[domain_name.upper()]
    
    # Try lookup by full name
    for domain in _DOMAINS.values():
        if domain_name.lower() in domain.name.lower():
            return domain
    
    # Not found
    return None


def get_all_domains() -> List[DevelopmentalDomain]:
    """
    Get all developmental domains
    
    Returns:
        List of all developmental domains
    """
    return list(_DOMAINS.values()) 