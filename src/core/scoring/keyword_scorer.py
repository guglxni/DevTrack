"""
Keyword-Based Scoring Module

This module implements a pattern and keyword-based scoring approach.
"""

import re
from typing import Dict, Any, Optional, List, Tuple, Set

from .base import BaseScorer, ScoringResult, Score


class KeywordBasedScorer(BaseScorer):
    """
    Scorer that uses pattern matching and keywords for scoring responses
    
    This implementation improves upon the original regex-based approach by:
    1. Using word boundary-aware pattern matching
    2. Implementing better negation detection
    3. Handling contextual keywords based on milestone domain
    4. Providing detailed reasoning for scores
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the keyword-based scorer"""
        super().__init__(config or self._default_config())
        self._patterns = self._compile_patterns()
        
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
            "confidence_threshold": 0.6,  # Minimum confidence needed
            "word_boundary_check": True,  # Whether to enforce word boundaries
            "detect_negations": True,     # Whether to detect negations
            "negation_scope": 5,          # Words to check after negation
            "min_pattern_matches": 1      # Minimum patterns needed for a score
        }
    
    def _compile_patterns(self) -> Dict[Score, List[re.Pattern]]:
        """Compile regex patterns for each score category"""
        patterns = {}
        
        # CANNOT_DO patterns
        patterns[Score.CANNOT_DO] = [
            r"\bno\b|\bnot\b|\bnever\b|\bdoesn'?t\b|\bcannot\b|\bcan'?t\b|\bunable\b",
            r"\bhasn'?t\b|\bhaven'?t\b|\bwouldn'?t\b|\bcouldn'?t\b|\bwon'?t\b",
            r"not\s+yet|not\s+able|not\s+capable|not\s+showing|not\s+demonstrated",
            r"hasn'?t\s+(started|begun|developed|mastered|learned|acquired)",
            r"doesn'?t\s+(do|show|demonstrate|exhibit|perform|understand|know|have)",
            r"zero\s+interest|no\s+interest|no\s+attempt|no\s+desire|no\s+sign",
            r"(complete(ly)?|total(ly)?)\s+(unable|incapable)",
            r"nowhere\s+near|not\s+close\s+to|far\s+from|lacking|absence\s+of"
        ]
        
        # LOST_SKILL patterns
        patterns[Score.LOST_SKILL] = [
            r"\bused\s+to\b|\bstopped\b|\bquit\b|\bceased\b|\blost\b|\bregressed\b|\bdeclined\b|\bdeteriorated\b",
            r"no\s+longer|not\s+anymore|previously|before\s+but\s+not\s+now",
            r"(had|could|did|was)\s+(mastered|able|capable)\s+but",
            r"ability\s+diminished|skill\s+deteriorated|went\s+backwards",
            r"(can'?t|doesn'?t|won'?t)\s+(do|perform)\s+anymore",
            r"regression|deterioration|setback|backward\s+step",
            r"lost\s+(interest|ability|skill|desire|capacity)"
        ]
        
        # EMERGING patterns
        patterns[Score.EMERGING] = [
            r"\bsometimes\b|\boccasionally\b|\bsporadically\b|\bintermittently\b",
            r"beginning\s+to|starting\s+to|learning\s+to|trying\s+to",
            r"inconsistent(ly)?|variable|varies|fluctuates",
            r"working\s+on|practicing|developing|emerging",
            r"some\s+progress|making\s+progress|getting\s+better",
            r"hit\s+and\s+miss|on\s+and\s+off|good\s+days\s+and\s+bad",
            r"partial(ly)?|somewhat|sort\s+of|kind\s+of|a\s+bit",
            r"(not\s+fully|not\s+completely)\s+(developed|mastered)"
        ]
        
        # WITH_SUPPORT patterns
        patterns[Score.WITH_SUPPORT] = [
            r"with\s+(help|support|assistance|guidance|prompting)",
            r"needs\s+(help|support|assistance|guidance|prompting|reminders)",
            r"requires\s+(help|support|assistance|guidance|prompting)",
            r"(can|does)\s+but\s+need[s]?\s+(help|support|assistance)",
            r"(aided|assisted|supported|guided|prompted)",
            r"hand[\s-]over[\s-]hand|physical\s+guidance",
            r"visual\s+(support|cues|reminders)|verbal\s+(cues|prompts)",
            r"(parent|caregiver|adult|teacher)\s+(helps|assists|supports)"
        ]
        
        # INDEPENDENT patterns
        patterns[Score.INDEPENDENT] = [
            r"\b(yes|always|consistently|completely|absolutely|definitely)\b",
            r"(independent(ly)?|autonomous(ly)?|by\s+(him|her|them)self)",
            r"without\s+(help|support|assistance|guidance|prompting)",
            r"no\s+help\s+needed|no\s+assistance\s+required",
            r"mastered|proficient|skilled|capable|competent",
            r"easily|confidently|routinely|regularly|habitually",
            r"fully\s+able|completely\s+able|totally\s+capable",
            r"no\s+problem|with\s+ease|very\s+well|excels\s+at"
        ]
        
        # Domain-specific patterns
        self.domain_patterns = {
            "MOTOR": {
                Score.CANNOT_DO: [
                    r"physically\s+unable|not\s+strong\s+enough|lacks\s+coordination",
                    r"can'?t\s+(move|control|balance|coordinate)",
                    r"not\s+mobile|immobile|no\s+movement",
                    r"does\s+not\s+(walk|crawl|stand|sit|reach|grasp)"
                ],
                Score.LOST_SKILL: [
                    r"used\s+to\s+(walk|crawl|stand|sit|reach|grasp)",
                    r"stopped\s+(walking|crawling|standing|reaching)",
                    r"regression\s+in\s+(mobility|motor|movement|coordination)",
                    r"lost\s+ability\s+to\s+(walk|crawl|stand|balance)"
                ],
                Score.EMERGING: [
                    r"wobbles|unsteady|shaky|tippy",
                    r"trying\s+to\s+(walk|crawl|stand|balance|grasp)",
                    r"emerging\s+(coordination|balance|strength)",
                    r"improving\s+motor\s+skills"
                ],
                Score.WITH_SUPPORT: [
                    r"walks\s+with\s+(help|support|assistance)",
                    r"needs\s+hand\s+held|needs\s+balance\s+support",
                    r"requires\s+physical\s+guidance\s+to\s+(move|manipulate)",
                    r"can\s+do\s+with\s+adapted\s+equipment"
                ],
                Score.INDEPENDENT: [
                    r"physically\s+capable|well[-\s]coordinated",
                    r"strong\s+(grasp|grip|arms|legs|core)",
                    r"excellent\s+(balance|coordination|motor\s+control)",
                    r"athletic|agile|dexterous"
                ]
            },
            "COMMUNICATION": {
                Score.CANNOT_DO: [
                    r"non[-\s]verbal|doesn'?t\s+speak|no\s+words",
                    r"no\s+vocalizations|silent|mute|doesn'?t\s+talk",
                    r"hasn'?t\s+started\s+(speaking|talking|communicating)",
                    r"makes\s+no\s+attempt\s+to\s+communicate"
                ],
                Score.LOST_SKILL: [
                    r"used\s+to\s+(say|speak|talk|communicate)",
                    r"lost\s+(words|language|speech|vocabulary)",
                    r"stopped\s+(talking|speaking|communicating)",
                    r"regressed\s+in\s+(speech|language|communication)"
                ],
                Score.EMERGING: [
                    r"babbling|cooing|attempting\s+words",
                    r"trying\s+to\s+(speak|talk|communicate|express)",
                    r"uses\s+some\s+words\s+but\s+not\s+sentences",
                    r"beginning\s+to\s+form\s+(words|phrases|sentences)"
                ],
                Score.WITH_SUPPORT: [
                    r"communicates\s+with\s+(picture\s+cards|signs|device)",
                    r"needs\s+(prompting|cues)\s+to\s+(speak|communicate)",
                    r"repeats\s+after\s+being\s+told|echolalic",
                    r"responds\s+but\s+doesn'?t\s+initiate\s+communication"
                ],
                Score.INDEPENDENT: [
                    r"verbally\s+fluent|communicates\s+clearly",
                    r"extensive\s+vocabulary|complex\s+sentences",
                    r"expressive\s+language|articulate",
                    r"initiates\s+and\s+maintains\s+conversation"
                ]
            },
            "SOCIAL": {
                Score.CANNOT_DO: [
                    r"not\s+interested\s+in\s+people|avoids\s+interaction",
                    r"no\s+eye\s+contact|doesn'?t\s+look\s+at\s+people",
                    r"ignores\s+others|oblivious\s+to\s+people",
                    r"no\s+social\s+awareness|socially\s+disconnected"
                ],
                Score.LOST_SKILL: [
                    r"used\s+to\s+be\s+social|was\s+previously\s+engaging",
                    r"stopped\s+(smiling|interacting|responding)\s+to\s+others",
                    r"lost\s+interest\s+in\s+social\s+engagement",
                    r"regressed\s+in\s+social\s+skills"
                ],
                Score.EMERGING: [
                    r"occasional\s+eye\s+contact|fleeting\s+engagement",
                    r"beginning\s+to\s+notice\s+others|some\s+interest\s+in\s+people",
                    r"inconsistent\s+social\s+response|variable\s+engagement",
                    r"shows\s+interest\s+but\s+doesn'?t\s+engage\s+fully"
                ],
                Score.WITH_SUPPORT: [
                    r"engages\s+when\s+others\s+initiate|responds\s+to\s+social\s+overtures",
                    r"participates\s+with\s+prompting|joins\s+with\s+encouragement",
                    r"parallel\s+play|plays\s+alongside\s+but\s+not\s+with\s+others",
                    r"needs\s+facilitation\s+in\s+group\s+settings"
                ],
                Score.INDEPENDENT: [
                    r"socially\s+engaged|seeks\s+out\s+others",
                    r"initiates\s+social\s+interaction|makes\s+friends\s+easily",
                    r"empathetic|aware\s+of\s+others'\s+feelings",
                    r"cooperative\s+play|takes\s+turns|shares\s+willingly"
                ]
            },
            "COGNITIVE": {
                Score.CANNOT_DO: [
                    r"doesn'?t\s+understand|cannot\s+comprehend",
                    r"no\s+concept\s+of|no\s+awareness\s+of",
                    r"unable\s+to\s+problem[-\s]solve|can'?t\s+figure\s+out",
                    r"not\s+interested\s+in\s+exploring|no\s+curiosity"
                ],
                Score.LOST_SKILL: [
                    r"used\s+to\s+understand|previously\s+demonstrated\s+knowledge",
                    r"forgotten\s+how\s+to|no\s+longer\s+remembers",
                    r"lost\s+ability\s+to\s+solve\s+problems|cognitive\s+regression",
                    r"skills\s+have\s+diminished|concepts\s+previously\s+mastered"
                ],
                Score.EMERGING: [
                    r"beginning\s+to\s+understand|starting\s+to\s+grasp\s+concepts",
                    r"sometimes\s+shows\s+understanding|inconsistently\s+comprehends",
                    r"working\s+on\s+problem[-\s]solving|developing\s+reasoning",
                    r"shows\s+interest\s+but\s+limited\s+mastery"
                ],
                Score.WITH_SUPPORT: [
                    r"understands\s+with\s+explanation|needs\s+concepts\s+broken\s+down",
                    r"solves\s+problems\s+with\s+hints|requires\s+guidance\s+to\s+think\s+through",
                    r"follows\s+instructions\s+with\s+reminders|needs\s+redirection",
                    r"learns\s+with\s+extensive\s+repetition|needs\s+multiple\s+examples"
                ],
                Score.INDEPENDENT: [
                    r"quick\s+learner|grasps\s+concepts\s+easily",
                    r"problem[-\s]solver|figures\s+things\s+out",
                    r"curious|inquisitive|explores\s+independently",
                    r"good\s+memory|retains\s+information|applies\s+knowledge"
                ]
            }
        }
        
        # Compile domain-specific patterns
        self.compiled_domain_patterns = {}
        for domain, domain_score_patterns in self.domain_patterns.items():
            self.compiled_domain_patterns[domain] = {}
            for score, pattern_list in domain_score_patterns.items():
                self.compiled_domain_patterns[domain][score] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
                ]
        
        # Compile all patterns with re.IGNORECASE
        compiled_patterns = {}
        for score, pattern_list in patterns.items():
            compiled_patterns[score] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
        return compiled_patterns
    
    def detect_negations(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect negation phrases in text
        
        Args:
            text: The text to analyze
            
        Returns:
            List of (start, end) positions of negation phrases
        """
        # Common negation words and phrases
        negation_patterns = [
            r"\bnot\b",
            r"\bno\b",
            r"\bnever\b",
            r"\bdon't\b",
            r"\bdoesn't\b",
            r"\bcan't\b",
            r"\bcannot\b",
            r"\bwon't\b",
            r"\bisn't\b",
            r"\baren't\b",
            r"\bwasn't\b",
            r"\bweren't\b",
            r"\bhadn't\b",
            r"\bhasn't\b",
            r"\bhaven't\b",
            r"\bwouldn't\b",
            r"\bcouldn't\b",
            r"\bshouldn't\b",
            r"\bwithout\b",
            r"\black\s+of\b",
            r"\babsence\s+of\b",
            r"\bfail(s|ed|ing)?\s+to\b",
            r"\bunable\s+to\b",
            r"\brefuse(s|d)?\s+to\b"
        ]
        
        # Compile patterns
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in negation_patterns]
        
        # Find all matches
        negations = []
        for pattern in compiled_patterns:
            for match in pattern.finditer(text):
                negations.append((match.start(), match.end()))
        
        # Sort by position
        negations.sort()
        
        # Detect double negations (two negations close to each other)
        double_negations = []
        for i in range(len(negations) - 1):
            start1, end1 = negations[i]
            start2, end2 = negations[i + 1]
            
            # If two negations are within 5 words of each other, consider it a double negation
            if start2 - end1 <= 30:  # Approximately 5 words
                # Check if there's no other negation between them that would make it a triple negation
                is_double = True
                for j in range(len(negations)):
                    if j != i and j != i + 1:
                        start3, end3 = negations[j]
                        if end1 < start3 < start2:
                            is_double = False
                            break
                
                if is_double:
                    double_negations.append((start1, end2))
        
        # Return both single and double negations
        return negations, double_negations
    
    def score(self, 
              response: str, 
              milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a response using keyword-based analysis
        
        Args:
            response: The response text to score
            milestone_context: Optional context about the milestone
            
        Returns:
            ScoringResult: The scoring result
        """
        # Detect negations
        negations, double_negations = self.detect_negations(response)
        
        # Find matches for each category
        matches = {}
        for score, patterns in self._patterns.items():
            matches[score] = []
            for pattern in patterns:
                for match in pattern.finditer(response):
                    # Check if this match is part of a double negation
                    match_start, match_end = match.span()
                    is_in_double_negation = False
                    
                    for start, end in double_negations:
                        if start <= match_start and match_end <= end:
                            is_in_double_negation = True
                            break
                    
                    # Only count the match if it's not part of a double negation
                    # or if we're specifically looking for negation patterns
                    if not is_in_double_negation or score == Score.CANNOT_DO or score == Score.LOST_SKILL:
                        matches[score].append({
                            'pattern': pattern.pattern,
                            'matched_text': match.group(),
                            'start': match.start(),
                            'end': match.end()
                        })
        
        # Determine the most likely category
        category_counts = {s: len(m) for s, m in matches.items()}
        
        # If we have matches, find the category with the most matches
        if any(category_counts.values()):
            # Get the category with the most matches
            best_category = max(category_counts.items(), key=lambda x: x[1])
            category, count = best_category
            
            # Calculate confidence based on match count and uniqueness
            total_matches = sum(category_counts.values())
            confidence = min(0.6 + (count / 10), 0.9)  # Base confidence
            
            # Boost confidence if this category has significantly more matches than others
            if count > 0 and total_matches > 0:
                dominance = count / total_matches
                if dominance > 0.7:  # If this category has >70% of all matches
                    confidence = min(confidence + 0.1, 0.9)
            
            # Handle double negations for CANNOT_DO and LOST_SKILL
            if category in [Score.CANNOT_DO, Score.LOST_SKILL] and double_negations:
                # Check if the matches for this category are part of double negations
                category_match_positions = [(m['start'], m['end']) for m in matches[category]]
                
                # Count how many category matches are in double negations
                matches_in_double_negation = 0
                for cat_start, cat_end in category_match_positions:
                    for neg_start, neg_end in double_negations:
                        if neg_start <= cat_start and cat_end <= neg_end:
                            matches_in_double_negation += 1
                            break
                
                # If most of the matches are in double negations, this might actually be a positive statement
                if matches_in_double_negation > 0 and matches_in_double_negation / len(category_match_positions) > 0.5:
                    # Flip to the opposite category
                    if category == Score.CANNOT_DO:
                        category = Score.INDEPENDENT
                    elif category == Score.LOST_SKILL:
                        # For LOST_SKILL, a double negative might mean they still have the skill
                        category = Score.INDEPENDENT
                    
                    # Reduce confidence due to the complexity
                    confidence = max(0.5, confidence - 0.2)
            
            # Create reasoning text
            if count > 0:
                matched_texts = [m['matched_text'] for m in matches[category]]
                unique_matches = set(matched_texts)
                reasoning = f"Matched patterns for {category.name}: {', '.join(unique_matches)}"
            else:
                reasoning = "No clear pattern matched"
                category = Score.NOT_RATED
                confidence = 0.0
        else:
            # No matches found
            category = Score.NOT_RATED
            confidence = 0.0
            reasoning = "No clear pattern matched"
        
        # If confidence is too low, return NOT_RATED
        if confidence < self.config.get("confidence_threshold", 0.6):
            if category != Score.NOT_RATED:
                reasoning = f"Low confidence: {reasoning}"
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=confidence,
                method="keyword",
                reasoning=reasoning,
                details={"matches": matches}
            )
        
        return ScoringResult(
            score=category,
            confidence=confidence,
            method="keyword",
            reasoning=reasoning,
            details={"matches": matches}
        ) 