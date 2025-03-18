from typing import Dict, List, Optional, Tuple, Set, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re
import functools
import asyncio
import concurrent.futures
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os

# Import the original AssessmentEngine components
from src.core.assessment_engine import AssessmentEngine, STOPWORDS
# Import the original AssessmentEngine components
class Score(Enum):
    NOT_RATED = -1
    CANNOT_DO = 0      # Skill not acquired
    LOST_SKILL = 1     # Acquired but lost
    EMERGING = 2       # Emerging and inconsistent
    WITH_SUPPORT = 3   # Acquired but consistent in specific situations only
    INDEPENDENT = 4    # Acquired and present in all situations

@dataclass
class DevelopmentalMilestone:
    behavior: str
    criteria: str
    age_range: str
    domain: str
    keywords: List[str]
    scoring_rules: Dict[str, Score]

class EnhancedAssessmentEngine:
    """Enhanced assessment engine with NLP-based scoring and parallel processing"""
    
    def __init__(self, use_embeddings: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the enhanced assessment engine
        
        Args:
            use_embeddings: Whether to use sentence embeddings for scoring
            model_name: The sentence transformer model to use for embeddings
        """
        self.milestones = self._initialize_milestones()
        self.child_age: Optional[int] = None
        self.scores: Dict[str, Score] = {}
        self.current_milestone_index = 0
        self.assessed_milestones: Set[str] = set()
        self._scoring_keywords_cache = {}
        
        # Enhanced NLP scoring with embeddings
        self.use_embeddings = use_embeddings
        if use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                self.score_embeddings = self._initialize_score_embeddings()
                print(f"Successfully loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Warning: Could not load embedding model ({e}). Falling back to keyword-based scoring.")
                self.use_embeddings = False
    
    def _initialize_score_embeddings(self) -> Dict[Score, torch.Tensor]:
        """Initialize embeddings for each score type with representative phrases"""
        score_phrases = {
            Score.CANNOT_DO: [
                "cannot do this at all", 
                "not able to perform this task",
                "struggles completely with this", 
                "shows no ability in this area",
                "never demonstrates this skill"
            ],
            Score.LOST_SKILL: [
                "used to be able to do this",
                "could do this before but lost the skill",
                "previously demonstrated this but regressed",
                "had this ability but no longer shows it",
                "showed this skill in the past but not anymore"
            ],
            Score.EMERGING: [
                "beginning to show this skill",
                "sometimes demonstrates this",
                "occasionally shows this ability",
                "starting to develop this skill",
                "shows early signs of this ability"
            ],
            Score.WITH_SUPPORT: [
                "can do this with help",
                "performs this with assistance",
                "demonstrates this when supported",
                "shows this skill with guidance",
                "able to do this with adult help"
            ],
            Score.INDEPENDENT: [
                "does this independently",
                "fully demonstrates this skill",
                "consistently shows this ability",
                "performs this task on their own",
                "mastered this skill completely"
            ]
        }
        
        # Compute embeddings for each score type
        embeddings = {}
        for score, phrases in score_phrases.items():
            with torch.no_grad():
                # Compute embeddings for all phrases
                phrase_embeddings = self.embedding_model.encode(phrases, convert_to_tensor=True)
                # Average the embeddings
                embeddings[score] = torch.mean(phrase_embeddings, dim=0)
        
        return embeddings
    
    def reset_scores(self):
        """Reset all scores and milestone tracking for a fresh assessment"""
        self.scores = {}
        self.current_milestone_index = 0
        self.assessed_milestones = set()
        self._scoring_keywords_cache = {}
        print("Assessment engine scores and milestone tracking reset")
        
    def _initialize_milestones(self) -> List[DevelopmentalMilestone]:
        """Initialize developmental milestones by loading from CSV files in data/ directory"""
        milestones = []
        data_dir = "data"
        
        try:
            # Get all CSV files in the data directory
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            if not csv_files:
                print("Warning: No CSV files found in data/ directory, falling back to hardcoded milestones")
                return self._initialize_hardcoded_milestones()
            
            # Process each CSV file (each represents a domain)
            for csv_file in csv_files:
                # Extract domain from filename (e.g., "CDDC GM-Table 1.csv" -> "GM")
                # Format is typically "CDDC DomainCode-Table 1.csv"
                domain = None
                if 'CDDC' in csv_file and '-' in csv_file:
                    parts = csv_file.split(' ')
                    if len(parts) > 1:
                        domain = parts[1].split('-')[0]
                
                if not domain:
                    print(f"Warning: Could not extract domain from filename {csv_file}, skipping")
                    continue
                    
                print(f"Processing domain: {domain} from file: {csv_file}")
                
                # Load data from CSV
                file_path = os.path.join(data_dir, csv_file)
                try:
                    # Read CSV file with appropriate encoding
                    df = pd.read_csv(file_path, encoding='utf-8')
                    
                    # Check if DataFrame is valid - it should have Age, Checklist columns
                    required_columns = ['Checklist']
                    age_column = None
                    for col in df.columns:
                        if 'age' in col.lower() or 'month' in col.lower():
                            age_column = col
                            break
                    
                    if df.empty or 'Checklist' not in df.columns or not age_column:
                        print(f"Warning: CSV file {csv_file} has invalid format, skipping")
                        continue
                    
                    # Clean up data and extract milestones
                    current_age_range = None
                    
                    # Iterate through rows
                    for _, row in df.iterrows():
                        # Skip rows without checklist item
                        if pd.isna(row['Checklist']) or str(row['Checklist']).strip() == '':
                            continue
                        
                        # Update age range if present
                        if not pd.isna(row[age_column]) and str(row[age_column]).strip() != '':
                            age_str = str(row[age_column]).strip()
                            # Convert age format if needed (e.g., "6-12 m" to "6-12 months")
                            if 'm' in age_str and 'months' not in age_str:
                                age_str = age_str.replace('m', 'months')
                            current_age_range = age_str
                        
                        # Skip if no age range has been set
                        if not current_age_range:
                            continue
                        
                        behavior = str(row['Checklist']).strip()
                        
                        # Get criteria if available, otherwise use behavior as criteria
                        criteria = ""
                        criteria_col = 'Criteria to train BOT'
                        if criteria_col in df.columns and not pd.isna(row[criteria_col]) and str(row[criteria_col]).strip() != '':
                            criteria = str(row[criteria_col]).strip()
                        else:
                            criteria = behavior
                        
                        # Generate keywords and scoring rules
                        keywords = self._generate_keywords(behavior, criteria)
                        scoring_rules = self._get_scoring_rules(behavior)
                        
                        # Create milestone and add to list
                        milestone = DevelopmentalMilestone(
                            behavior=behavior,
                            criteria=criteria,
                            age_range=current_age_range,
                            domain=domain,
                            keywords=keywords,
                            scoring_rules=scoring_rules
                        )
                        milestones.append(milestone)
                        print(f"Added milestone: {behavior} ({domain}, {current_age_range})")
                    
                except Exception as e:
                    print(f"Error processing CSV file {csv_file}: {str(e)}")
            
            # If no milestones were loaded from CSV files, fall back to hardcoded ones
            if not milestones:
                print("Warning: No valid milestones found in CSV files, falling back to hardcoded milestones")
                return self._initialize_hardcoded_milestones()
                
            print(f"Successfully loaded {len(milestones)} milestones from {len(csv_files)} CSV files")
            return milestones
                
        except Exception as e:
            print(f"Error loading milestone data from CSV files: {str(e)}")
            print("Falling back to hardcoded milestones")
            return self._initialize_hardcoded_milestones()
    
    def _initialize_hardcoded_milestones(self) -> List[DevelopmentalMilestone]:
        """Initialize developmental milestones with hardcoded criteria and scoring rules (fallback method)"""
        age_ranges = [
            "0-6 months",
            "6-12 months",
            "12-18 months",
            "18-24 months",
            "24-30 months",
            "30-36 months"
        ]
        
        # Pre-define all milestone data
        milestone_data = {
            'GM': [  # Gross Motor - 3 milestones per age range
                # 0-6 months
                ("Head Control", "Raises head and chest when lying on stomach"),
                ("Rolling to Side", "Rolls from back to side"),
                ("Early Sitting", "Sits with support"),
                # 6-12 months
                ("Independent Sitting", "Sits without support"),
                ("Crawling", "Crawls on hands and knees"),
                ("Pulling to Stand", "Pulls to stand using furniture"),
                # 12-18 months
                ("Walking", "Walks independently"),
                ("Squatting", "Squats to pick up toys"),
                ("Climbing Stairs", "Climbs stairs with support"),
                # 18-24 months
                ("Running", "Runs with increasing control"),
                ("Kicking Ball", "Kicks ball forward"),
                ("Jumping", "Jumps with both feet"),
                # 24-30 months
                ("Balance", "Stands on one foot briefly"),
                ("Throwing", "Throws ball overhand"),
                ("Pedaling", "Pedals tricycle"),
                # 30-36 months
                ("Hopping", "Hops on one foot"),
                ("Climbing", "Climbs playground equipment"),
                ("Complex Movement", "Navigates obstacles while running")
            ],
            'FM': [  # Fine Motor - 3 milestones per age range
                # 0-6 months
                ("Grasping", "Grasps and holds objects placed in hand"),
                ("Reaching", "Reaches and grasps objects"),
                ("Transferring", "Transfers objects from hand to hand"),
                # 6-12 months
                ("Pincer Grasp", "Uses thumb and index finger to pick up small objects"),
                ("Finger Feeding", "Uses fingers to feed self"),
                ("Object Manipulation", "Bangs objects together"),
                # 12-18 months
                ("Stacking", "Stacks 2-3 blocks"),
                ("Container Play", "Places objects in container"),
                ("Page Turning", "Turns pages in book"),
                # 18-24 months
                ("Scribbling", "Makes marks on paper"),
                ("Tool Use", "Uses spoon or fork"),
                ("Block Building", "Builds tower of 4-6 blocks"),
                # 24-30 months
                ("Drawing Lines", "Copies straight lines"),
                ("Snipping", "Snips with scissors"),
                ("Shape Sorting", "Matches shapes in sorter"),
                # 30-36 months
                ("Drawing Circles", "Copies circular shapes"),
                ("Stringing Beads", "Strings large beads"),
                ("Complex Building", "Builds complex block structures")
            ],
            'ADL': [  # Activities of Daily Living - 3 milestones per age range
                # 0-6 months
                ("Feeding Response", "Opens mouth for spoon"),
                ("Sucking", "Coordinates sucking and swallowing"),
                ("Hand to Mouth", "Brings hands to mouth"),
                # 6-12 months
                ("Self-Feeding", "Finger feeds self"),
                ("Bottle Holding", "Holds own bottle"),
                ("Cup Drinking", "Drinks from cup with help"),
                # 12-18 months
                ("Spoon Use", "Uses spoon with some spilling"),
                ("Removing Clothes", "Removes simple items"),
                ("Washing Hands", "Participates in hand washing"),
                # 18-24 months
                ("Independent Feeding", "Uses utensils with less spilling"),
                ("Dressing Help", "Helps with dressing"),
                ("Tooth Brushing", "Allows tooth brushing"),
                # 24-30 months
                ("Undressing", "Removes most clothing items"),
                ("Hand Washing", "Washes hands independently"),
                ("Potty Training", "Shows interest in toilet training"),
                # 30-36 months
                ("Dressing", "Puts on simple clothing"),
                ("Grooming", "Brushes teeth with help"),
                ("Toileting", "Uses toilet with minimal help")
            ],
            'RL': [  # Receptive Language - 3 milestones per age range
                # 0-6 months
                ("Sound Response", "Turns to sounds and voices"),
                ("Name Recognition", "Responds to own name"),
                ("Comfort Words", "Responds to soothing words"),
                # 6-12 months
                ("Simple Commands", "Responds to 'no' and 'bye-bye'"),
                ("Object Recognition", "Recognizes familiar objects when named"),
                ("Gesture Understanding", "Understands simple gestures"),
                # 12-18 months
                ("Body Parts", "Points to 1-2 body parts"),
                ("Action Words", "Understands simple action words"),
                ("Object Location", "Understands 'in' and 'on'"),
                # 18-24 months
                ("Two-Step Commands", "Follows two related commands"),
                ("Picture Recognition", "Points to pictures in books"),
                ("Possession", "Understands 'mine' and 'yours'"),
                # 24-30 months
                ("Complex Commands", "Follows two unrelated commands"),
                ("Categories", "Understands basic categories"),
                ("Prepositions", "Understands more prepositions"),
                # 30-36 months
                ("Sequence", "Understands before/after concepts"),
                ("Questions", "Understands who/what/where questions"),
                ("Time Concepts", "Understands basic time concepts")
            ],
            'EL': [  # Expressive Language - 3 milestones per age range
                # 0-6 months
                ("Vocalization", "Makes different sounds"),
                ("Cooing", "Coos and laughs"),
                ("Babbling", "Begins consonant sounds"),
                # 6-12 months
                ("Varied Babbling", "Uses varied consonant sounds"),
                ("Jargon", "Uses speech-like babbling"),
                ("First Words", "Says 1-2 words"),
                # 12-18 months
                ("Vocabulary", "Uses 5-10 words"),
                ("Word Approximations", "Attempts to imitate words"),
                ("Naming", "Names familiar objects"),
                # 18-24 months
                ("Two-Word Phrases", "Combines two words"),
                ("Questions", "Uses what and where"),
                ("Expanded Vocabulary", "Uses 20+ words"),
                # 24-30 months
                ("Three-Word Phrases", "Uses three-word sentences"),
                ("Pronouns", "Uses me, you, mine"),
                ("Action Words", "Uses action words"),
                # 30-36 months
                ("Complex Sentences", "Uses 4+ word sentences"),
                ("Past Tense", "Uses past tense"),
                ("Conversation", "Engages in simple conversations")
            ],
            'COG': [  # Cognitive - 3 milestones per age range
                # 0-6 months
                ("Visual Tracking", "Follows moving objects"),
                ("Object Awareness", "Looks for dropped objects"),
                ("Exploration", "Explores objects with mouth"),
                # 6-12 months
                ("Object Permanence", "Finds partially hidden objects"),
                ("Cause & Effect", "Repeats actions for results"),
                ("Problem Solving", "Uses objects as tools"),
                # 12-18 months
                ("Imitation", "Imitates simple actions"),
                ("Matching", "Matches similar objects"),
                ("Memory", "Remembers hidden objects"),
                # 18-24 months
                ("Pretend Play", "Begins simple pretend play"),
                ("Sorting", "Sorts by basic categories"),
                ("Building", "Uses objects appropriately"),
                # 24-30 months
                ("Complex Play", "Sequences pretend play"),
                ("Puzzles", "Completes simple puzzles"),
                ("Counting", "Recites some numbers"),
                # 30-36 months
                ("Symbolic Play", "Uses objects symbolically"),
                ("Colors", "Names some colors"),
                ("Problem Solving", "Solves simple problems")
            ],
            'SOC': [  # Social - 3 milestones per age range
                # 0-6 months
                ("Social Smile", "Smiles responsively"),
                ("Social Interest", "Shows interest in faces"),
                ("Social Play", "Enjoys social games"),
                # 6-12 months
                ("Social Games", "Plays peek-a-boo"),
                ("Social Anticipation", "Anticipates familiar games"),
                ("Social Gestures", "Waves bye-bye"),
                # 12-18 months
                ("Peer Interest", "Shows interest in peers"),
                ("Sharing", "Gives objects to others"),
                ("Social Referencing", "Looks to others for reactions"),
                # 18-24 months
                ("Parallel Play", "Plays near other children"),
                ("Imitative Play", "Imitates others in play"),
                ("Social Greetings", "Uses greetings"),
                # 24-30 months
                ("Group Play", "Joins others in play"),
                ("Turn Taking", "Takes turns with support"),
                ("Helping", "Helps with simple tasks"),
                # 30-36 months
                ("Cooperative Play", "Plays cooperatively"),
                ("Friendship", "Shows preference for friends"),
                ("Social Rules", "Follows simple social rules")
            ],
            'EMO': [  # Emotional - 3 milestones per age range
                # 0-6 months
                ("Emotional Expression", "Shows distinct emotions"),
                ("Self-Soothing", "Uses simple self-soothing"),
                ("Emotional Response", "Responds to others' emotions"),
                # 6-12 months
                ("Attachment", "Shows clear preferences"),
                ("Fear Response", "Shows stranger anxiety"),
                ("Emotional Regulation", "Calms with support"),
                # 12-18 months
                ("Self-Awareness", "Shows self-awareness"),
                ("Emotional Range", "Shows varied emotions"),
                ("Comfort Seeking", "Seeks comfort when upset"),
                # 18-24 months
                ("Independence", "Shows independence"),
                ("Empathy", "Shows concern for others"),
                ("Emotional Expression", "Expresses emotions verbally"),
                # 24-30 months
                ("Self-Control", "Shows beginning self-control"),
                ("Pride", "Shows pride in accomplishments"),
                ("Emotional Understanding", "Labels basic emotions"),
                # 30-36 months
                ("Complex Emotions", "Shows complex emotions"),
                ("Coping Strategies", "Uses simple coping strategies"),
                ("Emotional Regulation", "Manages emotions better")
            ]
        }
        
        # Process all milestones more efficiently
        milestones = []
        for domain, behaviors in milestone_data.items():
            age_index = 0
            for i, (behavior, criteria) in enumerate(behaviors):
                # Calculate age range index more efficiently
                if i > 0 and i % 3 == 0:  # 3 milestones per age range
                    age_index += 1
                
                # Stay within bounds
                if age_index >= len(age_ranges):
                    break
                    
                age_range = age_ranges[age_index]
                keywords = self._generate_keywords(behavior, criteria)
                scoring_rules = self._get_scoring_rules(behavior)
                
                milestones.append(DevelopmentalMilestone(
                    behavior=behavior,
                    criteria=criteria,
                    age_range=age_range,
                    domain=domain,
                    keywords=keywords,
                    scoring_rules=scoring_rules
                ))
                
        return milestones
    
    # Compile regex patterns once for better performance
    _word_pattern = re.compile(r'\b\w+\b')
    _special_chars_pattern = re.compile(r'[^\w\s]')
    
    def _generate_keywords(self, behavior: str, criteria: str) -> List[str]:
        """Generate keywords for milestone behavior"""
        # Combine behavior and criteria for keyword extraction
        text = f"{behavior} {criteria}"
        
        # Simple keyword extraction (can be enhanced with NLP techniques)
        keywords = set()
        # Remove punctuation and convert to lowercase
        for word in re.sub(r'[^\w\s]', ' ', text.lower()).split():
            if len(word) > 3 and word not in STOPWORDS:  # Skip short words and stopwords
                keywords.add(word)
        
        return list(keywords)
        
    @functools.lru_cache(maxsize=128)
    def _get_scoring_rules(self, behavior: str) -> Dict[str, Score]:
        """Retrieve scoring rules for a behavior (with caching for performance)"""
        # Default scoring rules
        scoring_rules = {
            "cannot": Score.CANNOT_DO,
            "lost": Score.LOST_SKILL,
            "emerging": Score.EMERGING,
            "support": Score.WITH_SUPPORT,
            "independent": Score.INDEPENDENT
        }
        
        # This could be extended to have behavior-specific scoring rules
        return scoring_rules

    def set_child_age(self, age: int):
        """Set child's age and initialize appropriate milestones"""
        self.child_age = age
        self.current_milestone_index = 0
        
        # Filter and sort milestones based on age
        self.active_milestones = []
        for milestone in self.milestones:
            # Extract age range numbers
            age_range = milestone.age_range
            start_age = int(age_range.split('-')[0].strip())
            end_age = int(age_range.split('-')[1].split()[0].strip())
            
            # Include all milestones from 0 months up to and including current age range
            if end_age <= age:
                # Include all past milestones
                self.active_milestones.append(milestone)
            elif start_age <= age <= end_age:
                # Include current age range milestones
                self.active_milestones.append(milestone)
        
        # Sort milestones chronologically by age range first, then by domain
        # This ensures we assess earlier milestones before later ones
        self.active_milestones.sort(key=lambda x: (
            # Extract start age for sorting
            int(x.age_range.split('-')[0].strip()),
            # Then sort by domain
            x.domain
        ))
        
        # Initialize tracking set for asked milestones to prevent repetition
        self.asked_milestones = set()

    def get_next_milestone(self) -> Optional[DevelopmentalMilestone]:
        """Get the next milestone to assess"""
        # Initialize attributes if they don't exist
        if not hasattr(self, 'active_milestones') or not self.active_milestones:
            # If active_milestones doesn't exist, use milestones
            if not hasattr(self, 'milestones') or not self.milestones:
                return None
            self.active_milestones = self.milestones
        
        if not hasattr(self, 'current_milestone_index'):
            self.current_milestone_index = 0
        
        if not hasattr(self, 'asked_milestones'):
            self.asked_milestones = set()
        
        if not hasattr(self, 'assessed_milestones'):
            self.assessed_milestones = set()
        
        # Check if we've gone through all milestones
        if self.current_milestone_index >= len(self.active_milestones):
            return None
        
        # Get the current domain we're assessing
        current_domain = None
        if self.current_milestone_index > 0 and self.current_milestone_index <= len(self.active_milestones):
            previous_milestone = self.active_milestones[self.current_milestone_index - 1]
            current_domain = previous_milestone.domain
        
        # If we have a current domain, try to find the next milestone in the same domain
        if current_domain:
            # Check if there are any unassessed milestones in the current domain
            domain_milestones = []
            for i, milestone in enumerate(self.active_milestones):
                if milestone.domain == current_domain and milestone.behavior not in self.assessed_milestones:
                    domain_milestones.append((i, milestone))
            
            # If we found unassessed milestones in the current domain, use the first one
            if domain_milestones:
                index, next_milestone = domain_milestones[0]
                self.current_milestone_index = index  # Update the index to point to this milestone
                
                # Mark this milestone as asked
                self.asked_milestones.add(next_milestone.behavior)
                
                return next_milestone
        
        # If we don't have a current domain or all milestones in the domain are assessed,
        # proceed with the original logic
        
        # Get the next milestone
        next_milestone = self.active_milestones[self.current_milestone_index]
        
        # Create a unique identifier for this milestone
        milestone_id = next_milestone.behavior
        
        # Check if we've already asked about this milestone
        while milestone_id in self.asked_milestones:
            # Move to the next milestone
            self.current_milestone_index += 1
            
            # Check if we've gone through all milestones
            if self.current_milestone_index >= len(self.active_milestones):
                return None
            
            # Get the next milestone
            next_milestone = self.active_milestones[self.current_milestone_index]
            milestone_id = next_milestone.behavior
        
        # Mark this milestone as asked
        self.asked_milestones.add(milestone_id)
        
        return next_milestone
    
    async def analyze_response_embeddings(self, response: str, milestone: DevelopmentalMilestone) -> Score:
        """
        Analyze a response using sentence embeddings for more nuanced scoring
        
        This method computes embeddings for the response and compares it with
        pre-computed embeddings for each score category to find the best match.
        """
        if not self.use_embeddings:
            # Fall back to keyword-based scoring if embeddings not available
            return self.analyze_response_keywords(response, milestone)
        
        # Generate embedding for the response
        with torch.no_grad():
            response_embedding = self.embedding_model.encode(response, convert_to_tensor=True)
        
        # Calculate cosine similarity with each score embedding
        similarities = {}
        for score, embedding in self.score_embeddings.items():
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                response_embedding.unsqueeze(0), 
                embedding.unsqueeze(0)
            ).item()
            similarities[score] = similarity
        
        # If all similarities are very low, return NOT_RATED
        if all(sim < 0.3 for sim in similarities.values()):
            return Score.NOT_RATED
        
        # Find the score with highest similarity
        best_score = max(similarities.items(), key=lambda x: x[1])[0]
        return best_score
        
    def analyze_response_keywords(self, response: str, milestone: DevelopmentalMilestone) -> Score:
        """Analyze a response using keyword-based scoring (fallback method)"""
        # Get the milestone key for caching
        milestone_key = self._get_milestone_key(milestone)
        print(f"Analyzing response for milestone: {milestone.behavior} with key: {milestone_key}")
        
        # Check if we already have keywords cached for this milestone's scoring rules
        if milestone_key not in self._scoring_keywords_cache:
            print(f"No cached keywords found for milestone: {milestone.behavior}. Initializing new keyword map.")
            # Initialize the keyword mapping
            keyword_map = {}
            for key, score in milestone.scoring_rules.items():
                # Generate variations of keywords for better matching
                variations = [
                    key,
                    f"{key}s",
                    f"{key}es",
                    f"{key}ing",
                    f"{key}ed"
                ]
                for variation in variations:
                    keyword_map[variation] = score
            
            # Additional common phrases mapped to scores
            phrase_map = {
                # CANNOT_DO (0) - Skill not acquired
                "not able to": Score.CANNOT_DO,
                "can't": Score.CANNOT_DO,
                "cannot": Score.CANNOT_DO,
                "doesn't": Score.CANNOT_DO,
                "does not": Score.CANNOT_DO,
                "never": Score.CANNOT_DO,
                "not at all": Score.CANNOT_DO,
                "not yet": Score.CANNOT_DO,
                "no": Score.CANNOT_DO,
                
                # LOST_SKILL (1) - Acquired but lost
                "used to": Score.LOST_SKILL,
                "no longer": Score.LOST_SKILL,
                "stopped": Score.LOST_SKILL,
                "lost the ability": Score.LOST_SKILL,
                "regressed": Score.LOST_SKILL,
                "forgotten": Score.LOST_SKILL,
                "not anymore": Score.LOST_SKILL,
                
                # EMERGING (2) - Emerging and inconsistent
                "sometimes": Score.EMERGING,
                "occasionally": Score.EMERGING,
                "trying to": Score.EMERGING,
                "beginning to": Score.EMERGING,
                "starting to": Score.EMERGING,
                "inconsistent": Score.EMERGING,
                "not consistent": Score.EMERGING,
                "varies": Score.EMERGING,
                "some days": Score.EMERGING,
                
                # WITH_SUPPORT (3) - Acquired but consistent in specific situations only
                "with help": Score.WITH_SUPPORT,
                "when assisted": Score.WITH_SUPPORT,
                "with support": Score.WITH_SUPPORT,
                "with guidance": Score.WITH_SUPPORT,
                "needs help": Score.WITH_SUPPORT,
                "when prompted": Score.WITH_SUPPORT,
                "specific situations": Score.WITH_SUPPORT,
                "certain contexts": Score.WITH_SUPPORT,
                
                # INDEPENDENT (4) - Acquired and present in all situations
                "independently": Score.INDEPENDENT,
                "by themselves": Score.INDEPENDENT,
                "on their own": Score.INDEPENDENT,
                "without help": Score.INDEPENDENT,
                "always": Score.INDEPENDENT,
                "consistently": Score.INDEPENDENT,
                "in all situations": Score.INDEPENDENT,
                "mastered": Score.INDEPENDENT,
                "yes": Score.INDEPENDENT
            }
            
            # Combine both maps
            keyword_map.update(phrase_map)
            
            # Cache the result
            self._scoring_keywords_cache[milestone_key] = keyword_map
            print(f"Created and cached keyword map with {len(keyword_map)} entries for {milestone_key}")
        else:
            print(f"Using cached keyword map for milestone: {milestone.behavior}")
        
        # Get the keyword mapping from the cache
        keyword_map = self._scoring_keywords_cache[milestone_key]
        print(f"Keyword map has {len(keyword_map)} entries")
        
        # Process response
        response_lower = response.lower()
        print(f"Analyzing response: '{response_lower}'")
        
        # Count occurrences of each scoring keyword/phrase
        score_counts = {score: 0 for score in Score}
        
        # Track matched keywords for debugging
        matched_keywords = {score.name: [] for score in Score}
        
        # Split the response into words for better matching
        words = response_lower.split()
        
        for keyword, score in keyword_map.items():
            keyword_lower = keyword.lower()
            
            # Check for exact matches (whole words or phrases)
            if keyword_lower in response_lower:
                score_counts[score] += 1
                matched_keywords[score.name].append(keyword)
                continue
            
            # Check for word matches (for single words)
            if len(keyword_lower.split()) == 1 and keyword_lower in words:
                score_counts[score] += 1
                matched_keywords[score.name].append(keyword)
                continue
            
            # Check for substring matches in words (for partial matches)
            for word in words:
                if len(word) > 3 and len(keyword_lower) > 3 and keyword_lower in word:
                    score_counts[score] += 1
                    matched_keywords[score.name].append(f"{keyword} (in {word})")
                    break
        
        # Print matched keywords for debugging
        for score_name, keywords in matched_keywords.items():
            if keywords:
                print(f"Matched {score_name} keywords: {', '.join(keywords)}")
        
        # If no scores were found, default to NOT_RATED
        if all(count == 0 for count in score_counts.values()):
            print(f"No keywords matched in response for milestone: {milestone.behavior}")
            return Score.NOT_RATED
        
        # Find the score with the highest count
        max_score = max(score_counts.items(), key=lambda x: x[1])[0]
        
        # If the highest count is 0, return NOT_RATED
        if score_counts[max_score] == 0:
            print(f"No keywords matched in response for milestone: {milestone.behavior}")
            return Score.NOT_RATED
        
        print(f"Selected score {max_score.name} with count {score_counts[max_score]} for milestone: {milestone.behavior}")
        return max_score
        
    async def analyze_response(self, response: str, milestone: DevelopmentalMilestone) -> Score:
        """
        Analyze a response to determine the appropriate score for a milestone
        
        This uses embedding-based scoring if available, otherwise falls back to keywords
        """
        if self.use_embeddings:
            return await self.analyze_response_embeddings(response, milestone)
        else:
            return self.analyze_response_keywords(response, milestone)
    
    async def batch_analyze_responses(self, responses: List[Tuple[str, DevelopmentalMilestone]]) -> List[Score]:
        """
        Analyze multiple responses in parallel for better performance
        
        Args:
            responses: List of (response, milestone) tuples to analyze
            
        Returns:
            List of Score enums corresponding to each response
        """
        tasks = [self.analyze_response(response, milestone) for response, milestone in responses]
        return await asyncio.gather(*tasks)
    
    def set_milestone_score(self, milestone: DevelopmentalMilestone, score: Score):
        """Set the score for a milestone"""
        if not isinstance(score, Score):
            try:
                score = Score(score)
            except ValueError:
                score = Score.NOT_RATED
        
        self.scores[milestone.behavior] = score
        self.assessed_milestones.add(milestone.behavior)
        print(f"Scored {milestone.behavior} ({milestone.domain}): {score.name}")
        
        return score
    
    def _get_milestone_key(self, milestone: DevelopmentalMilestone) -> str:
        """Generate a unique key for a milestone"""
        return f"{milestone.domain}_{milestone.behavior}_{milestone.age_range}"
    
    def get_milestone_score(self, milestone: DevelopmentalMilestone) -> Optional[Score]:
        """Get score for a milestone"""
        return self.scores.get(milestone.behavior)
    
    def generate_report(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate assessment report with detailed scoring"""
        data = []
        
        # Group milestones by domain for organized reporting
        domain_milestones = {}
        for milestone in self.milestones:
            if milestone.domain not in domain_milestones:
                domain_milestones[milestone.domain] = []
            domain_milestones[milestone.domain].append(milestone)
        
        # Create report data with domain organization
        for domain, milestones in domain_milestones.items():
            for milestone in milestones:
                score = self.get_milestone_score(milestone)
                data.append({
                    'Domain': domain,
                    'Age Range': milestone.age_range,
                    'Behavior': milestone.behavior,
                    'Criteria': milestone.criteria,
                    'Score': score.value if score else Score.NOT_RATED.value,
                    'Score Label': score.name if score else Score.NOT_RATED.name
                })
        
        df = pd.DataFrame(data)
        
        # Calculate domain quotients with weighted scoring
        domain_quotients = {}
        for domain in df['Domain'].unique():
            domain_data = df[df['Domain'] == domain]
            if not domain_data.empty:
                # Calculate weighted average based on age appropriateness
                total_weight = 0
                weighted_sum = 0
                for _, row in domain_data.iterrows():
                    age_range = row['Age Range']
                    start_age = int(age_range.split('-')[0].strip())
                    end_age = int(age_range.split('-')[1].split()[0].strip())
                    
                    # Weight is higher for age-appropriate milestones
                    if self.child_age is not None:
                        if start_age <= self.child_age <= end_age:
                            weight = 1.0
                        else:
                            weight = 0.5
                        
                        score = row['Score']
                        if score != Score.NOT_RATED.value:
                            weighted_sum += score * weight
                            total_weight += weight
                
                if total_weight > 0:
                    avg_score = weighted_sum / total_weight
                    domain_quotients[domain] = (avg_score / Score.INDEPENDENT.value) * 100
                else:
                    domain_quotients[domain] = 0
        
        return df, domain_quotients
    
    def zero_shot_classify(self, response: str, milestone: DevelopmentalMilestone) -> Score:
        """
        Use zero-shot learning to classify a response without explicit training
        
        This method uses the embedding model to compare the response against
        descriptions of each score category, providing a more flexible scoring approach.
        """
        if not self.use_embeddings:
            return self.analyze_response_keywords(response, milestone)
        
        # Define template descriptions for each score category
        score_descriptions = {
            Score.CANNOT_DO: f"Child cannot perform the skill: {milestone.behavior}. {milestone.criteria}",
            Score.LOST_SKILL: f"Child used to be able to {milestone.behavior.lower()} but has lost this skill.",
            Score.EMERGING: f"Child is beginning to show the skill: {milestone.behavior}. Occasionally {milestone.criteria.lower()}",
            Score.WITH_SUPPORT: f"Child can {milestone.behavior.lower()} with support or assistance. {milestone.criteria} with help.",
            Score.INDEPENDENT: f"Child can independently {milestone.behavior.lower()}. Consistently {milestone.criteria.lower()} without help."
        }
        
        # Compute embeddings for the response and each score description
        with torch.no_grad():
            response_embedding = self.embedding_model.encode(response, convert_to_tensor=True)
            
            # Calculate similarity with each score description
            similarities = {}
            for score, description in score_descriptions.items():
                description_embedding = self.embedding_model.encode(description, convert_to_tensor=True)
                similarity = torch.nn.functional.cosine_similarity(
                    response_embedding.unsqueeze(0),
                    description_embedding.unsqueeze(0)
                ).item()
                similarities[score] = similarity
        
        # If all similarities are very low, return NOT_RATED
        if all(sim < 0.3 for sim in similarities.values()):
            return Score.NOT_RATED
            
        # Return the score with the highest similarity
        return max(similarities.items(), key=lambda x: x[1])[0]

    def find_milestone_by_name(self, name):
        """Find a milestone by name, using fuzzy matching if exact match not found"""
        # First try exact match (case-insensitive)
        for milestone in self.milestones:
            if milestone.behavior.lower() == name.lower():
                return milestone
                
        # If no exact match, try fuzzy matching using NLP embeddings
        if hasattr(self, 'embedding_model') and self.embedding_model:
            try:
                print(f"Using embedding model to find milestone '{name}'")
                # Get embedding for the queried name
                query_embedding = self.embedding_model.encode(name, convert_to_tensor=True)
                
                # Compare with embeddings of all milestone behaviors
                behavior_embeddings = self.embedding_model.encode([m.behavior for m in self.milestones], convert_to_tensor=True)
                
                # Calculate cosine similarities using torch
                query_embedding = query_embedding.unsqueeze(0)  # Add batch dimension
                behavior_embeddings = behavior_embeddings.unsqueeze(0)  # Add batch dimension
                
                # Calculate cosine similarity
                similarities = torch.nn.functional.cosine_similarity(
                    query_embedding, behavior_embeddings, dim=2
                )[0]
                
                # Find the most similar milestone if similarity is above threshold
                best_idx = torch.argmax(similarities).item()
                best_similarity = similarities[best_idx].item()
                
                print(f"Looking for milestone '{name}', best match: '{self.milestones[best_idx].behavior}' with similarity {best_similarity:.2f}")
                
                if best_similarity > 0.7:  # Set an appropriate threshold
                    return self.milestones[best_idx]
            except Exception as e:
                print(f"Error in fuzzy milestone matching: {str(e)}")
                
        # No match found above threshold
        return None 

    def score_response(self, milestone_behavior, response_text):
        """
        Score a response for a specific milestone behavior
        
        Args:
            milestone_behavior: The behavior to score
            response_text: The caregiver's response text
            
        Returns:
            Score: The score for the response
        """
        # Find the milestone
        milestone = None
        for m in self.milestones:
            if m.behavior == milestone_behavior:
                milestone = m
                break
                
        if not milestone:
            # Try to find by fuzzy matching
            milestone = self.find_milestone_by_name(milestone_behavior)
            
        if not milestone:
            print(f"Milestone '{milestone_behavior}' not found")
            return Score.NOT_RATED
        
        # Try to use the advanced NLP module for response analysis if available
        try:
            # Import dynamically to avoid circular imports
            import importlib
            try:
                advanced_nlp_module = importlib.import_module('advanced_nlp')
                if hasattr(advanced_nlp_module, 'AdvancedResponseAnalyzer'):
                    analyzer = advanced_nlp_module.AdvancedResponseAnalyzer()
                    
                    # Analyze the response using advanced NLP
                    analysis_result = analyzer.analyze_response(milestone_behavior, response_text, milestone.domain)
                    
                    # Extract the score from the analysis result
                    if analysis_result and 'score' in analysis_result:
                        score_value = analysis_result['score']
                        
                        # Convert numeric score to Score enum
                        score_mapping = {
                            0: Score.CANNOT_DO,
                            1: Score.LOST_SKILL,
                            2: Score.EMERGING,
                            3: Score.WITH_SUPPORT,
                            4: Score.INDEPENDENT,
                            -1: Score.NOT_RATED
                        }
                        
                        if score_value in score_mapping:
                            print(f"Advanced NLP scored response '{response_text}' for milestone '{milestone_behavior}' as {score_mapping[score_value].name}")
                            return score_mapping[score_value]
            except (ImportError, AttributeError, Exception) as e:
                print(f"Advanced NLP module not available or error during analysis: {str(e)}")
                # Continue with regular scoring if advanced NLP fails
        except Exception as e:
            print(f"Error using advanced NLP: {str(e)}")
            # Continue with regular scoring
        
        # First, try keyword-based scoring using the cache
        response_lower = response_text.lower()
        score = self.analyze_response_keywords(response_text, milestone)
        
        # If we got a valid score (not NOT_RATED), return it
        if score != Score.NOT_RATED:
            return score
        
        # Fallback scoring logic for common patterns if keyword scoring didn't match
        # CANNOT_DO (0) patterns
        if re.search(r"\b(no|not),?\s+(yet|yet started|started yet)", response_lower) or \
           re.search(r"not at all", response_lower) or \
           re.search(r"\bnever\b", response_lower) or \
           re.search(r"(doesn't|does not|can't|cannot|hasn't|has not)\s+([a-z]+\s){0,3}(do|show|perform|demonstrate)", response_lower) or \
           response_lower.strip() == "no":
            print(f"Fallback pattern detected CANNOT_DO for response: '{response_text}'")
            return Score.CANNOT_DO
            
        # LOST_SKILL (1) patterns
        if re.search(r"used to", response_lower) or \
           re.search(r"(was able to|could before|previously|before but)", response_lower) or \
           re.search(r"(lost|regressed|stopped|no longer|not anymore)", response_lower):
            print(f"Fallback pattern detected LOST_SKILL for response: '{response_text}'")
            return Score.LOST_SKILL
            
        # EMERGING (2) patterns
        if re.search(r"\b(sometimes|occasionally)\b", response_lower) or \
           re.search(r"not (always|consistently)", response_lower) or \
           re.search(r"(trying|beginning|starting|learning) to", response_lower) or \
           re.search(r"(inconsistent|developing|in progress)", response_lower):
            print(f"Fallback pattern detected EMERGING for response: '{response_text}'")
            return Score.EMERGING
            
        # WITH_SUPPORT (3) patterns
        if re.search(r"with (help|support|assistance)", response_lower) or \
           re.search(r"when (prompted|reminded|guided|helped|assisted)", response_lower) or \
           re.search(r"needs (help|support|assistance|prompting|reminding)", response_lower) or \
           re.search(r"(if i help|if we help|if someone helps)", response_lower):
            print(f"Fallback pattern detected WITH_SUPPORT for response: '{response_text}'")
            return Score.WITH_SUPPORT
            
        # Default for positive responses - INDEPENDENT (4)
        if re.search(r"\b(yes|yeah|yep|sure|absolutely|definitely|always|consistently)\b", response_lower) or \
           re.search(r"(does|can|is able to|performs|demonstrates)", response_lower) or \
           re.search(r"(mastered|achieved|accomplished)", response_lower):
            print(f"Fallback pattern detected INDEPENDENT for response: '{response_text}'")
            return Score.INDEPENDENT
            
        # If nothing matched, default to NOT_RATED
        print(f"No pattern matched for response: '{response_text}'. Defaulting to NOT_RATED")
        return Score.NOT_RATED 

    def get_all_milestones(self) -> List[Dict[str, Any]]:
        """
        Get all available milestones
        
        Returns:
            List of milestone dictionaries
        """
        milestones = []
        
        for milestone in self.milestones:
            milestones.append({
                "behavior": milestone.behavior,
                "criteria": milestone.criteria,
                "domain": milestone.domain,
                "age_range": milestone.age_range
            })
            
        return milestones 