"""
Improved Developmental Scoring Engine Module

This module implements the main improved scoring engine that orchestrates
all components of the scoring system.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import time
import os
import csv

from .base import Score, ScoringResult, BaseScorer, EnsembleScorer
from .keyword_scorer import KeywordBasedScorer
from .embedding_scorer import SemanticEmbeddingScorer
from .transformer_scorer import TransformerBasedScorer
from .confidence_tracker import ConfidenceTracker
from .audit_logger import AuditLogger
from .continuous_learning import ContinuousLearningEngine
from .dynamic_ensemble import DynamicEnsembleScorer
from .keyword_manager import KeywordManager
from .component_specialization import (
    SpecializedScorer, KeywordSpecializedScorer, 
    EmbeddingSpecializedScorer, TransformerSpecializedScorer,
    LLMSpecializedScorer, analyze_response_features
)

# Import LLM-based scorer if available
try:
    from .llm_scorer import LLMBasedScorer
    LLM_SCORER_AVAILABLE = True
except ImportError:
    LLM_SCORER_AVAILABLE = False

# Define Milestone class to be used throughout the engine
class Milestone:
    """Class representing a developmental milestone."""
    def __init__(self):
        self.behavior = ""
        self.criteria = ""
        self.domain = ""
        self.age_range = ""
        self.keywords = []
        
    def __str__(self):
        return f"{self.behavior} ({self.domain}, {self.age_range})"

class ImprovedDevelopmentalScoringEngine:
    """
    Main engine for developmental milestone scoring
    
    This implementation:
    1. Uses a modular architecture for composable scoring components
    2. Combines multiple scoring approaches for robust predictions
    3. Tracks confidence and uncertainty for quality control
    4. Provides detailed explanations and reasoning
    5. Enables continuous learning and improvement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the scoring engine"""
        self.config = config or self._default_config()
        
        # Configure logging
        self.logger = logging.getLogger("improved_scoring_engine")
        self.logger.setLevel(logging.INFO)
        
        # Initialize keyword manager
        self.keyword_manager = KeywordManager()
        
        # Ensure score_weights is initialized
        if "score_weights" not in self.config:
            self.config["score_weights"] = {
                "keyword": 0.6,
                "embedding": 0.4,
                "transformer": 0.3,
                "llm": 0.2
            }
        
        # Initialize component scorers
        self._init_scorers()
        
        # Initialize supporting systems
        self.confidence_tracker = ConfidenceTracker(self.config.get("confidence_tracker", {}))
        self.audit_logger = AuditLogger(self.config.get("audit_logger", {}))
        
        # Initialize learning engine if enabled
        if self.config.get("enable_continuous_learning", True):
            self.learning_engine = ContinuousLearningEngine(
                self.config.get("continuous_learning", {})
            )
        else:
            self.learning_engine = None
        
        # Load milestone data
        self.milestone_data = self._load_milestone_data()
        
        # Store scores
        self.scores = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the scoring engine"""
        return {
            # Component enablement
            "enable_keyword_scorer": True,
            "enable_embedding_scorer": True,
            "enable_transformer_scorer": False,  # Disabled by default due to resource requirements
            "enable_llm_scorer": False,  # Disabled by default due to cost
            
            # Confidence thresholds
            "high_confidence_threshold": 0.8,
            "low_confidence_threshold": 0.5,
            "minimum_confidence": 0.3,
            
            # Weights for ensemble scoring
            "keyword_weight": 1.0,
            "embedding_weight": 1.0,
            "transformer_weight": 1.5,
            "llm_weight": 2.0,
            
            # Tiered approach configuration
            "use_tiered_approach": True,  # Whether to use tiered scoring approach
            "fast_scorer_agreement_threshold": 0.7,  # Threshold for fast scorer agreement
            
            # Continuous learning
            "enable_continuous_learning": False,  # Whether to enable continuous learning
            "review_threshold": 0.6,  # Confidence threshold for requesting review
            
            # Component specialization
            "enable_component_specialization": False,  # Whether to enable component specialization
            
            # Audit logging
            "enable_audit_logging": True,
            "audit_log_path": "logs/scoring_audit.log",
            "audit_log_level": "INFO",
            
            # Performance tracking
            "track_performance": True,
            "performance_metrics_path": "data/performance_metrics.json"
        }
    
    def _init_scorers(self) -> None:
        """Initialize component scorers"""
        self._scorers = {}
        
        # Initialize keyword-based scorer if enabled
        if self.config["enable_keyword_scorer"]:
            try:
                self._scorers["keyword"] = KeywordBasedScorer(
                    self.config.get("keyword_scorer", {})
                )
                self.logger.info("KeywordBasedScorer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize KeywordBasedScorer: {str(e)}")
        
        # Initialize embedding-based scorer if enabled
        if self.config["enable_embedding_scorer"]:
            try:
                self._scorers["embedding"] = SemanticEmbeddingScorer(
                    self.config.get("embedding_scorer", {})
                )
                self.logger.info("SemanticEmbeddingScorer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize SemanticEmbeddingScorer: {str(e)}")
        
        # Initialize transformer-based scorer if enabled
        if self.config["enable_transformer_scorer"]:
            try:
                self._scorers["transformer"] = TransformerBasedScorer(
                    self.config.get("transformer_scorer", {})
                )
                self.logger.info("TransformerBasedScorer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize TransformerBasedScorer: {str(e)}")
        
        # Initialize LLM-based scorer if enabled and available
        if self.config.get("enable_llm_scorer", False) and LLM_SCORER_AVAILABLE:
            try:
                self._scorers["llm"] = LLMBasedScorer(
                    self.config.get("llm_scorer", {})
                )
                # Check if the model was actually loaded
                if hasattr(self._scorers["llm"], "model") and self._scorers["llm"].model is not None:
                    self.logger.info("LLMBasedScorer initialized successfully")
                else:
                    self.logger.warning("LLMBasedScorer initialized but model failed to load")
                    del self._scorers["llm"]
            except Exception as e:
                self.logger.error(f"Failed to initialize LLMBasedScorer: {str(e)}")
        
        if not self._scorers:
            self.logger.warning("No scoring components were successfully initialized")
        
        # Initialize ensemble scorer
        self._init_ensemble()
    
    def _init_ensemble(self) -> None:
        """Initialize the ensemble scorer"""
        if not self._scorers:
            raise ValueError("No scorers enabled. At least one scorer must be enabled.")
        
        # Extract weights from config
        weights = []
        component_scorers = []
        
        for name, scorer in self._scorers.items():
            component_scorers.append(scorer)
            weight = self.config.get("score_weights", {}).get(name, 1.0)
            weights.append(weight)
        
        # Create ensemble scorer - use dynamic ensemble if enabled
        if self.config.get("enable_dynamic_ensemble", True):
            self.logger.info("Initializing DynamicEnsembleScorer")
            dynamic_config = self.config.get("dynamic_ensemble", {})
            self.ensemble = DynamicEnsembleScorer(
                scorers=component_scorers,
                weights=weights,
                config=dynamic_config
            )
        else:
            self.logger.info("Initializing standard EnsembleScorer")
            self.ensemble = EnsembleScorer(
                scorers=component_scorers,
                weights=weights
            )
    
    def _load_milestone_data(self) -> Dict[str, Any]:
        """
        Load milestone data from configured sources
        
        Returns:
            Dictionary of milestone data
        """
        self.logger.info("Loading milestone data")
        
        # This is a placeholder implementation
        # In a real implementation, this would load data from CSV files, databases, etc.
        milestone_data = {
            "milestones": [],
            "domains": ["GM", "FM", "RL", "EL", "SL", "SE"],
            "age_ranges": ["0-3", "3-6", "6-9", "9-12", "12-18", "18-24", "24-30", "30-36"]
        }
        
        self.logger.info(f"Loaded milestone data with {len(milestone_data['domains'])} domains")
        return milestone_data
    
    def score_response(self, 
                      response: str, 
                      milestone_context: Optional[Dict[str, Any]] = None,
                      detailed: bool = False) -> Union[Score, ScoringResult, Dict[str, Any]]:
        """
        Score a response given milestone context
        
        Args:
            response: The text response to score
            milestone_context: Optional context about the milestone
            detailed: Whether to return detailed results
            
        Returns:
            Score enum, ScoringResult, or detailed dictionary
        """
        if not milestone_context:
            milestone_context = {}
            
        start_time = time.time()
        all_results = []
        
        try:
            # TIER 1: Try LLM scoring first if enabled (most accurate but expensive)
            if "llm" in self._scorers and self.config["enable_llm_scorer"]:
                try:
                    self.logger.info(f"Attempting LLM scoring first (highest accuracy)")
                    llm_result = self._scorers["llm"].score(response, milestone_context)
                    all_results.append(llm_result)
                    
                    # If LLM gives a confident result, use it directly
                    if llm_result.score != Score.NOT_RATED and llm_result.confidence >= 0.7:
                        self.logger.info(f"Using high-confidence LLM result: {llm_result.score.name} (confidence: {llm_result.confidence:.2f})")
                        scoring_time = time.time() - start_time
                        
                        # Return early with the LLM result
                        if detailed:
                            return self._create_detailed_result(llm_result, [llm_result], scoring_time)
                        else:
                            return llm_result
                except Exception as e:
                    self.logger.error(f"Error in LLM scorer: {str(e)}")
            
            # TIER 2: Try keyword scoring if no high-confidence LLM result
            if "keyword" in self._scorers and self.config["enable_keyword_scorer"]:
                try:
                    self.logger.info(f"Trying keyword scoring (reliable for clear patterns)")
                    keyword_result = self._scorers["keyword"].score(response, milestone_context)
                    all_results.append(keyword_result)
                    
                    # If keyword gives a confident result, use it directly
                    if keyword_result.score != Score.NOT_RATED and keyword_result.confidence >= 0.85:
                        self.logger.info(f"Using high-confidence keyword result: {keyword_result.score.name} (confidence: {keyword_result.confidence:.2f})")
                        scoring_time = time.time() - start_time
                        
                        # Return early with the keyword result
                        if detailed:
                            return self._create_detailed_result(keyword_result, [keyword_result], scoring_time)
                        else:
                            return keyword_result
                except Exception as e:
                    self.logger.error(f"Error in keyword scorer: {str(e)}")
            
            # TIER 3: Try transformer-based scoring if still no high-confidence result
            if "transformer" in self._scorers and self.config["enable_transformer_scorer"]:
                try:
                    self.logger.info(f"Using transformer-based scoring for nuanced analysis")
                    transformer_result = self._scorers["transformer"].score(response, milestone_context)
                    all_results.append(transformer_result)
                    
                    # If transformer gives a confident result, use it
                    if transformer_result.score != Score.NOT_RATED and transformer_result.confidence >= 0.8:
                        self.logger.info(f"Using high-confidence transformer result: {transformer_result.score.name} (confidence: {transformer_result.confidence:.2f})")
                        scoring_time = time.time() - start_time
                        
                        # Return early with the transformer result
                        if detailed:
                            return self._create_detailed_result(transformer_result, [transformer_result], scoring_time)
                        else:
                            return transformer_result
                except Exception as e:
                    self.logger.error(f"Error in transformer scorer: {str(e)}")
            
            # TIER 4: Try embedding scoring as last resort
            if len(all_results) < 2 and "embedding" in self._scorers and self.config["enable_embedding_scorer"]:
                try:
                    self.logger.info(f"Using embedding scoring as additional signal")
                    embedding_result = self._scorers["embedding"].score(response, milestone_context)
                    all_results.append(embedding_result)
                except Exception as e:
                    self.logger.error(f"Error in embedding scorer: {str(e)}")
            
            # Step 5: If we have results, combine them using weighted ensemble
            if all_results:
                self.logger.info(f"Combining {len(all_results)} scoring results using weighted ensemble")
                final_result = self._combine_results(all_results)
                
                # Record scoring in audit log
                scoring_time = time.time() - start_time
                self.logger.info(f"Scoring completed in {scoring_time:.2f} seconds")
                
                # Log the scoring decision
                if self.audit_logger:
                    needs_review = False
                    if self.confidence_tracker:
                        needs_review = self.confidence_tracker.should_request_review(final_result)
                    
                    self.audit_logger.record_scoring(
                        response=response,
                        milestone=milestone_context,
                        score=final_result,
                        needs_review=needs_review
                    )
                    
                    # Log to standard logger
                    confidence_str = f"{final_result.confidence:.2f}" if final_result.confidence is not None else "N/A"
                    review_str = " - Needs review" if needs_review else ""
                    self.logger.info(f"Scored: {final_result.score.name} (confidence: {confidence_str}){review_str}")
                    
                    # Queue for review if needed and continuous learning is enabled
                    if needs_review and self.learning_engine and self.config["enable_continuous_learning"]:
                        review_id = self.learning_engine.queue_for_review(
                            response=response,
                            milestone_context=milestone_context,
                            predicted_score=final_result
                        )
                        if review_id:
                            self.logger.info(f"Queued response for review: {review_id}")
                
                # Return appropriate format based on detailed flag
                if detailed:
                    return self._create_detailed_result(final_result, all_results, scoring_time)
                else:
                    return final_result
            else:
                # If no scorers were successful, return NOT_RATED
                self.logger.warning(f"No scoring methods were successful, returning NOT_RATED")
                error_result = ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=0.0,
                    method="none",
                    reasoning="No scoring methods were successful"
                )
                
                if detailed:
                    return self._create_detailed_result(error_result, [], time.time() - start_time)
                else:
                    return error_result
                
        except Exception as e:
            # Log error
            if self.audit_logger:
                self.audit_logger.record_error(
                    error_message=str(e),
                    context={
                        "response": response,
                        "milestone": milestone_context
                    }
                )
            
            self.logger.error(f"Error scoring response: {str(e)}")
            
            # Return NOT_RATED with error information
            error_result = ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="error",
                reasoning=f"Error: {str(e)}"
            )
            
            if detailed:
                return self._create_detailed_result(error_result, [], time.time() - start_time)
            else:
                return error_result
    
    def with_expert_feedback(self, 
                            response: str, 
                            milestone_context: Dict[str, Any],
                            correct_score: Score,
                            notes: Optional[str] = None) -> None:
        """
        Provide expert feedback for a response
        
        Args:
            response: The text response
            milestone_context: Context about the milestone
            correct_score: The correct score according to expert
            notes: Optional expert notes
        """
        # Score the response to get a prediction
        result = self.score_response(response, milestone_context)
        
        # Update confidence tracker with feedback
        if isinstance(result, ScoringResult):
            was_correct = result.score == correct_score
            self.confidence_tracker.update_with_expert_feedback(result, was_correct)
        
        # Log the expert feedback
        self.audit_logger.record_scoring(
            response=response,
            milestone=milestone_context,
            score=correct_score,
            expert_feedback={
                "score": correct_score.name,
                "notes": notes
            }
        )
        
        # Add to learning engine if available
        if self.learning_engine:
            # Create a synthetic review ID
            import hashlib
            import time
            
            review_id = f"manual_{hashlib.md5(f'{response}_{time.time()}'.encode()).hexdigest()[:10]}"
            
            # Queue for review
            self.learning_engine.queue_for_review(
                response=response,
                milestone_context=milestone_context,
                predicted_score=result
            )
            
            # Add feedback (this will add to training examples)
            self.learning_engine.add_expert_feedback(
                review_id=review_id,
                correct_score=correct_score,
                expert_notes=notes
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the scoring engine"""
        if not self.config.get("track_performance", True):
            return {"tracking_disabled": True}
        
        # Placeholder for actual metrics
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.79,
            "f1_score": 0.80,
            "confidence_calibration": 0.92,
            "samples_processed": 1250,
            "samples_reviewed": 75,
            "improvement_rate": 0.03
        }
    
    def get_pending_reviews(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get responses pending expert review"""
        # Placeholder implementation
        return []
    
    def find_milestone_by_name(self, milestone_behavior: str) -> Optional[Dict[str, Any]]:
        """
        Find a milestone by its behavior name.
        
        Args:
            milestone_behavior: The milestone behavior to find
        
        Returns:
            The milestone data if found, None otherwise
        """
        # First, check if the milestone data is loaded
        if hasattr(self, 'milestone_data') and self.milestone_data:
            # Search through loaded milestone data
            for domain, milestones in self.milestone_data.items():
                for milestone in milestones:
                    # Check if milestone is a dict or an object
                    if isinstance(milestone, dict):
                        # Dictionary milestone
                        if milestone.get('behavior') and milestone.get('behavior').lower() == milestone_behavior.lower():
                            return milestone
                    elif hasattr(milestone, 'behavior'):
                        # Object milestone
                        if milestone.behavior and milestone.behavior.lower() == milestone_behavior.lower():
                            return milestone
        
        # If we get here, either milestone data isn't loaded or the milestone wasn't found
        # Create a Milestone with the data we have
        try:
            # For backward compatibility, still handle the 'Walks independently' case explicitly
            if milestone_behavior == "Walks independently":
                milestone = Milestone()
                milestone.behavior = "Walks independently"
                milestone.domain = "GM"
                milestone.age_range = "12-18"
                milestone.criteria = "Child walks without holding on to a person or object for support"
                return milestone
            
            # For any other milestone, search the available CSV data directly
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'CDDC' in f and 'Table 1' in f]
            
            for csv_file in csv_files:
                file_path = os.path.join(data_dir, csv_file)
                domain = csv_file.split('CDDC')[1].split('-')[0].strip()
                
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    current_age_range = ''
                    
                    for row in reader:
                        # Get the field that contains the milestone behavior
                        behavior_field = None
                        age_range_field = None
                        criteria_field = None
                        
                        for field in row.keys():
                            if 'Age' in field:
                                age_range_field = field
                            elif 'Criteria' in field:
                                criteria_field = field
                            elif field not in ['SI.No', 'train'] and 'Age' not in field and 'Criteria' not in field:
                                behavior_field = field
                        
                        if not behavior_field:
                            continue
                        
                        # Update the current age range if this row has one
                        if age_range_field and row[age_range_field] and row[age_range_field] != 'Age':
                            current_age_range = row[age_range_field].replace('months', '').replace('m', '').strip()
                        
                        # Check if this is the milestone we're looking for
                        if behavior_field and row[behavior_field] and row[behavior_field].lower() == milestone_behavior.lower():
                            milestone = Milestone()
                            milestone.behavior = row[behavior_field]
                            milestone.domain = domain
                            milestone.age_range = current_age_range
                            milestone.criteria = row.get(criteria_field, '')
                            return milestone
            
            # If we still haven't found the milestone, log a warning and return None
            self.logger.warning(f"Milestone not found: {milestone_behavior}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error finding milestone by name: {str(e)}")
            return None
    
    def get_all_milestones(self) -> List[Dict[str, Any]]:
        """
        Get all available milestones
        
        Returns:
            List of milestone dictionaries
        """
        milestones = []
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
        
        try:
            # Get all CSV files in the data directory
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'CDDC' in f and 'Table 1' in f]
            
            if not csv_files:
                self.logger.warning("No CSV files found in data/ directory, returning placeholder milestone")
                # Return placeholder milestone if no CSV files found
                milestone = Milestone()
                milestone.behavior = "Walks independently"
                milestone.criteria = "Child walks without holding on to a person or object for support"
                milestone.domain = "GM"
                milestone.age_range = "12-18"
                return [milestone]
            
            # Process each CSV file (each represents a domain)
            for csv_file in csv_files:
                # Extract domain from filename (e.g., "CDDC GM-Table 1.csv" -> "GM")
                domain = None
                if 'CDDC' in csv_file and '-' in csv_file:
                    parts = csv_file.split(' ')
                    if len(parts) > 1:
                        domain = parts[1].split('-')[0]
                
                if not domain:
                    self.logger.warning(f"Could not extract domain from filename {csv_file}, skipping")
                    continue
                    
                self.logger.info(f"Processing domain: {domain} from file: {csv_file}")
                
                # Load data from CSV
                file_path = os.path.join(data_dir, csv_file)
                try:
                    with open(file_path, 'r') as f:
                        # Special handling for Emo domain which has a different format
                        if domain == "Emo":
                            reader = csv.reader(f)
                            current_age_range = ''
                            
                            for row in reader:
                                if len(row) < 3:  # Skip rows without enough columns
                                    continue
                                    
                                # Check if this row has an age range
                                if row[1] and ('m' in row[1] or 'months' in row[1] or '-' in row[1]):
                                    current_age_range = row[1].replace('months', '').replace('m', '').strip()
                                
                                # Skip if no age range has been set or behavior is empty
                                if not current_age_range or not row[2]:
                                    continue
                                
                                behavior = row[2].strip()
                                
                                # Get criteria if available, otherwise use behavior as criteria
                                criteria = ""
                                if len(row) > 3 and row[3]:
                                    criteria = row[3].strip()
                                else:
                                    criteria = behavior
                                
                                # Create milestone and add to list
                                milestone = Milestone()
                                milestone.behavior = behavior
                                milestone.criteria = criteria
                                milestone.domain = domain
                                milestone.age_range = current_age_range
                                milestones.append(milestone)
                                self.logger.info(f"Added milestone: {behavior} ({domain}, {current_age_range})")
                        else:
                            reader = csv.DictReader(f)
                            current_age_range = ''
                            
                            for row in reader:
                                # Get the field that contains the milestone behavior
                                behavior_field = None
                                age_range_field = None
                                criteria_field = None
                                
                                for field in row.keys():
                                    if 'Age' in field:
                                        age_range_field = field
                                    elif 'Criteria' in field:
                                        criteria_field = field
                                    elif field not in ['SI.No', 'train'] and 'Age' not in field and 'Criteria' not in field:
                                        behavior_field = field
                                
                                if not behavior_field:
                                    continue
                                
                                # Update the current age range if this row has one
                                if age_range_field and row[age_range_field] and row[age_range_field] != 'Age':
                                    current_age_range = row[age_range_field].replace('months', '').replace('m', '').strip()
                                
                                # Skip if no age range has been set or behavior is empty
                                if not current_age_range or not row[behavior_field]:
                                    continue
                                
                                behavior = row[behavior_field].strip()
                                
                                # Get criteria if available, otherwise use behavior as criteria
                                criteria = ""
                                if criteria_field and row.get(criteria_field):
                                    criteria = row[criteria_field].strip()
                                else:
                                    criteria = behavior
                                
                                # Create milestone and add to list
                                milestone = Milestone()
                                milestone.behavior = behavior
                                milestone.criteria = criteria
                                milestone.domain = domain
                                milestone.age_range = current_age_range
                                milestones.append(milestone)
                                self.logger.info(f"Added milestone: {behavior} ({domain}, {current_age_range})")
                
                except Exception as e:
                    self.logger.error(f"Error processing CSV file {csv_file}: {str(e)}")
            
            # If no milestones were loaded from CSV files, return placeholder
            if not milestones:
                self.logger.warning("No valid milestones found in CSV files, returning placeholder milestone")
                milestone = Milestone()
                milestone.behavior = "Walks independently"
                milestone.criteria = "Child walks without holding on to a person or object for support"
                milestone.domain = "GM"
                milestone.age_range = "12-18"
                return [milestone]
                
            self.logger.info(f"Successfully loaded {len(milestones)} milestones from {len(csv_files)} CSV files")
            return milestones
                
        except Exception as e:
            self.logger.error(f"Error loading milestone data from CSV files: {str(e)}")
            # Return placeholder milestone if error occurs
            milestone = Milestone()
            milestone.behavior = "Walks independently"
            milestone.criteria = "Child walks without holding on to a person or object for support"
            milestone.domain = "GM"
            milestone.age_range = "12-18"
            return [milestone]
    
    def _get_milestone_key(self, milestone) -> str:
        """
        Generate a unique key for a milestone
        
        Args:
            milestone: The milestone object
            
        Returns:
            A unique string key
        """
        return f"{milestone.domain}_{milestone.behavior.replace(' ', '_').lower()}"
    
    async def analyze_response(self, response: str, milestone) -> Score:
        """
        Analyze a response for a specific milestone
        
        Args:
            response: The parent/caregiver response text
            milestone: The milestone object
            
        Returns:
            Score enum representing the assessment result
        """
        self.logger.info(f"Analyzing response for milestone: {milestone.behavior}")
        
        # Create milestone context dictionary
        milestone_context = {
            "behavior": milestone.behavior,
            "criteria": milestone.criteria,
            "domain": milestone.domain,
            "age_range": milestone.age_range
        }
        
        # Use our scoring method
        result = self.score_response(response, milestone_context)
        
        # If it's a detailed result, extract the score
        if isinstance(result, dict) and "score" in result:
            score_value = result["score"]
            # Convert to Score enum if it's a numeric value
            if isinstance(score_value, int):
                for score in Score:
                    if score.value == score_value:
                        return score
                # Default to NOT_RATED if no match
                return Score.NOT_RATED
            # If it's already a Score enum, return it
            elif isinstance(score_value, Score):
                return score_value
            # Default case
            return Score.NOT_RATED
        # If it's a ScoringResult, extract the score
        elif isinstance(result, ScoringResult):
            return result.score
        # If it's already a Score enum, return it
        elif isinstance(result, Score):
            return result
        # Default case
        return Score.NOT_RATED
    
    def set_milestone_score(self, milestone, score: Score) -> None:
        """
        Set the score for a milestone
        
        Args:
            milestone: The milestone object
            score: The Score enum value
        """
        milestone_key = self._get_milestone_key(milestone)
        self.logger.info(f"Setting score for milestone {milestone_key}: {score.name}")
        
        # In a real implementation, this would store the score in a database or other persistent storage
        # For now, we'll just log it
        self.logger.info(f"Score set: {milestone.behavior} ({milestone.domain}) = {score.name}")
    
    def _combine_results(self, results: List[ScoringResult]) -> ScoringResult:
        """
        Combine multiple scoring results into a single result
        
        Args:
            results: List of scoring results from different components
            
        Returns:
            ScoringResult: Combined result
        """
        if not results:
            # If no results, return EMERGING with medium confidence as a default
            return ScoringResult(
                score=Score.EMERGING,
                confidence=0.5,
                method="default",
                reasoning="No scoring results available"
            )
        
        # Filter out NOT_RATED scores if we have other scores
        valid_results = [r for r in results if r.score != Score.NOT_RATED]
        
        # If all scores are NOT_RATED, use them all
        if not valid_results and results:
            valid_results = results
        
        # Calculate weighted scores for each category
        category_scores = {s: 0.0 for s in Score}
        
        # Get weights from config, ensuring it exists
        if "score_weights" not in self.config:
            self.config["score_weights"] = {
                "keyword": 0.6,
                "embedding": 0.4,
                "transformer": 0.3,
                "llm": 0.2
            }
        
        weights = {}
        for result in valid_results:
            method = result.method.lower()
            if "keyword" in method:
                weights[method] = self.config["score_weights"].get("keyword", 0.2)
            elif "embedding" in method:
                weights[method] = self.config["score_weights"].get("embedding", 0.4)
            elif "transformer" in method:
                weights[method] = self.config["score_weights"].get("transformer", 0.3)
            elif "llm" in method:
                weights[method] = self.config["score_weights"].get("llm", 0.8)
            else:
                weights[method] = 0.5  # Default weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            normalized_weights = {k: 1.0/len(weights) for k in weights}
        
        # Apply weighted voting
        for result in valid_results:
            weight = normalized_weights.get(result.method.lower(), 1.0/len(valid_results))
            category_scores[result.score] += result.confidence * weight
        
        # Select best score and confidence
        best_score, best_score_value = max(category_scores.items(), key=lambda x: x[1])
        
        # If best score is NOT_RATED but we have other scores with some confidence,
        # choose the next best score
        if best_score == Score.NOT_RATED and len(valid_results) > 0:
            # Remove NOT_RATED and find next best
            del category_scores[Score.NOT_RATED]
            if category_scores:
                best_score, best_score_value = max(category_scores.items(), key=lambda x: x[1])
        
        # Calculate overall confidence based on agreement and individual confidences
        confidence = best_score_value if best_score_value > 0 else 0.4
        
        # Combine reasoning from all components
        reasoning_parts = []
        for result in valid_results:
            if result.reasoning:
                reasoning_parts.append(f"{result.method}: {result.reasoning}")
        
        combined_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Combined multiple scoring methods"
        
        return ScoringResult(
            score=best_score,
            confidence=confidence,
            method="ensemble",
            reasoning=combined_reasoning,
            details={
                "component_results": [r.to_dict() for r in results],
                "category_scores": {k.name: v for k, v in category_scores.items()}
            }
        )

    def _create_detailed_result(self, result: ScoringResult, component_results: List[ScoringResult], elapsed_time: float) -> Dict[str, Any]:
        """
        Create a detailed result dictionary from a ScoringResult and component results
        
        Args:
            result: The final ScoringResult
            component_results: List of ScoringResult objects from individual components
            elapsed_time: Time taken to score the response
            
        Returns:
            Dict: Detailed result dictionary
        """
        return {
            "score": result.score,
            "score_name": result.score.name,
            "score_value": result.score.value,
            "confidence": result.confidence,
            "method": result.method,
            "reasoning": result.reasoning,
            "component_results": [r.to_dict() for r in component_results],
            "elapsed_time": elapsed_time
        } 