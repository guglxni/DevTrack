"""
R2R Enhanced Active Learning

This module enhances the active learning capabilities with R2R (Reason to Retrieve)
for more effective knowledge retrieval, feedback processing, and model improvement.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import uuid

from src.core.scoring.base import Score
from src.core.scoring.active_learning import ActiveLearningEngine
from src.core.retrieval.r2r_client import R2RClient

# Configure logging
logger = logging.getLogger(__name__)

class R2RActiveLearningSystems(ActiveLearningEngine):
    """
    Active Learning System integrated with Reason-to-Retrieve (R2R) for enhanced
    developmental milestone assessment.
    
    This system manages the collection, prioritization, and expert review of
    assessment responses to continuously improve the scoring model.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the R2R enhanced active learning system.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        self.r2r_config = self.config.get("r2r_config")
        self.r2r_client = R2RClient(self.r2r_config)
        self._setup_collections()
        logger.info("R2R Active Learning System initialized")
        
    def _default_config(self) -> Dict[str, Any]:
        """
        Default configuration for the R2R active learning system.
        
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        # Get the base config from parent class
        config = super()._default_config()
        
        # Add R2R specific configuration
        config.update({
            "r2r_config": None,  # Will use default R2R client config
            "use_retrieval_for_prioritization": True,
            "expert_feedback_collection": "scoring_examples",
            "research_collection": "developmental_research",
            "model_history_collection": "model_versions",
            "use_contextual_reasoning": True,
            "semantic_similarity_threshold": 0.75,
            "store_expert_feedback": True,
            "max_context_length": 5000
        })
        
        return config
        
    def _setup_collections(self) -> None:
        """Set up the R2R collections needed for active learning"""
        try:
            # Ensure all required collections exist
            collections = [
                ("expert_feedback_collection", "Collection for storing expert feedback"),
                ("research_collection", "Collection for developmental research materials"),
                ("model_history_collection", "Collection for model version history")
            ]
            
            for collection_key, description in collections:
                collection_name = self.config[collection_key]
                # Check if we need to add some initial content to collections
                if collection_key == "research_collection":
                    self._setup_research_collection(collection_name)
                    
            logger.info("R2R collections set up successfully")
        except Exception as e:
            logger.error(f"Failed to set up R2R collections: {str(e)}")
            
    def _setup_research_collection(self, collection_name: str) -> None:
        """
        Set up the research collection with initial content.
        
        Args:
            collection_name: Name of the research collection
        """
        # Check if the collection already has documents
        results = self.r2r_client.search(
            query="developmental milestone assessment",
            collection_key="developmental_research",
            limit=1
        )
        
        if not results:
            # Add some initial research documents
            initial_docs = [
                {
                    "title": "Developmental Milestone Assessment Framework",
                    "content": """
                    Developmental milestones are behaviors or physical skills seen in infants 
                    and children as they grow and develop. Rolling over, crawling, walking, 
                    and talking are all considered milestones. The milestones are different 
                    for each age range.
                    
                    Scoring categories:
                    - CANNOT_DO (0): Child cannot demonstrate the skill at all despite appropriate opportunity.
                    - LOST_SKILL (1): Child previously demonstrated the skill but has since lost it.
                    - EMERGING (2): Child shows beginning signs of the skill but is inconsistent or requires significant support.
                    - WITH_SUPPORT (3): Child can demonstrate the skill with minimal help or prompting.
                    - INDEPENDENT (4): Child consistently demonstrates the skill independently.
                    """,
                    "domain": "general",
                    "type": "framework"
                },
                {
                    "title": "Motor Skills Assessment",
                    "content": """
                    Motor development refers to the development of a child's bones, muscles and ability to move around and manipulate their environment.
                    
                    Motor development can be divided into two sections:
                    - Gross Motor: Development of skills involving the whole body, such as sitting, crawling, or walking
                    - Fine Motor: Development of skills involving smaller movements, such as grasping, manipulation of objects, or finger dexterity
                    
                    Key considerations for assessment:
                    - Physical opportunity to practice the skill
                    - Environmental factors that may limit demonstrations
                    - Physical constraints or medical conditions
                    - Age-appropriate expectations
                    """,
                    "domain": "MOTOR",
                    "type": "domain_guidance"
                },
                {
                    "title": "Communication Skills Assessment",
                    "content": """
                    Communication development involves both receptive language (understanding) and expressive language (speaking).
                    
                    Key areas to assess:
                    - Receptive language: Ability to understand words and language
                    - Expressive language: Using words, gestures, or communication devices
                    - Pragmatic language: Social use of language in different contexts
                    - Speech development: Pronunciation and articulation
                    
                    Key considerations for assessment:
                    - Bilingual or multilingual environments
                    - Cultural differences in communication patterns
                    - Use of non-verbal communication methods
                    - Consistency across different settings
                    """,
                    "domain": "COMMUNICATION",
                    "type": "domain_guidance"
                }
            ]
            
            for doc in initial_docs:
                metadata = {
                    "title": doc["title"],
                    "domain": doc["domain"],
                    "type": doc["type"],
                    "date_added": datetime.now().isoformat()
                }
                
                self.r2r_client.ingest_document(
                    document=doc["content"],
                    collection_key="developmental_research",
                    metadata=metadata
                )
                
            logger.info(f"Added {len(initial_docs)} initial documents to research collection")
            
    def _calculate_semantic_novelty(self, response: Dict[str, Any]) -> float:
        """
        Calculate semantic novelty score for a response using R2R.
        
        Args:
            response: Response dictionary
            
        Returns:
            float: Semantic novelty score (0.0-1.0)
        """
        try:
            if not self.config["use_retrieval_for_prioritization"]:
                # Fallback to basic calculation if retrieval not enabled
                return super()._calculate_linguistic_novelty(response)
                
            # Extract the response text
            response_text = response.get("response", "")
            milestone_context = response.get("milestone_context", {})
            
            # Create a search query
            domain = milestone_context.get("domain", "")
            behavior = milestone_context.get("behavior", "")
            
            query = f"{response_text} {behavior} {domain} developmental assessment"
            
            # Search for similar examples in expert feedback collection
            results = self.r2r_client.search(
                query=query,
                collection_key=self.config["expert_feedback_collection"],
                limit=5
            )
            
            if not results:
                # No similar examples found, high novelty
                return 1.0
                
            # Calculate similarity score
            similarity_scores = []
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Create a TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            
            # Prepare texts for comparison
            texts = [response_text] + [result["text"] for result in results]
            
            try:
                # Transform texts to TF-IDF representations
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # Calculate cosine similarity between the response and each result
                for i in range(1, len(texts)):
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[i:i+1])[0][0]
                    similarity_scores.append(similarity)
                    
                # Calculate novelty as inverse of maximum similarity
                if similarity_scores:
                    max_similarity = max(similarity_scores)
                    novelty = 1.0 - max_similarity
                    return max(0.0, min(1.0, novelty))
                else:
                    return 0.8
            except Exception as e:
                logger.error(f"Error calculating similarity: {str(e)}")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating semantic novelty: {str(e)}")
            return 0.5
            
    def queue_with_enhanced_priority(self, 
                                   response: str, 
                                   milestone_context: Dict[str, Any], 
                                   predicted_score: Score, 
                                   confidence: float,
                                   component_scores: Optional[Dict[str, Any]] = None) -> str:
        """
        Queue a response for expert review with enhanced priority scoring.
        
        Args:
            response: The response text
            milestone_context: Milestone context
            predicted_score: Predicted score
            confidence: Confidence value
            component_scores: Optional component scores
            
        Returns:
            str: Review ID
        """
        # Create review item
        review_item = {
            "id": str(uuid.uuid4()),
            "response": response,
            "milestone_context": milestone_context,
            "predicted_score": predicted_score.name,
            "predicted_score_value": predicted_score.value,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "component_scores": component_scores or {}
        }
        
        # Calculate priority using R2R enhanced methods
        disagreement = self._calculate_component_disagreement(review_item)
        uncertainty = self._calculate_uncertainty_score(review_item)
        
        # Use R2R for semantic novelty calculation
        novelty = self._calculate_semantic_novelty(review_item)
        
        # Standard domain coverage calculation
        domain_coverage = self._calculate_domain_coverage_score(review_item)
        
        # Calculate final priority score
        priority = self._calculate_information_gain(
            uncertainty, disagreement, novelty, domain_coverage
        )
        
        # Add priority to review item
        review_item["priority"] = priority
        
        # Add to review queue
        review_queue = self._load_review_queue()
        review_queue.append(review_item)
        
        # Sort by priority (descending)
        review_queue.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        # Save queue
        self._save_review_queue()
        
        logger.info(f"Queued response for review with priority {priority:.2f}, ID: {review_item['id']}")
        
        return review_item["id"]
        
    def add_expert_feedback_with_context(self, 
                                       review_id: str, 
                                       correct_score: Score, 
                                       expert_notes: Optional[str] = None,
                                       store_in_r2r: bool = True) -> bool:
        """
        Add expert feedback with additional contextual information.
        
        Args:
            review_id: The review ID
            correct_score: The correct score according to the expert
            expert_notes: Optional notes from the expert
            store_in_r2r: Whether to store feedback in R2R
            
        Returns:
            bool: Success status
        """
        # Call the parent method first
        success = super().add_expert_feedback(review_id, correct_score, expert_notes)
        
        if success and store_in_r2r and self.config["store_expert_feedback"]:
            # Find the review item
            review_queue = self._load_review_queue()
            review_item = None
            
            for item in review_queue:
                if item.get("id") == review_id:
                    review_item = item
                    break
                    
            if review_item:
                try:
                    # Prepare document content
                    milestone = review_item.get("milestone_context", {})
                    behavior = milestone.get("behavior", "")
                    domain = milestone.get("domain", "")
                    
                    document_content = {
                        "response": review_item.get("response", ""),
                        "milestone_behavior": behavior,
                        "domain": domain,
                        "predicted_score": review_item.get("predicted_score", ""),
                        "predicted_score_value": review_item.get("predicted_score_value", -1),
                        "correct_score": correct_score.name,
                        "correct_score_value": correct_score.value,
                        "expert_notes": expert_notes or "",
                        "review_id": review_id
                    }
                    
                    # Prepare metadata
                    metadata = {
                        "domain": domain,
                        "score_category": correct_score.name,
                        "review_id": review_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Ingest into R2R
                    self.r2r_client.ingest_document(
                        document=document_content,
                        collection_key=self.config["expert_feedback_collection"],
                        metadata=metadata
                    )
                    
                    logger.info(f"Stored expert feedback in R2R for review {review_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to store expert feedback in R2R: {str(e)}")
                    
        return success
        
    def get_similar_examples(self, 
                           response: str, 
                           milestone_context: Optional[Dict[str, Any]] = None, 
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar examples to a response from the feedback collection.
        
        Args:
            response: The response text
            milestone_context: Optional milestone context
            limit: Maximum number of examples to retrieve
            
        Returns:
            List[Dict[str, Any]]: Similar examples
        """
        try:
            # Build query from response and milestone context
            query_parts = [response]
            
            if milestone_context:
                if "behavior" in milestone_context:
                    query_parts.append(milestone_context["behavior"])
                    
                if "domain" in milestone_context:
                    query_parts.append(f"domain:{milestone_context['domain']}")
                    
            query = " ".join(query_parts)
            
            # Get filters based on milestone context
            filters = None
            if milestone_context and "domain" in milestone_context:
                filters = {"domain": milestone_context["domain"]}
                
            # Search for similar examples
            results = self.r2r_client.search(
                query=query,
                collection_key=self.config["expert_feedback_collection"],
                limit=limit,
                filter_criteria=filters
            )
            
            # Process and format results
            examples = []
            for result in results:
                # Parse the document content
                if isinstance(result["text"], str):
                    try:
                        content = json.loads(result["text"])
                    except:
                        content = {"response": result["text"]}
                else:
                    content = result["text"]
                    
                # Create a formatted example
                example = {
                    "response": content.get("response", ""),
                    "milestone_behavior": content.get("milestone_behavior", ""),
                    "domain": content.get("domain", ""),
                    "correct_score": content.get("correct_score", ""),
                    "correct_score_value": content.get("correct_score_value", -1),
                    "expert_notes": content.get("expert_notes", ""),
                    "id": content.get("review_id", result.get("id", ""))
                }
                
                examples.append(example)
                
            return examples
            
        except Exception as e:
            logger.error(f"Error retrieving similar examples: {str(e)}")
            return []
            
    def get_contextual_information(self, 
                                 query: str, 
                                 domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get contextual information from the research collection.
        
        Args:
            query: The search query
            domain: Optional domain to filter results
            
        Returns:
            List[Dict[str, Any]]: Research information
        """
        try:
            # Prepare filters
            filters = None
            if domain:
                filters = {"domain": domain}
                
            # Search the research collection
            results = self.r2r_client.search(
                query=query,
                collection_key=self.config["research_collection"],
                limit=5,
                filter_criteria=filters
            )
            
            # Format results
            info = []
            for result in results:
                item = {
                    "content": result["text"],
                    "title": result.get("metadata", {}).get("title", "Research Information"),
                    "domain": result.get("metadata", {}).get("domain", "general"),
                    "id": result.get("id", "")
                }
                
                info.append(item)
                
            return info
            
        except Exception as e:
            logger.error(f"Error retrieving contextual information: {str(e)}")
            return []
            
    def store_model_version_data(self, version_data: Dict[str, Any]) -> bool:
        """
        Store model version data in the R2R collection.
        
        Args:
            version_data: Model version data
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare metadata
            metadata = {
                "version": version_data.get("version", "unknown"),
                "timestamp": version_data.get("timestamp", datetime.now().isoformat()),
                "type": "model_version"
            }
            
            # Store in R2R
            self.r2r_client.ingest_document(
                document=version_data,
                collection_key=self.config["model_history_collection"],
                metadata=metadata
            )
            
            logger.info(f"Stored model version {version_data.get('version', 'unknown')} in R2R")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store model version in R2R: {str(e)}")
            return False
            
    def _increment_model_version(self, description: str = None) -> Dict[str, Any]:
        """
        Increment model version and store in R2R.
        
        Args:
            description: Optional description of the version
            
        Returns:
            Dict[str, Any]: Updated version info
        """
        # Call parent method to increment version
        version_info = super()._increment_model_version(description)
        
        # Store in R2R
        self.store_model_version_data(version_info)
        
        return version_info
        
    def get_model_version_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the history of model versions with their metrics and descriptions.
        
        Args:
            limit (int): Maximum number of versions to return
            
        Returns:
            List[Dict[str, Any]]: List of model version data
        """
        versions = self._load_model_versions()
        return versions[:limit]

# Add compatibility alias for backward compatibility
R2RActiveLearningSystem = R2RActiveLearningSystems  # Alias for backward compatibility 