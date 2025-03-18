"""
R2R Client Module

Integrates the R2R (Reason to Retrieve) system from SciPhi-AI for enhanced
retrieval capabilities in the developmental assessment API.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import tempfile
import uuid

# Use llama_cpp for local model
from llama_cpp import Llama

# Import R2R and its types
import r2r
from r2r import R2RClient as OriginalR2RClient
from r2r import Document, Message, MessageType, SearchSettings, SearchMode, GenerationConfig

# Configure logging
logger = logging.getLogger(__name__)

class R2RClient:
    """
    Client for interacting with the R2R (Reason to Retrieve) service.
    This implementation supports both remote R2R service and local LLM models.
    """

    def __init__(
        self,
        llm_provider: str = "local",
        llm_config: Optional[Dict[str, Any]] = None,
        data_dir: str = "data/documents",
        r2r_base_url: Optional[str] = None,
        timeout: float = 300.0
    ):
        """
        Initialize the R2R client.

        Args:
            llm_provider: The LLM provider to use ('local', 'mistral', etc.)
            llm_config: Configuration for the LLM provider
            data_dir: Directory where documents are stored
            r2r_base_url: Base URL for the R2R service (if using remote service)
            timeout: Request timeout in seconds (for remote service)
        """
        self.llm_provider = llm_provider
        self.llm_config = llm_config or {}
        self.data_dir = data_dir
        self.timeout = timeout
        self.r2r_base_url = r2r_base_url
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize model if using local provider
        self.model = None
        if llm_provider == "local":
            self._init_local_model()
        
        # Initialize R2R client
        try:
            self.client = OriginalR2RClient(
                base_url=self.r2r_base_url,
                timeout=self.timeout
            )
            logger.info(f"Initialized R2R client with base URL: {self.r2r_base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize R2R client: {e}")
            self.client = None
            
        # Initialize default collections
        self.collections = {
            "milestone_data": "milestone_collection",
            "assessment_feedback": "assessment_feedback_collection",
            "clinical_guidelines": "clinical_guidelines_collection",
            "research_papers": "research_papers_collection"
        }

    def _init_local_model(self):
        """Initialize the local LLM model."""
        model_path = self.llm_config.get("model_path", "models/mistral-7b-instruct-v0.2.Q3_K_S.gguf")
        try:
            # Initialize the model
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_batch=512,
                verbose=False
            )
            logger.info(f"Initialized local Mistral model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            self.model = None

    def _get_llm_config(self) -> Dict[str, Any]:
        """Get the LLM configuration for generation."""
        config = {
            "temperature": self.llm_config.get("temperature", 0.7),
            "max_tokens": self.llm_config.get("max_tokens", 2048)
        }
        return config

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response using the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for the LLM

        Returns:
            Dict containing the generated text and other metadata
        """
        if self.llm_provider == "local" and self.model:
            return self._generate_local(prompt, system_prompt)
        else:
            return {"text": "LLM generation not available", "error": "No valid LLM provider configured"}

    def _generate_local(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response using the local LLM."""
        if not self.model:
            return {"text": "", "error": "Local model not initialized"}

        try:
            # Prepare the messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Generate the response
            config = self._get_llm_config()
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )

            # Extract the assistant's message
            generated_text = response["choices"][0]["message"]["content"]
            return {
                "text": generated_text,
                "model": "local-mistral",
                "usage": response.get("usage", {}),
            }
        except Exception as e:
            logger.error(f"Error generating with local model: {e}")
            return {"text": "", "error": str(e)}

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents related to the query.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        if not self.client:
            return [{"error": "R2R client not initialized"}]
        
        try:
            # Use the async search method if available
            return self.client.search(query=query, limit=limit)
        except Exception as e:
            logger.error(f"Error searching with R2R: {e}")
            return [{"error": f"Search failed: {str(e)}"}]

    def generate_with_search(
        self, query: str, system_prompt: Optional[str] = None, limit: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a response with search context.

        Args:
            query: The search query and generation prompt
            system_prompt: Optional system prompt for the LLM
            limit: Maximum number of search results to include

        Returns:
            Dict containing the generated text, search results, and other metadata
        """
        # Search for relevant documents
        search_results = self.search(query, limit)
        
        # Format search results as context
        context = ""
        for idx, result in enumerate(search_results):
            if "error" in result:
                continue
            context += f"Document {idx+1}: {result.get('content', '')}\n\n"
        
        # Combine context with the query
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate response
        response = self.generate(full_prompt, system_prompt)
        
        # Add search results to the response
        response["sources"] = search_results
        
        return response

    def ingest_document(self, document: Union[str, Dict[str, Any]], 
                        collection_key: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ingest a document into a specified collection.
        
        Args:
            document: The document content (text or structured data)
            collection_key: Key of the collection to ingest into
            metadata: Optional metadata for the document
            
        Returns:
            str: Document ID
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection key '{collection_key}' not found")
            
        collection_name = self.collections[collection_key]
        
        # Prepare document content
        if isinstance(document, dict):
            document_content = json.dumps(document)
        else:
            document_content = document
            
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        # Add timestamp to metadata
        metadata["ingestion_time"] = datetime.now().isoformat()
        
        try:
            # Create Document object
            doc = Document(
                text=document_content,
                metadata=metadata
            )
            
            # Add document to collection
            response = self.client.add_documents(
                collection_name=collection_name,
                documents=[doc],
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            
            # Extract document ID from response
            doc_id = response.document_ids[0] if response.document_ids else str(uuid.uuid4())
            
            logger.info(f"Document ingested to '{collection_name}' with ID {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to ingest document: {str(e)}")
            raise
            
    def delete_document(self, document_id: str, collection_key: str) -> bool:
        """
        Delete a document from a collection.
        
        Args:
            document_id: ID of the document to delete
            collection_key: Key of the collection
            
        Returns:
            bool: True if successful, False otherwise
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection key '{collection_key}' not found")
            
        collection_name = self.collections[collection_key]
        
        try:
            # Delete document
            self.client.delete_documents(
                collection_name=collection_name,
                document_ids=[document_id]
            )
            
            logger.info(f"Document {document_id} deleted from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
            
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections.
        
        Returns:
            List of collection information dictionaries
        """
        if not self.client:
            logger.error("Failed to list collections: R2R client not initialized")
            return [{"error": "R2R client not initialized"}]
            
        try:
            collections = []
            # Use self.collections dictionary to return collection info
            for key, name in self.collections.items():
                collections.append({
                    "key": key,
                    "name": name,
                    "exists": True  # We assume these collections exist in this implementation
                })
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return [{"error": f"Failed to list collections: {str(e)}"}]
            
    def get_collection_info(self, collection_key: str) -> Dict[str, Any]:
        """
        Get information about a specific collection.
        
        Args:
            collection_key: The key of the collection to get info for
            
        Returns:
            Dictionary with collection information
            
        Raises:
            ValueError: If the collection key is not found
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection key '{collection_key}' not found")
            
        return {
            "key": collection_key,
            "name": self.collections[collection_key],
            "exists": True,
            "document_count": 0  # This would need to be implemented with a real count
        } 