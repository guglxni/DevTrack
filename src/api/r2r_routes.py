"""
R2R API Routes

This module provides API routes for the R2R (Reason to Retrieve) integration,
allowing access to retrieval and knowledge management capabilities.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, Field

import logging
from fastapi.responses import HTMLResponse, JSONResponse
import os

from src.core.retrieval.r2r_client import R2RClient

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/r2r",
    tags=["r2r"],
    responses={404: {"description": "Not found"}}
)

# Health check endpoint
@router.get("/health")
async def r2r_health():
    """Health check endpoint for the R2R API."""
    return {"status": "healthy", "message": "R2R API is operational"}

# ----- Models -----

class Document(BaseModel):
    """Document to be stored in R2R"""
    content: str = Field(..., description="The document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    collection_key: str = Field(..., description="Collection key to store the document in")

class SearchQuery(BaseModel):
    """Search query for R2R"""
    query: str = Field(..., description="Search query")
    collection_key: str = Field(..., description="Collection to search in")
    limit: int = Field(5, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")

class GenerateQuery(BaseModel):
    """Query for retrieval and generation"""
    query: str = Field(..., description="User query")
    collection_key: str = Field(..., description="Collection to search in")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    prompt_template: Optional[str] = Field(None, description="Optional prompt template")

class DeleteRequest(BaseModel):
    """Request to delete a document"""
    document_id: str = Field(..., description="ID of the document to delete")
    collection_key: str = Field(..., description="Collection key")

class CollectionInfo(BaseModel):
    """Information about a collection"""
    name: str = Field(..., description="Collection name")
    key: str = Field(..., description="Collection key")
    exists: bool = Field(..., description="Whether the collection exists")

# ----- Dependencies -----

def get_r2r_client() -> R2RClient:
    """Get or create the R2R client"""
    try:
        client = R2RClient()
        return client
    except Exception as e:
        logger.error(f"Error initializing R2RClient: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize R2R client: {str(e)}"
        )

# ----- Routes -----

@router.post("/ingest", status_code=201)
async def ingest_document(
    document: Document,
    client: R2RClient = Depends(get_r2r_client)
):
    """
    Ingest a document into a collection
    """
    try:
        doc_id = client.ingest_document(
            document=document.content,
            collection_key=document.collection_key,
            metadata=document.metadata
        )
        
        return {"id": doc_id, "status": "success", "collection": document.collection_key}
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )

@router.post("/search")
async def search_documents(
    search: SearchQuery,
    client: R2RClient = Depends(get_r2r_client)
):
    """
    Search for documents in a collection
    """
    try:
        results = client.search(
            query=search.query,
            collection_key=search.collection_key,
            limit=search.limit,
            filter_criteria=search.filters
        )
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search documents: {str(e)}"
        )

@router.post("/generate")
async def retrieve_and_generate(
    query: GenerateQuery,
    client: R2RClient = Depends(get_r2r_client)
):
    """
    Retrieve information and generate a response
    """
    try:
        result = client.retrieve_and_generate(
            query=query.query,
            collection_key=query.collection_key,
            system_prompt=query.system_prompt,
            prompt_template=query.prompt_template
        )
        
        return result
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )

@router.delete("/document")
async def delete_document(
    request: DeleteRequest,
    client: R2RClient = Depends(get_r2r_client)
):
    """
    Delete a document from a collection
    """
    try:
        success = client.delete_document(
            document_id=request.document_id,
            collection_key=request.collection_key
        )
        
        if success:
            return {"status": "success", "id": request.document_id}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to delete document with ID {request.document_id}"
            )
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/collections")
async def list_collections(
    client: R2RClient = Depends(get_r2r_client)
):
    """
    List all collections
    """
    try:
        collections = client.list_collections()
        return {"collections": collections, "count": len(collections)}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}"
        )

@router.get("/collection/{collection_key}")
async def get_collection_info(
    collection_key: str,
    client: R2RClient = Depends(get_r2r_client)
):
    """
    Get information about a collection
    """
    try:
        info = client.get_collection_info(collection_key)
        return info
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_key}' not found"
        )
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}"
        )

@router.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Get the R2R Dashboard HTML page
    """
    dashboard_path = os.path.join("src", "web", "static", "r2r", "index.html")
    with open(dashboard_path, "r") as f:
        return f.read()

@router.get("/search.html", response_class=HTMLResponse)
async def get_search_page():
    """
    Get the Search HTML page
    """
    page_path = os.path.join("src", "web", "static", "r2r", "search.html")
    with open(page_path, "r") as f:
        return f.read()

@router.get("/collections.html", response_class=HTMLResponse)
async def get_collections_page():
    """
    Get the Collections HTML page
    """
    page_path = os.path.join("src", "web", "static", "r2r", "collections.html")
    with open(page_path, "r") as f:
        return f.read()

@router.get("/generate.html", response_class=HTMLResponse)
async def get_generate_page():
    """
    Get the Generate HTML page
    """
    page_path = os.path.join("src", "web", "static", "r2r", "generate.html")
    with open(page_path, "r") as f:
        return f.read()

@router.get("/ingest.html", response_class=HTMLResponse)
async def get_ingest_page():
    """
    Get the Ingest HTML page
    """
    page_path = os.path.join("src", "web", "static", "r2r", "ingest.html")
    with open(page_path, "r") as f:
        return f.read()

# Function to register routes with the main app
def add_routes_to_app(app):
    """Add R2R routes to the FastAPI app"""
    app.include_router(router) 