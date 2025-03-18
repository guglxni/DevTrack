"""
Longitudinal API Routes

This module provides API routes for longitudinal data tracking.
"""

from fastapi import APIRouter

# Create router
router = APIRouter(
    prefix="/longitudinal",
    tags=["longitudinal"],
    responses={404: {"description": "Not found"}}
)

@router.get("/health")
async def health_check():
    """Health check endpoint for the longitudinal API."""
    return {"status": "healthy", "message": "Longitudinal API is working"}