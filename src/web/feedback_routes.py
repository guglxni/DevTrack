"""
Web routes for the feedback interface.
Provides a web interface for expert feedback collection and model improvement tracking.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

from src.core.scoring.active_learning import ActiveLearningEngine

# Set up templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Create router
router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    responses={404: {"description": "Not found"}},
)

# Dependency to get the active learning engine
def get_active_learning_engine():
    # This could be a singleton or loaded from a factory
    return ActiveLearningEngine()

@router.get("/", response_class=HTMLResponse)
async def get_feedback_interface(request: Request):
    """Renders the feedback interface for expert review."""
    return templates.TemplateResponse(
        "feedback_interface.html", 
        {"request": request}
    )

def add_routes_to_app(app):
    """Register the feedback routes with the main application."""
    app.include_router(router) 