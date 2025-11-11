from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.learning_data import (
    get_certifications,
    get_learning_path,
    get_resources,
)
from app.core.project_catalog import list_projects
from app.routes.auth_ext import get_current_user
from app.models import User

router = APIRouter()


@router.get("/learning/path")
async def learning_path(
    domain: str = Query(..., description="Domain name e.g. AI/ML, Cybersecurity"),
    user: User = Depends(get_current_user),
):
    """Return curated skill roadmap for the requested domain."""
    path = get_learning_path(domain)
    if not path:
        raise HTTPException(status_code=404, detail="Learning path not available for this domain yet.")
    return {
        "domain": domain,
        "headline": path.get("headline"),
        "steps": path.get("steps", []),
        "projects": path.get("projects", []),
    }


@router.get("/resources")
async def curated_resources(
    domain: str = Query(..., description="Domain name e.g. AI/ML"),
    skill: str | None = Query(None, description="Optional skill tag to filter resources"),
    user: User = Depends(get_current_user),
):
    """Return curated resources for a domain/skill combination."""
    items = get_resources(domain, skill)
    if not items:
        raise HTTPException(status_code=404, detail="Resources not yet curated for this selection.")
    return {"domain": domain, "skill": skill, "resources": items}


@router.get("/certifications")
async def certification_roadmap(
    domain: str = Query(..., description="Domain name e.g. AI/ML"),
    user: User = Depends(get_current_user),
):
    """Return certification roadmap for the selected domain."""
    items = get_certifications(domain)
    if not items:
        raise HTTPException(status_code=404, detail="Certification roadmap not available for this domain yet.")
    return {"domain": domain, "certifications": items}


