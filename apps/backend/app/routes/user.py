from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import User
from app.routes.auth_ext import get_current_user
from pydantic import BaseModel

# Simple in-memory store for demo; replace with DB persistence
_USER_PREFS: Dict[str, List[str]] = {}
_DEMO_USER = "demo@user"  # replace with auth subject


class DomainRequest(BaseModel):
    domains: List[str]


class DomainResponse(BaseModel):
    domains: List[str]


router = APIRouter()


@router.get("/domain", response_model=DomainResponse)
def get_domain(user: User = Depends(get_current_user)) -> DomainResponse:
    domains = _USER_PREFS.get(str(user.id), [])
    return DomainResponse(domains=domains)


@router.post("/domain", response_model=DomainResponse)
def set_domain(req: DomainRequest, user: User = Depends(get_current_user)) -> DomainResponse:
    if not isinstance(req.domains, list):
        raise HTTPException(status_code=400, detail="domains must be a list")
    _USER_PREFS[str(user.id)] = req.domains
    return DomainResponse(domains=req.domains)


