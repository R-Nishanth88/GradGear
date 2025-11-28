from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from app.core.domain_utils import normalize_domain
from app.db import get_db, get_mongo_db
from app.models import Progress, User
from app.routes.auth_ext import get_current_user

router = APIRouter()

XP_LEVELS = [
    {"name": "Beginner", "min": 0, "max": 200},
    {"name": "Intermediate", "min": 201, "max": 600},
    {"name": "Advanced", "min": 601, "max": 999999},
]


async def _fetch_trending_skills(
    mongo_db: AsyncIOMotorDatabase,
    *,
    domain_key: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    cursor = (
        mongo_db.trending_skills
        .find({"domain": domain_key})
        .sort("generated_at", -1)
        .limit(limit)
    )
    results: List[Dict[str, Any]] = []
    async for doc in cursor:
        skills = doc.get("skills") or []
        if isinstance(skills, list):
            results = skills[:limit]
            break
    return results


async def _fetch_skill_gap(
    mongo_db: AsyncIOMotorDatabase,
    *,
    user_id: int,
    domain_key: str,
) -> List[str]:
    profile = await mongo_db.skill_gap_profiles.find_one({"user_id": user_id, "domain": domain_key})
    if profile and isinstance(profile.get("missing_skills"), list):
        return profile["missing_skills"]
    return []


def _compute_level(xp: int) -> Dict[str, Any]:
    for level in XP_LEVELS:
        if level["min"] <= xp <= level["max"]:
            span = level["max"] - level["min"]
            progress = 0 if span <= 0 else (xp - level["min"]) / span
            return {"name": level["name"], "progress": round(progress, 2)}
    return {"name": "Beginner", "progress": 0.0}


@router.get("/trending")
async def get_trending_skills(
    domain: str = Query(..., description="Domain label e.g. AI/ML"),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    domain_key = normalize_domain(domain)
    items = await _fetch_trending_skills(mongo_db, domain_key=domain_key)
    if not items:
        return {"domain": domain, "skills": [], "message": "Trending intelligence not populated yet. Trigger the Spark job to update this feed."}
    return {"domain": domain, "skills": items}


@router.get("/skill-gap")
async def get_personal_skill_gap(
    domain: str = Query(..., description="Domain label"),
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    domain_key = normalize_domain(domain)
    missing = await _fetch_skill_gap(mongo_db, user_id=user.id, domain_key=domain_key)
    return {"domain": domain, "missing_skills": missing}


@router.get("/user/insights")
async def get_user_progress_overview(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    progress = db.query(Progress).filter(Progress.user_id == user.id).first()
    data = progress.data if progress and isinstance(progress.data, dict) else {}
    coding_state = data.get("coding", {}) if isinstance(data, dict) else {}

    xp = int(data.get("xp", 0))
    level = _compute_level(xp)
    streak = coding_state.get("streak", 0)
    badges = data.get("badges", []) if isinstance(data, dict) else []

    domain_key = normalize_domain(user.domain or "")
    trending = await _fetch_trending_skills(mongo_db, domain_key=domain_key, limit=5)

    next_skill_doc = await mongo_db.coding_next_skills.find_one({"user_id": user.id, "domain": domain_key})
    next_skill = None
    if next_skill_doc and isinstance(next_skill_doc.get("skills"), list) and next_skill_doc["skills"]:
        next_skill = next_skill_doc["skills"][0]

    return {
        "xp": xp,
        "level": level,
        "streak": streak,
        "badges": badges,
        "trending_skills": trending,
        "next_suggestion": next_skill,
    }
