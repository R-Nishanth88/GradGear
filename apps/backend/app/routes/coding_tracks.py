from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db, get_mongo_db
from app.routes.auth_ext import get_current_user
from app.schemas import CodingTrack, CodingTrackListResponse
from app.models import Progress, User
from app.core.domain_utils import normalize_domain

router = APIRouter()


def _serialize_track(
    doc: Dict[str, Any],
    *,
    completed_tasks: int = 0,
    total_tasks: int = 0,
) -> CodingTrack:
    payload = {
        "track_id": doc.get("track_id"),
        "name": doc.get("name"),
        "description": doc.get("description"),
        "order": doc.get("order", 0),
        "difficulty": doc.get("difficulty", "Beginner"),
        "xp": doc.get("xp", 0),
        "total_tasks": total_tasks if total_tasks else doc.get("total_tasks", 0),
        "completed_tasks": completed_tasks,
        "badge": doc.get("badge"),
        "cover_image": doc.get("cover_image"),
    }

    missing_keys = [key for key, value in payload.items() if value is None and key in {"track_id", "name"}]
    if missing_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Track document is missing required fields: {', '.join(missing_keys)}",
        )
    return CodingTrack(**payload)


@router.get("/coding/tracks", response_model=CodingTrackListResponse)
async def list_coding_tracks(
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    sql_db: Session = Depends(get_db),
):
    if not user or not user.domain:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User domain is required to list tracks.")

    domain_key = normalize_domain(user.domain)

    progress_record = sql_db.query(Progress).filter(Progress.user_id == user.id).first()
    coding_state: Dict[str, Any] = {}
    completed_task_ids: List[str] = []
    if progress_record and isinstance(progress_record.data, dict):
        coding_state = progress_record.data.get("coding", {}) or {}
        if isinstance(coding_state, dict):
            completed = coding_state.get("completed_tasks", [])
            if isinstance(completed, list):
                completed_task_ids = completed

    cursor = mongo_db.coding_tracks.find({"domain": domain_key}).sort("order", 1)
    tracks: List[CodingTrack] = []

    async for document in cursor:
        track_id = document.get("track_id")
        if not track_id:
            continue

        total_tasks = document.get("total_tasks", 0)
        if not total_tasks:
            total_tasks = await mongo_db.coding_tasks.count_documents(
                {"domain": domain_key, "track_id": track_id}
            )

        completed_count = 0
        if completed_task_ids:
            completed_count = await mongo_db.coding_tasks.count_documents(
                {
                    "domain": domain_key,
                    "track_id": track_id,
                    "task_id": {"$in": list(completed_task_ids)},
                }
            )

        try:
            track = _serialize_track(
                document,
                completed_tasks=completed_count,
                total_tasks=total_tasks,
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unable to serialize coding track: {exc}",
            )
        tracks.append(track)

    return CodingTrackListResponse(tracks=tracks)

