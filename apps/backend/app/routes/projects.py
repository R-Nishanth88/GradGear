from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.project_catalog import (
    ProjectDefinition,
    ProjectStep,
    ProjectResource,
    list_projects,
    get_project,
)
from app.core.ai_service import AIService
from app.db import get_db, Base, engine
from app.models import ProjectProgress, Progress, User
from app.schemas import (
    ProjectSummary,
    ProjectDetail,
    ProjectProgressEntry,
    ProjectProgressResponse,
    ProjectStartRequest,
    ProjectStepUpdateRequest,
    ProjectSubmitRequest,
)
from app.routes.auth_ext import get_current_user


router = APIRouter(prefix="/projects", tags=["projects"])

ai_service = AIService()

PROJECT_START_XP = 40
PROJECT_STEP_XP = 20
PROJECT_COMPLETE_XP = 120
PROJECT_STEP_BADGE = "Project Sprint"
PROJECT_COMPLETE_BADGE = "Portfolio Finisher"


def _ensure_tables() -> None:
    try:
        ProjectProgress.__table__.create(bind=engine, checkfirst=True)
    except OperationalError:
        Base.metadata.create_all(bind=engine, tables=[ProjectProgress.__table__])


def _project_to_summary(project: ProjectDefinition, progress: Optional[ProjectProgress]) -> ProjectSummary:
    completed_steps = len(progress.completed_steps) if progress else 0
    total_steps = progress.total_steps if progress else len(project.steps)
    status = progress.status if progress else "not-started"
    return ProjectSummary(
        id=project.id,
        domain=project.domain,
        title=project.title,
        difficulty=project.difficulty,
        time_commitment=project.time_commitment,
        estimated_time=project.estimated_time,
        tech_stack=project.tech_stack,
        tags=project.tags,
        summary=project.summary,
        market_alignment=project.market_alignment,
        progress_status=status,
        completed_steps=completed_steps,
        total_steps=total_steps,
    )


def _resource_schema(resource: ProjectResource) -> dict:
    return {
        "title": resource.title,
        "url": resource.url,
        "kind": resource.kind,
        "provider": resource.provider,
    }


def _step_schema(step: ProjectStep) -> dict:
    return {
        "id": step.id,
        "title": step.title,
        "description": step.description,
        "xp_reward": step.xp_reward,
        "resources": [_resource_schema(r) for r in step.resources],
    }


def _project_to_detail(project: ProjectDefinition, progress: Optional[ProjectProgress]) -> ProjectDetail:
    summary = _project_to_summary(project, progress)
    return ProjectDetail(
        **summary.model_dump(),
        why_it_matters=project.why_it_matters,
        description=project.description,
        dataset=project.dataset,
        expected_output=project.expected_output,
        github_template=project.github_template,
        resources=[_resource_schema(r) for r in project.resources],
        steps=[_step_schema(s) for s in project.steps],
    )


def _progress_to_schema(progress: ProjectProgress) -> ProjectProgressEntry:
    return ProjectProgressEntry.from_orm(progress)


def _get_or_create_progress(db: Session, user_id: int, project: ProjectDefinition) -> ProjectProgress:
    instance = (
        db.query(ProjectProgress)
        .filter(ProjectProgress.user_id == user_id, ProjectProgress.project_id == project.id)
        .first()
    )
    if not instance:
        instance = ProjectProgress(
            user_id=user_id,
            project_id=project.id,
            status="not-started",
            completed_steps=[],
            total_steps=len(project.steps),
            time_commitment=project.time_commitment,
            difficulty=project.difficulty,
        )
        db.add(instance)
        db.flush()
    return instance


def _award_xp(db: Session, user: User, amount: int, *, badge: Optional[str] = None, note: Optional[str] = None) -> None:
    progress = db.query(Progress).filter(Progress.user_id == user.id).first()
    if not progress:
        progress = Progress(user_id=user.id, data={})
        db.add(progress)
        db.flush()

    data = progress.data or {}
    data.setdefault("xp", 0)
    data["xp"] += max(0, amount)

    if badge:
        badges = set(data.get("badges", []))
        badges.add(badge)
        data["badges"] = list(badges)

    history = data.setdefault("project_history", [])
    history.append(
        {
            "note": note or "Project update",
            "xp": amount,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    data["project_history"] = history[-30:]
    progress.data = data
    db.flush()


def _store_resume_project(db: Session, user: User, project: ProjectDefinition, bullet: str, repo_url: str) -> None:
    progress = db.query(Progress).filter(Progress.user_id == user.id).first()
    if not progress:
        progress = Progress(user_id=user.id, data={})
        db.add(progress)
        db.flush()

    data = progress.data or {}
    portfolio = data.setdefault("resume_projects", [])
    portfolio.append(
        {
            "project_id": project.id,
            "title": project.title,
            "bullet": bullet,
            "repo_url": repo_url,
            "generated_at": datetime.utcnow().isoformat(),
        }
    )
    # keep only latest 15
    data["resume_projects"] = portfolio[-15:]
    progress.data = data
    db.flush()


@router.get("", response_model=List[ProjectSummary])
def list_project_summaries(
    domain: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    time_commitment: Optional[str] = Query(None, alias="time"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_tables()

    active_domain = domain or user.domain
    if not active_domain:
        raise HTTPException(status_code=400, detail="Domain not provided and not set on user profile.")

    catalog = list_projects(active_domain, difficulty=difficulty, time_commitment=time_commitment)
    project_ids = [p.id for p in catalog]

    progress_rows = (
        db.query(ProjectProgress)
        .filter(ProjectProgress.user_id == user.id, ProjectProgress.project_id.in_(project_ids))
        .all()
    )
    progress_map = {row.project_id: row for row in progress_rows}

    return [_project_to_summary(project, progress_map.get(project.id)) for project in catalog]


@router.get("/{project_id}", response_model=ProjectProgressResponse)
def get_project_detail(
    project_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_tables()
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    progress = (
        db.query(ProjectProgress)
        .filter(ProjectProgress.user_id == user.id, ProjectProgress.project_id == project_id)
        .first()
    )
    return ProjectProgressResponse(
        project=_project_to_detail(project, progress),
        progress=_progress_to_schema(progress) if progress else None,
    )


@router.post("/{project_id}/start", response_model=ProjectProgressResponse)
def start_project(
    project_id: str,
    payload: ProjectStartRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_tables()
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    progress = _get_or_create_progress(db, user.id, project)
    if progress.status not in ("not-started", "completed") and not payload.overwrite:
        raise HTTPException(status_code=409, detail="Project already in progress.")

    progress.status = "in-progress"
    progress.started_at = datetime.utcnow()
    progress.completed_at = None
    progress.completed_steps = []
    progress.total_steps = len(project.steps)
    progress.github_link_submitted = None
    progress.updated_at = datetime.utcnow()
    db.commit()

    _award_xp(db, user, PROJECT_START_XP, note=f"Started project {project.title}")
    db.commit()

    refreshed = _get_or_create_progress(db, user.id, project)
    return ProjectProgressResponse(
        project=_project_to_detail(project, refreshed),
        progress=_progress_to_schema(refreshed),
    )


@router.post("/{project_id}/steps/complete", response_model=ProjectProgressResponse)
def complete_project_step(
    project_id: str,
    payload: ProjectStepUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_tables()
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    progress = _get_or_create_progress(db, user.id, project)
    if progress.status == "not-started":
        progress.status = "in-progress"
        progress.started_at = progress.started_at or datetime.utcnow()

    valid_steps = {step.id for step in project.steps}
    if payload.step_id not in valid_steps:
        raise HTTPException(status_code=400, detail="Unknown project step.")

    if payload.step_id not in progress.completed_steps:
        progress.completed_steps.append(payload.step_id)
        progress.updated_at = datetime.utcnow()
        db.commit()
        _award_xp(db, user, PROJECT_STEP_XP, note=f"Completed step {payload.step_id} in {project.title}")
        db.commit()

    refreshed = _get_or_create_progress(db, user.id, project)
    return ProjectProgressResponse(
        project=_project_to_detail(project, refreshed),
        progress=_progress_to_schema(refreshed),
    )


@router.post("/{project_id}/submit", response_model=ProjectProgressResponse)
async def submit_project(
    project_id: str,
    payload: ProjectSubmitRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_tables()
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    progress = _get_or_create_progress(db, user.id, project)
    if len(progress.completed_steps) < progress.total_steps:
        raise HTTPException(status_code=400, detail="Complete all project steps before submission.")

    progress.status = "completed"
    progress.github_link_submitted = payload.github_url
    progress.completed_at = datetime.utcnow()
    progress.updated_at = datetime.utcnow()
    db.commit()

    # Generate resume bullet with AI fallback
    bullet = await ai_service.generate_project_bullet(
        title=project.title,
        summary=project.summary,
        tech_stack=project.tech_stack,
        impact_summary=project.why_it_matters,
    )

    _store_resume_project(db, user, project, bullet, payload.github_url)
    _award_xp(
        db,
        user,
        PROJECT_COMPLETE_XP,
        badge=PROJECT_COMPLETE_BADGE,
        note=f"Completed project {project.title}",
    )
    db.commit()

    refreshed = _get_or_create_progress(db, user.id, project)
    return ProjectProgressResponse(
        project=_project_to_detail(project, refreshed),
        progress=_progress_to_schema(refreshed),
    )

