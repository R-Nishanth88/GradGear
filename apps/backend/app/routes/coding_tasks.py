from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.domain_utils import normalize_domain
from app.db import get_db, get_mongo_db
from app.models import Progress, User
from app.routes.auth_ext import get_current_user
from app.schemas import (
    CodingRunRequest,
    CodingRunResponse,
    CodingRunTestResult,
    CodingTaskDetail,
    CodingTaskListResponse,
    CodingTaskSummary,
)

router = APIRouter()


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (set, tuple)):
        return list(value)
    return [value]


def _serialize_task_summary(doc: Dict[str, Any], status: str) -> CodingTaskSummary:
    try:
        summary = CodingTaskSummary(
            task_id=doc["task_id"],
            track_id=doc["track_id"],
            title=doc["title"],
            difficulty=doc.get("difficulty", "Beginner"),
            estimate_minutes=int(doc.get("estimate_minutes", 20)),
            skill_tag=doc.get("skill_tag"),
            category=doc.get("category", "Beginner"),
            order=int(doc.get("order", 0)),
            xp=int(doc.get("xp", 50)),
            status=status,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coding task document missing required field: {exc}",
        ) from exc
    return summary


def _serialize_task_detail(doc: Dict[str, Any], status: str) -> CodingTaskDetail:
    description = doc.get("description") or doc.get("prompt") or ""
    entry_function = (
        doc.get("entry_function")
        or doc.get("entrypoint")
        or doc.get("entry_point")
    )

    try:
        detail = CodingTaskDetail(
            task_id=doc["task_id"],
            track_id=doc["track_id"],
            title=doc["title"],
            difficulty=doc.get("difficulty", "Beginner"),
            estimate_minutes=int(doc.get("estimate_minutes", 20)),
            skill_tag=doc.get("skill_tag"),
            category=doc.get("category", "Beginner"),
            order=int(doc.get("order", 0)),
            xp=int(doc.get("xp", 50)),
            status=status,
            description=description,
            language=doc.get("language", "python"),
            starter_code=doc.get("starter_code", ""),
            entry_function=entry_function,
            hints=_coerce_list(doc.get("hints") or doc.get("public_hints")),
            resources=_coerce_list(doc.get("resources")),
            tests=_coerce_list(doc.get("tests")),
            is_project=bool(doc.get("is_project", False)),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coding task document missing required field: {exc}",
        ) from exc
    return detail


async def _fetch_task_document(mongo_db, task_id: str, domain_key: str) -> Dict[str, Any]:
    doc = await mongo_db.coding_tasks.find_one({"task_id": task_id, "domain": domain_key})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Coding task not found.")
    return doc


def _evaluate_python_code(
    source_code: str,
    entry_point: str,
    tests: List[Dict[str, Any]],
) -> Tuple[List[CodingRunTestResult], int, int]:
    namespace: Dict[str, Any] = {}
    results: List[CodingRunTestResult] = []

    try:
        compiled = compile(source_code, "<submission>", "exec")
        exec(compiled, {}, namespace)
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            CodingRunTestResult(
                name="Syntax / Runtime error",
                passed=False,
                error=str(exc),
            )
        )
        return results, 0, len(tests) if tests else 1

    func = namespace.get(entry_point)
    if not callable(func):
        results.append(
            CodingRunTestResult(
                name="Entry point",
                passed=False,
                error=f"Function '{entry_point}' was not defined.",
            )
        )
        return results, 0, len(tests) if tests else 1

    passed_tests = 0
    total_tests = len(tests)
    for idx, test in enumerate(tests, start=1):
        name = test.get("name") or f"Test case {idx}"
        hint = test.get("hint")
        args = test.get("input", [])
        kwargs = test.get("kwargs", {})
        expected = test.get("expected")

        if isinstance(args, dict) and not kwargs:
            # Allow shorthand for keyword-only tests
            kwargs = args
            args = []

        if not isinstance(args, (list, tuple)):
            args = [args]

        try:
            output = func(*args, **kwargs)
            passed = output == expected
            if passed:
                passed_tests += 1
            result = CodingRunTestResult(
                name=name,
                passed=passed,
                expected=expected,
                received=output,
                hint=None if passed else hint,
            )
        except Exception as exc:  # pragma: no cover - defensive
            result = CodingRunTestResult(
                name=name,
                passed=False,
                hint=hint,
                error=str(exc),
            )
        results.append(result)

    return results, passed_tests, total_tests


async def _evaluate_submission(task_doc: Dict[str, Any], payload: CodingRunRequest) -> Tuple[List[CodingRunTestResult], int, int, str]:
    language = task_doc.get("language", "python").lower()
    entry_point = task_doc.get("entry_point")
    tests = task_doc.get("tests", [])

    if language != "python":
        result = CodingRunTestResult(
            name="Language support",
            passed=False,
            message="Only Python tasks are supported in the current sandbox.",
            hint="Switch the task language to Python or choose a Python-based challenge.",
        )
        return [result], 0, len(tests) if tests else 1, language

    if not entry_point:
        result = CodingRunTestResult(
            name="Task configuration",
            passed=False,
            error="Task entry point not configured. Contact support.",
        )
        return [result], 0, len(tests) if tests else 1, language

    results, passed_tests, total_tests = _evaluate_python_code(
        payload.code,
        entry_point=entry_point,
        tests=tests,
    )
    return results, passed_tests, total_tests, language


def _compute_streak(last_submission_iso: str | None) -> Tuple[int, datetime]:
    now = datetime.now(timezone.utc)
    if not last_submission_iso:
        return 1, now

    try:
        previous = datetime.fromisoformat(last_submission_iso)
    except ValueError:  # pragma: no cover - defensive
        return 1, now

    delta_days = (now.date() - previous.date()).days
    if delta_days == 0:
        # Same day, keep existing streak
        return 0, now
    if delta_days == 1:
        return 1, now
    return -1, now  # sentinel to indicate reset


@router.get("/coding/tracks/{track_id}/tasks", response_model=CodingTaskListResponse)
async def list_coding_tasks(
    track_id: str,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    sql_db: Session = Depends(get_db),
):
    if not user or not user.domain:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User domain is required.")

    domain_key = normalize_domain(user.domain)
    cursor = mongo_db.coding_tasks.find({"track_id": track_id, "domain": domain_key}).sort("order", 1)

    progress_record = sql_db.query(Progress).filter(Progress.user_id == user.id).first()
    coding_state = {}
    if progress_record and isinstance(progress_record.data, dict):
        coding_state = progress_record.data.get("coding", {}) or {}

    completed_task_ids = set(_coerce_list(coding_state.get("completed_tasks")))

    tasks: List[CodingTaskSummary] = []
    async for document in cursor:
        status = "completed" if document.get("task_id") in completed_task_ids else "not-started"
        tasks.append(_serialize_task_summary(document, status=status))

    return CodingTaskListResponse(tasks=tasks)


@router.get("/coding/tasks/{task_id}", response_model=CodingTaskDetail)
async def get_coding_task_detail(
    task_id: str,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    sql_db: Session = Depends(get_db),
):
    if not user or not user.domain:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User domain is required.")

    domain_key = normalize_domain(user.domain)
    document = await _fetch_task_document(mongo_db, task_id, domain_key)

    progress_record = sql_db.query(Progress).filter(Progress.user_id == user.id).first()
    coding_state = {}
    if progress_record and isinstance(progress_record.data, dict):
        coding_state = progress_record.data.get("coding", {}) or {}
    completed_task_ids = set(_coerce_list(coding_state.get("completed_tasks")))

    status = "completed" if task_id in completed_task_ids else "not-started"
    return _serialize_task_detail(document, status=status)


@router.post("/coding/tasks/{task_id}/run", response_model=CodingRunResponse)
async def run_coding_task(
    task_id: str,
    payload: CodingRunRequest,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
):
    if not user or not user.domain:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User domain is required.")

    domain_key = normalize_domain(user.domain)
    document = await _fetch_task_document(mongo_db, task_id, domain_key)

    results, passed_tests, total_tests, language = await _evaluate_submission(document, payload)

    total_tests = max(total_tests, 1)
    score = round((passed_tests / total_tests) * 100, 2)

    return CodingRunResponse(
        passed=passed_tests == total_tests,
        score=score,
        total_tests=total_tests,
        passed_tests=passed_tests,
        results=results,
        execution_time_ms=None,
        detail=f"Executed in {language.upper()} environment (simulated)",
    )


@router.post("/coding/tasks/{task_id}/submit", response_model=CodingRunResponse)
async def submit_coding_task(
    task_id: str,
    payload: CodingRunRequest,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    sql_db: Session = Depends(get_db),
):
    if not user or not user.domain:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User domain is required.")

    domain_key = normalize_domain(user.domain)
    document = await _fetch_task_document(mongo_db, task_id, domain_key)
    track_id = document.get("track_id")
    if not track_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task configuration missing track.")

    results, passed_tests, total_tests, language = await _evaluate_submission(document, payload)
    total_tests = max(total_tests, 1)
    score = round((passed_tests / total_tests) * 100, 2)
    passed = passed_tests == total_tests

    xp_awarded = 0
    streak_value = None
    total_xp = None
    coding_xp = None
    badges_awarded: List[str] = []
    track_completed = False
    awarded_badge = None

    if passed:
        xp_awarded = int(document.get("xp", 50))
        now = datetime.now(timezone.utc)

        progress_record = sql_db.query(Progress).filter(Progress.user_id == user.id).first()
        if not progress_record:
            progress_record = Progress(user_id=user.id, data={})
            sql_db.add(progress_record)
            sql_db.flush()

        data = progress_record.data or {}
        coding_state = data.get("coding", {}) or {}

        completed_tasks = set(_coerce_list(coding_state.get("completed_tasks")))
        already_completed = task_id in completed_tasks
        if not already_completed:
            completed_tasks.add(task_id)

        last_submission = coding_state.get("last_submission_date")
        streak_delta, submitted_at = _compute_streak(last_submission)
        if streak_delta == 0:
            streak_value = coding_state.get("streak", 1)
        elif streak_delta < 0:
            streak_value = 1
        else:
            streak_value = coding_state.get("streak", 0) + 1

        total_xp = coding_state.get("xp", 0)
        if not already_completed:
            total_xp += xp_awarded
        else:
            xp_awarded = 0  # prevent double awarding

        tracks_state = coding_state.get("tracks", {}) or {}
        track_state = tracks_state.get(track_id, {}) or {}
        completed_for_track = set(_coerce_list(track_state.get("completed_task_ids")))
        completed_for_track.add(task_id)
        track_state["completed_task_ids"] = list(completed_for_track)
        tracks_state[track_id] = track_state

        total_tasks = await mongo_db.coding_tasks.count_documents(
            {"domain": domain_key, "track_id": track_id}
        )
        if total_tasks and len(completed_for_track) >= total_tasks:
            track_completed = True
            badge = document.get("badge") or track_state.get("badge")
            if not badge:
                track_doc = await mongo_db.coding_tracks.find_one({"track_id": track_id, "domain": domain_key})
                badge = track_doc.get("badge") if track_doc else None
            badges = set(_coerce_list(coding_state.get("badges")))
            if badge:
                if badge not in badges:
                    badges.add(badge)
                    badges_awarded.append(badge)
                    awarded_badge = badge
            coding_state["badges"] = list(badges)

        coding_state["tracks"] = tracks_state
        coding_state["completed_tasks"] = list(completed_tasks)
        coding_state["xp"] = total_xp
        coding_state["last_submission_date"] = submitted_at.isoformat()
        coding_state["streak"] = streak_value
        coding_xp = total_xp

        data["coding"] = coding_state
        progress_record.data = data
        sql_db.commit()

        await mongo_db.coding_submissions.insert_one(
            {
                "user_id": user.id,
                "task_id": task_id,
                "track_id": track_id,
                "domain": domain_key,
                "code": payload.code,
                "passed": True,
                "score": score,
                "submitted_at": now.isoformat(),
                "xp_awarded": xp_awarded,
            }
        )

    else:
        await mongo_db.coding_submissions.insert_one(
            {
                "user_id": user.id,
                "task_id": task_id,
                "track_id": document.get("track_id"),
                "domain": domain_key,
                "code": payload.code,
                "passed": False,
                "score": score,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "xp_awarded": 0,
            }
        )

    return CodingRunResponse(
        passed=passed,
        score=score,
        total_tests=total_tests,
        passed_tests=passed_tests,
        results=results,
        execution_time_ms=None,
        detail=f"Executed in {language.upper()} environment (simulated)",
        xp_awarded=xp_awarded,
        streak=streak_value,
        total_xp=total_xp,
        coding_xp=coding_xp,
        badges=badges_awarded,
        track_completed=track_completed,
        awarded_badge=awarded_badge,
    )


