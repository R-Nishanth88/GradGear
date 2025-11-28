<<<<<<< HEAD
from fastapi import APIRouter, Depends
from app.routes.auth_ext import get_current_user
from app.models import User

router = APIRouter()

_CHALLENGES = [
    {
        "id": 1,
        "title": "Two Sum",
        "prompt": "Given an array and target, return indices of two numbers adding to target.",
        "signature": "def two_sum(nums, target):",
        "tests": [
            {"in": {"nums": [2,7,11,15], "target": 9}, "out": [0,1]},
        ],
    }
]


@router.get("/codingtest")
def get_challenge(user: User = Depends(get_current_user)):
    return {"items": _CHALLENGES}


@router.post("/codingtest/submit")
def submit_solution(user: User = Depends(get_current_user)):
    # For safety we do not execute user code here. In production, evaluate in a sandbox.
    return {"result": "received", "score": 0.8, "level": "Intermediate", "next_steps": ["Study algorithms: hash maps", "Practice 2-sum variants"]}


=======
from __future__ import annotations
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from app.db import get_db, get_mongo_db
from app.models import Progress, User
from app.routes.auth_ext import get_current_user
from app.schemas import (
    CodingRunRequest,
    CodingRunResponse,
    CodingSubmitResponse,
    CodingTaskDetail,
    CodingTaskListResponse,
    CodingTaskSummary,
    TestResult,
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _normalize_domain(value: str) -> str:
    return (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .lower()
    )


async def _fetch_task(
    mongo_db: AsyncIOMotorDatabase,
    task_id: str,
) -> Dict[str, Any]:
    task = await mongo_db.coding_tasks.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    return task


def _task_to_summary(
    doc: Dict[str, Any],
    *,
    status: str = "not-started",
) -> CodingTaskSummary:
    return CodingTaskSummary(
        task_id=doc["task_id"],
        track_id=doc["track_id"],
        title=doc.get("title", "Untitled task"),
        difficulty=doc.get("difficulty", "Beginner"),
        estimate_minutes=doc.get("estimate_minutes", 20),
        skill_tag=doc.get("skill_tag"),
        category=doc.get("category", "Beginner"),
        order=doc.get("order", 0),
        xp=doc.get("xp", 50),
        status=status,
    )


def _task_to_detail(
    doc: Dict[str, Any],
    *,
    status: str = "not-started",
) -> CodingTaskDetail:
    description = doc.get("description")
    if not description:
        description = doc.get("prompt", "")

    return CodingTaskDetail(
        task_id=doc["task_id"],
        track_id=doc["track_id"],
        title=doc.get("title", "Untitled task"),
        difficulty=doc.get("difficulty", "Beginner"),
        estimate_minutes=doc.get("estimate_minutes", 20),
        skill_tag=doc.get("skill_tag"),
        category=doc.get("category", "Beginner"),
        order=doc.get("order", 0),
        xp=doc.get("xp", 50),
        status=status,
        description=description,
        language=doc.get("language", "python"),
        starter_code=doc.get("starter_code", ""),
        entry_function=doc.get("entrypoint") or doc.get("entry_function"),
        hints=doc.get("public_hints") or doc.get("hints", []),
        resources=doc.get("resources", []),
        tests=doc.get("tests", []),
        is_project=bool(doc.get("is_project", False)),
    )


def _build_test_results(
    *,
    task: Dict[str, Any],
    outcomes: List[Dict[str, Any]],
) -> List[TestResult]:
    results: List[TestResult] = []
    for idx, outcome in enumerate(outcomes, start=1):
        expected = outcome.get("expected")
        received = outcome.get("received")
        results.append(
            TestResult(
                name=outcome.get("name") or f"Test {idx}",
                passed=outcome.get("passed", False),
                expected=repr(expected),
                received=repr(received),
                message=outcome.get("message"),
                hint=outcome.get("hint"),
                error=outcome.get("error"),
            )
        )
    return results


async def _run_python(
    code: str,
    *,
    entrypoint: str,
    tests: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Execute Python user code against predetermined tests."""

    def _executor() -> List[Dict[str, Any]]:
        namespace: Dict[str, Any] = {}
        outcomes: List[Dict[str, Any]] = []
        try:
            exec(code, {"__builtins__": __builtins__}, namespace)
        except Exception as exc:  # pragma: no cover - defensive
            return [
                {
                    "name": "compile",
                    "passed": False,
                    "message": "Code failed to execute.",
                    "error": repr(exc),
                    "expected": None,
                    "received": None,
                }
            ]

        func = namespace.get(entrypoint)
        if not callable(func):
            return [
                {
                    "name": "function-definition",
                    "passed": False,
                    "message": f"Function `{entrypoint}` was not defined.",
                    "hint": "Ensure the starter function name is unchanged.",
                    "expected": "Callable function",
                    "received": str(type(func)),
                }
            ]

        for test in tests:
            input_payload = test.get("input", {})
            if isinstance(input_payload, dict):
                args = input_payload.get("args", [])
                kwargs = input_payload.get("kwargs", {})
            else:
                args = test.get("args", [])
                kwargs = test.get("kwargs", {})
            if not isinstance(args, (list, tuple)):
                args = [args] if args is not None else []
            if not isinstance(kwargs, dict):
                kwargs = {}
            expected = test.get("expected")
            expected_exception = test.get("expected_exception")
            if isinstance(expected, str) and expected_exception is None and expected.endswith("Error"):
                expected_exception = expected
                expected = None
            name = test.get("name")
            hint = test.get("hint")
            try:
                received = func(*args, **kwargs)
                if expected_exception:
                    outcomes.append(
                        {
                            "name": name,
                            "passed": False,
                            "expected": expected_exception,
                            "received": received,
                            "message": f"Expected exception {expected_exception}, but function returned a value.",
                            "hint": hint,
                        }
                    )
                    continue
                passed = received == expected
                outcomes.append(
                    {
                        "name": name,
                        "passed": passed,
                        "expected": expected,
                        "received": received,
                        "message": "Great job!" if passed else test.get("message"),
                        "hint": None if passed else hint,
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                exc_name = type(exc).__name__
                if expected_exception and expected_exception in exc_name:
                    outcomes.append(
                        {
                            "name": name,
                            "passed": True,
                            "expected": expected_exception,
                            "received": exc_name,
                            "message": "Exception raised as expected.",
                            "hint": None,
                        }
                    )
                    continue
                outcomes.append(
                    {
                        "name": name,
                        "passed": False,
                        "expected": expected,
                        "received": None,
                        "message": "Code raised an exception.",
                        "error": repr(exc),
                        "hint": hint,
                    }
                )
        return outcomes

    return await run_in_threadpool(_executor)


async def _run_tests(
    *,
    task: Dict[str, Any],
    code: str,
    language: Optional[str],
) -> List[Dict[str, Any]]:
    tests = task.get("tests", [])
    entrypoint = task.get("entrypoint")
    if not tests or not entrypoint:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task is missing execution metadata.",
        )

    language = (language or task.get("language", "python") or "python").lower()
    if language != "python":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only Python-based tasks are supported currently.")

    return await _run_python(code, entrypoint=entrypoint, tests=tests)


async def _record_submission(
    mongo_db: AsyncIOMotorDatabase,
    *,
    user: User,
    task: Dict[str, Any],
    results: List[TestResult],
    score: float,
    passed: bool,
    code: str,
) -> None:
    await mongo_db.coding_submissions.insert_one(
        {
            "user_id": user.id,
            "task_id": task["task_id"],
            "track_id": task.get("track_id"),
            "domain": task.get("domain"),
            "score": score,
            "passed": passed,
            "results": jsonable_encoder(results),
            "code": code,
            "created_at": datetime.utcnow(),
        }
    )


def _apply_xp_and_streak(
    db: Session,
    user: User,
    *,
    xp_awarded: int,
    task_id: str,
    track: Dict[str, Any],
    task_completed: bool,
) -> Dict[str, Any]:
    progress = db.query(Progress).filter(Progress.user_id == user.id).first()
    if not progress:
        progress = Progress(user_id=user.id, data={})
        db.add(progress)
        db.flush()

    data = progress.data or {}
    data.setdefault("xp", 0)
    data["xp"] += max(0, xp_awarded)

    coding_data = data.setdefault("coding", {})
    coding_data["xp"] = int(coding_data.get("xp", 0)) + max(0, xp_awarded)

    tracks_state = coding_data.setdefault("tracks", {})
    track_state = tracks_state.setdefault(
        track["track_id"],
        {
            "completed_task_ids": [],
            "status": "not-started",
            "completed_at": None,
        },
    )

    if task_completed and task_id not in track_state["completed_task_ids"]:
        track_state["completed_task_ids"].append(task_id)

    now = datetime.utcnow()
    today = now.date()
    last_date_str = coding_data.get("last_submission_date")
    if last_date_str:
        last_date = date.fromisoformat(last_date_str)
        delta = (today - last_date).days
        if delta == 0:
            streak = coding_data.get("streak", 1)
        elif delta == 1:
            streak = coding_data.get("streak", 0) + 1
        else:
            streak = 1
    else:
        streak = 1

    coding_data["streak"] = streak
    coding_data["last_submission_date"] = today.isoformat()
    coding_data["last_submission_at"] = now.isoformat()

    history = coding_data.setdefault("history", [])
    history.append(
        {
            "task_id": task_id,
            "track_id": track["track_id"],
            "xp": xp_awarded,
            "timestamp": now.isoformat(),
        }
    )
    coding_data["history"] = history[-50:]

    badge = None
    track_completed = False
    awarded_badge: Optional[str] = None
    total_tasks = track.get("total_tasks", 0)
    completed_count = len(set(track_state["completed_task_ids"]))
    if total_tasks and completed_count >= total_tasks:
        track_state["status"] = "completed"
        if not track_state.get("completed_at"):
            track_state["completed_at"] = now.isoformat()
        track_completed = True
        badge = track.get("badge")
    else:
        track_state["status"] = "in-progress" if track_state["completed_task_ids"] else "not-started"

    if badge:
        badges = set(data.get("badges", []))
        badges.add(badge)
        data["badges"] = list(badges)
        coding_badges = set(coding_data.get("badges", []))
        coding_badges.add(badge)
        coding_data["badges"] = list(coding_badges)
        awarded_badge = badge

    progress.data = data
    db.flush()

    return {
        "streak": streak,
        "total_xp": data.get("xp", 0),
        "coding_xp": coding_data.get("xp", 0),
        "badges": data.get("badges", []),
        "track_completed": track_completed,
        "awarded_badge": awarded_badge,
        "track_state": track_state,
    }


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


@router.get("/coding/tracks/{track_id}/tasks", response_model=CodingTaskListResponse)
async def list_tasks_for_track(
    track_id: str,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    db: Session = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_key = _normalize_domain(user.domain or "")
    track = await mongo_db.coding_tracks.find_one({"track_id": track_id})
    if not track or track.get("domain") != domain_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track not found for your domain.")

    progress = db.query(Progress).filter(Progress.user_id == user.id).first()
    coding_state = (progress.data or {}).get("coding", {}) if progress and isinstance(progress.data, dict) else {}
    track_states = coding_state.get("tracks", {}) if isinstance(coding_state, dict) else {}
    track_entry = track_states.get(track_id, {}) if isinstance(track_states, dict) else {}
    completed_ids = set(track_entry.get("completed_task_ids", [])) if isinstance(track_entry, dict) else set()

    cursor = mongo_db.coding_tasks.find({"track_id": track_id}).sort("order", 1)
    tasks: List[CodingTaskSummary] = []
    async for task_doc in cursor:
        status = "completed" if task_doc["task_id"] in completed_ids else "not-started"
        tasks.append(_task_to_summary(task_doc, status=status))

    return CodingTaskListResponse(tasks=tasks)


@router.get("/coding/tasks/{task_id}", response_model=CodingTaskDetail)
async def get_task_detail(
    task_id: str,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    db: Session = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    task = await _fetch_task(mongo_db, task_id)
    track = await mongo_db.coding_tracks.find_one({"track_id": task["track_id"]})
    if not track or track.get("domain") != _normalize_domain(user.domain or ""):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not part of your domain.")

    progress = db.query(Progress).filter(Progress.user_id == user.id).first()
    coding_state = (progress.data or {}).get("coding", {}) if progress and isinstance(progress.data, dict) else {}
    track_states = coding_state.get("tracks", {}) if isinstance(coding_state, dict) else {}
    track_entry = track_states.get(task["track_id"], {}) if isinstance(track_states, dict) else {}
    completed_ids = set(track_entry.get("completed_task_ids", [])) if isinstance(track_entry, dict) else set()
    status = "completed" if task_id in completed_ids else "not-started"

    return _task_to_detail(task, status=status)


@router.post("/coding/tasks/{task_id}/run", response_model=CodingRunResponse)
async def run_code(
    task_id: str,
    payload: CodingRunRequest,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    task = await _fetch_task(mongo_db, task_id)
    outcomes = await _run_tests(task=task, code=payload.code, language=task.get("language", "python"))
    results = _build_test_results(task=task, outcomes=outcomes)
    total = len(results)
    passed_count = sum(1 for result in results if result.passed)
    passed = passed_count == total
    score = (passed_count / total * 100) if total else 0.0
    return CodingRunResponse(
        passed=passed,
        score=round(score, 2),
        total_tests=total,
        passed_tests=passed_count,
        results=results,
        execution_time_ms=None,
        error=None if passed else "Some tests failed.",
    )


@router.post("/coding/tasks/{task_id}/submit", response_model=CodingSubmitResponse)
async def submit_code(
    task_id: str,
    payload: CodingRunRequest,
    user: User = Depends(get_current_user),
    mongo_db=Depends(get_mongo_db),
    db: Session = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    task = await _fetch_task(mongo_db, task_id)
    track_id = task["track_id"]
    track = await mongo_db.coding_tracks.find_one({"track_id": track_id})
    if not track or track.get("domain") != _normalize_domain(user.domain or ""):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not part of your domain.")

    outcomes = await _run_tests(task=task, code=payload.code, language=task.get("language", "python"))
    results = _build_test_results(task=task, outcomes=outcomes)
    total = len(results)
    passed_count = sum(1 for result in results if result.passed)
    score = (passed_count / total * 100) if total else 0.0
    passed = passed_count == total

    await _record_submission(
        mongo_db,
        user=user,
        task=task,
        results=results,
        score=score,
        passed=passed,
        code=payload.code,
    )

    xp_reward = task.get("xp", 50) if passed else 0
    rewards = _apply_xp_and_streak(
        db,
        user,
        xp_awarded=xp_reward if passed else 0,
        task_id=task_id,
        track=track,
        task_completed=passed,
    )
    db.commit()

    return CodingSubmitResponse(
        passed=passed,
        score=round(score, 2),
        total_tests=total,
        passed_tests=passed_count,
        results=results,
        execution_time_ms=None,
        error=None if passed else "Some tests failed.",
        xp_awarded=xp_reward if passed else 0,
        streak=rewards.get("streak"),
        total_xp=rewards.get("total_xp"),
        coding_xp=rewards.get("coding_xp"),
        badges=rewards.get("badges", []),
        track_completed=rewards.get("track_completed", False),
        awarded_badge=rewards.get("awarded_badge"),
    )
>>>>>>> 1c14d9e200a05891a5ee3c222d804cb3085955f3
