from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from app.core.domain_utils import normalize_domain
from app.db import get_db, get_mongo_db
from app.models import Progress, User
from app.routes.auth_ext import get_current_user
from app.schemas import (
    CodingHintRequest,
    CodingHintResponse,
    CodingLessonResponse,
    CodingNextSkill,
    CodingOverviewProgress,
    CodingOverviewResponse,
    CodingOverviewWeeklyChallenge,
    CodingQuestionPayload,
    CodingQuestionResponse,
    CodingQuestionSubmission,
    CodingSkillProgress,
    CodingTrackProgressRequest,
    CodingTrackProgressResponse,
    CodingRunResponse,
    CodingRunTestResult,
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _sanitize_document(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_document(v) for k, v in value.items() if k != "_id"}
    if isinstance(value, list):
        return [_sanitize_document(item) for item in value]
    return value


async def _latest_document(
    collection, *, query: Dict[str, Any], sort_key: str = "generated_at"
) -> Optional[Dict[str, Any]]:
    cursor = collection.find(query).sort(sort_key, -1).limit(1)
    docs = await cursor.to_list(length=1)
    return docs[0] if docs else None


async def _load_track_for_domain(
    mongo_db: AsyncIOMotorDatabase,
    *,
    domain_key: str,
    level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    query: Dict[str, Any] = {"domain": domain_key}
    if level:
        query["level"] = level
    cursor = mongo_db.coding_tracks.find(query).sort("order", 1)
    tracks = await cursor.to_list(length=1)
    return tracks[0] if tracks else None


def _extract_progress(progress_model: Optional[Progress]) -> CodingOverviewProgress:
    data = progress_model.data if progress_model and isinstance(progress_model.data, dict) else {}
    coding_state = data.get("coding", {}) if isinstance(data, dict) else {}

    streak = int(coding_state.get("streak", 0)) if isinstance(coding_state, dict) else 0
    xp = int(coding_state.get("xp", data.get("xp", 0))) if isinstance(coding_state, dict) else int(data.get("xp", 0))
    stats = coding_state.get("stats", {}) if isinstance(coding_state, dict) else {}

    return CodingOverviewProgress(
        xp=xp,
        streak_days=streak,
        pass_rate=float(stats.get("pass_rate", 0.0)) if isinstance(stats, dict) else 0.0,
        mastery_rate=float(stats.get("mastery_rate", 0.0)) if isinstance(stats, dict) else 0.0,
        ladder_level=coding_state.get("ladder_level", "Beginner") if isinstance(coding_state, dict) else "Beginner",
        interview_ready_score=stats.get("interview_ready_score") if isinstance(stats, dict) else None,
        time_on_task_minutes=int(stats.get("time_on_task_minutes", 0)) if isinstance(stats, dict) else None,
    )


def _extract_skill_progress(coding_state: Dict[str, Any]) -> List[CodingSkillProgress]:
    if not isinstance(coding_state, dict):
        return []
    skills_meta = coding_state.get("skills", [])
    skill_progress: List[CodingSkillProgress] = []
    if isinstance(skills_meta, list):
        for entry in skills_meta:
            if not isinstance(entry, dict):
                continue
            entry = _sanitize_document(entry)
            skill_progress.append(
                CodingSkillProgress(
                    name=entry.get("name", ""),
                    mastery=float(entry.get("mastery", 0.0)),
                    attempts=int(entry.get("attempts", 0)),
                    level=entry.get("level", "Beginner"),
                    last_practiced_at=entry.get("lastPracticeAt") or entry.get("last_practiced_at"),
                )
            )
    return skill_progress


async def _fallback_next_skills(
    mongo_db: AsyncIOMotorDatabase,
    *,
    domain_key: str,
) -> List[CodingNextSkill]:
    # Use seeded tasks as fallback suggestions
    cursor = mongo_db.coding_tasks.find({"domain": domain_key}).sort("order", 1)
    docs = await cursor.to_list(length=5)
    suggestions: List[CodingNextSkill] = []
    for doc in docs:
        suggestions.append(
            CodingNextSkill(
                skill=doc.get("skill_tag") or doc.get("title", ""),
                difficulty=doc.get("difficulty", "Beginner"),
                reason="Seeded from curated track",
                sources=["curated_track"],
                recommended_resources=[
                    {"type": "task", "task_id": doc.get("task_id"), "title": doc.get("title")}
                ],
            )
        )
    return suggestions


async def _fetch_next_skills(
    mongo_db: AsyncIOMotorDatabase,
    *,
    user_id: int,
    domain_key: str,
) -> List[CodingNextSkill]:
    rec_doc = await _latest_document(
        mongo_db.coding_next_skills,
        query={"user_id": user_id, "domain": domain_key},
        sort_key="generated_at",
    )
    if not rec_doc:
        return await _fallback_next_skills(mongo_db, domain_key=domain_key)

    items = rec_doc.get("skills", [])
    results: List[CodingNextSkill] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item = _sanitize_document(item)
            results.append(
                CodingNextSkill(
                    skill=item.get("skill"),
                    difficulty=item.get("difficulty", "Beginner"),
                    reason=item.get("reason"),
                    sources=item.get("sources", []),
                    recommended_resources=item.get("recommended_resources", []),
                )
            )
    if not results:
        return await _fallback_next_skills(mongo_db, domain_key=domain_key)
    return results


async def _build_weekly_challenge(
    mongo_db: AsyncIOMotorDatabase,
    *,
    domain_key: str,
) -> Optional[CodingOverviewWeeklyChallenge]:
    challenge = await _latest_document(
        mongo_db.coding_weekly_challenges,
        query={"domain": domain_key},
        sort_key="published_at",
    )
    if not challenge:
        return None
    links = challenge.get("reference_links", [])
    if isinstance(links, list):
        links = [str(link) if not isinstance(link, dict) else jsonable_encoder(link).get("url") for link in links]
        links = [link for link in links if link]
    return CodingOverviewWeeklyChallenge(
        title=challenge.get("title", "Weekly Challenge"),
        description=challenge.get("description", ""),
        difficulty=challenge.get("difficulty", "Intermediate"),
        reward_xp=int(challenge.get("reward_xp", 150)),
        trend_reason=challenge.get("trend_reason"),
        reference_links=links,
    )


async def _fetch_lesson(
    mongo_db: AsyncIOMotorDatabase,
    *,
    domain_key: str,
    skill: str,
) -> Optional[Dict[str, Any]]:
    return await mongo_db.coding_lessons.find_one({"domain": domain_key, "skill": skill})


async def _fetch_question_from_playlist(
    mongo_db: AsyncIOMotorDatabase,
    *,
    domain_key: str,
    level: Optional[str],
) -> Optional[CodingQuestionPayload]:
    playlist = await _latest_document(
        mongo_db.coding_playlists_daily,
        query={"domain": domain_key, "level": level} if level else {"domain": domain_key},
        sort_key="date",
    )
    question_ids = playlist.get("question_ids") if playlist else []
    if not question_ids:
        return None

    for question_id in question_ids:
        question = await mongo_db.coding_questions.find_one({"question_id": question_id})
        if not question:
            continue
        return _serialize_question_doc(question)
    return None


def _serialize_question_doc(doc: Dict[str, Any]) -> CodingQuestionPayload:
    walkthrough = doc.get("walkthrough") or _build_walkthrough(doc)
    practice = doc.get("practice_question") or _build_practice_question(doc)
    return CodingQuestionPayload(
        question_id=doc.get("question_id") or doc.get("task_id"),
        skill=doc.get("skill") or doc.get("skill_tag", ""),
        title=doc.get("title", "Coding Exercise"),
        prompt=doc.get("prompt") or doc.get("description", ""),
        difficulty=doc.get("difficulty", "Beginner"),
        language=doc.get("language", "python"),
        starter_code=doc.get("starter_code", ""),
        tests=doc.get("tests", []),
        hints=doc.get("hints", []) or doc.get("public_hints", []),
        tags=doc.get("skill_tags", []),
        estimated_minutes=doc.get("estimate_minutes"),
        entry_function=doc.get("entry_function") or doc.get("entry_point") or doc.get("entrypoint"),
        guided_steps=doc.get("guided_steps", []),
        resources=doc.get("resources", []),
        walkthrough=walkthrough,
        practice_question=practice,
    )


async def _get_question_payload(
    mongo_db: AsyncIOMotorDatabase,
    *,
    question_id: str,
    domain_key: str,
) -> Optional[CodingQuestionPayload]:
    question = await mongo_db.coding_questions.find_one({"question_id": question_id, "domain": domain_key})
    if question:
        return _serialize_question_doc(question)

    task = await mongo_db.coding_tasks.find_one({"task_id": question_id, "domain": domain_key})
    if task:
        return _serialize_question_doc(task)
    return None


async def _pick_next_question(
    mongo_db: AsyncIOMotorDatabase,
    *,
    user: User,
    domain_key: str,
    level: Optional[str],
) -> CodingQuestionResponse:
    # Prefer outstanding assigned tasks
    assignment = await mongo_db.coding_tasks_assigned.find_one(
        {
            "user_id": user.id,
            "domain": domain_key,
            "status": {"$in": ["assigned", "in_progress"]},
        },
        sort=[("assigned_at", -1)],
    )
    if assignment:
        payload = await _get_question_payload(
            mongo_db,
            question_id=assignment.get("question_id"),
            domain_key=domain_key,
        )
        if payload:
            return CodingQuestionResponse(
                question=payload,
                playlist_source=assignment.get("source"),
                suggested_followups=assignment.get("suggested_followups", []),
            )

    playlist_question = await _fetch_question_from_playlist(
        mongo_db, domain_key=domain_key, level=level
    )
    if playlist_question:
        return CodingQuestionResponse(
            question=playlist_question,
            playlist_source="playlist",
            suggested_followups=[],
        )

    fallback_cursor = mongo_db.coding_tasks.find({"domain": domain_key}).sort("order", 1)
    fallback_task = await fallback_cursor.to_list(length=1)
    if not fallback_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No coding tasks available yet.")

    payload = _serialize_question_doc(fallback_task[0])
    return CodingQuestionResponse(question=payload, playlist_source="fallback", suggested_followups=[])


def _evaluate_python(code: str, payload: CodingQuestionPayload) -> List[Dict[str, Any]]:
    from textwrap import dedent

    compiled_namespace: Dict[str, Any] = {}
    execution_trace: List[Dict[str, Any]] = []

    try:
        exec(dedent(code), {"__builtins__": __builtins__}, compiled_namespace)
    except Exception as exc:  # pragma: no cover - defensive
        return [
            {
                "name": "compile",
                "passed": False,
                "message": "Code failed to execute.",
                "error": repr(exc),
            }
        ]

    entry_fn = None
    if payload.entry_function:
        entry_fn = compiled_namespace.get(payload.entry_function)
    if not entry_fn and payload.guided_steps:
        first_step = payload.guided_steps[0]
        if isinstance(first_step, dict):
            entry_fn = compiled_namespace.get(first_step.get("entry_function"))
    if not entry_fn:
        entry_fn = compiled_namespace.get("solve") or compiled_namespace.get("main")
    if not entry_fn and payload.tests:
        first_test = payload.tests[0]
        if isinstance(first_test, dict):
            entry_fn = compiled_namespace.get(first_test.get("entry_function", ""))
    if not callable(entry_fn):
        return [
            {
                "name": "function-definition",
                "passed": False,
                "message": "Expected solution function was not defined.",
                "hint": "Keep the provided function signature intact.",
            }
        ]

    for idx, test in enumerate(payload.tests or [], start=1):
        name = test.get("name", f"Test {idx}")
        args = test.get("input")
        kwargs = test.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        if isinstance(args, dict) and "args" in args:
            kwargs = args.get("kwargs", kwargs)
            args = args.get("args", [])
        if not isinstance(args, (list, tuple)):
            args = [args] if args is not None else []
        expected = test.get("expected")
        try:
            result = entry_fn(*args, **kwargs)
            passed = result == expected
            execution_trace.append(
                {
                    "name": name,
                    "passed": passed,
                    "expected": expected,
                    "received": result,
                    "hint": None if passed else test.get("hint"),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            execution_trace.append(
                {
                    "name": name,
                    "passed": False,
                    "expected": expected,
                    "received": None,
                    "hint": test.get("hint"),
                    "error": repr(exc),
                }
            )
    return execution_trace


def _build_run_results(outcomes: List[Dict[str, Any]]) -> CodingRunResponse:
    results: List[CodingRunTestResult] = []
    total = len(outcomes)
    passed_count = 0
    for outcome in outcomes:
        passed = bool(outcome.get("passed"))
        if passed:
            passed_count += 1
        results.append(
            CodingRunTestResult(
                name=outcome.get("name", "Test"),
                passed=passed,
                expected=outcome.get("expected"),
                received=outcome.get("received"),
                message=outcome.get("message"),
                hint=outcome.get("hint"),
                error=outcome.get("error"),
            )
        )
    score = (passed_count / total * 100) if total else 0.0
    return CodingRunResponse(
        passed=passed_count == total and total > 0,
        score=round(score, 2),
        total_tests=total,
        passed_tests=passed_count,
        results=results,
        execution_time_ms=None,
        error=None if passed_count == total else "Some tests failed.",
    )


def _build_walkthrough(doc: Dict[str, Any]) -> List[str]:
    prompt = doc.get("prompt") or doc.get("description")
    steps: List[str] = []
    if prompt:
        steps.append(f"Understand the goal: {prompt}")
    starter = doc.get("starter_code")
    if starter:
        if "TODO" in starter:
            steps.append("Identify the TODO section in the starter code and fill in the missing logic.")
        steps.append("Walk through each line of the starter code and predict what it should do before writing anything.")
    tests = doc.get("tests", [])
    if tests:
        sample_names = ", ".join(test.get("name", f"Test {idx+1}") for idx, test in enumerate(tests[:3]))
        steps.append(f"Think about how to satisfy the sample tests ({sample_names}).")
    if doc.get("solution_code"):
        lines = [line.strip() for line in doc["solution_code"].splitlines() if line.strip() and not line.strip().startswith("#")]
        for line in lines[:4]:
            explanation = _describe_line(line)
            if explanation:
                steps.append(explanation)
    steps.append("Run the tests locally, inspect failures, and iterate until all hidden cases pass.")
    return steps


def _build_practice_question(doc: Dict[str, Any]) -> Dict[str, Any]:
    title = doc.get("title", "Challenge")
    skill = doc.get("skill") or doc.get("skill_tag", "")
    prompt = doc.get("prompt") or doc.get("description", "")
    variation_prompt = "Extend your solution to handle an additional edge case: allow the input to include nested collections and flatten them before processing."
    if "sql" in (doc.get("language") or "").lower():
        variation_prompt = "Write a SQL query that produces the same metrics but filters for the top three categories per month."
    return {
        "title": f"{title} — stretch goal",
        "skill": skill,
        "prompt": variation_prompt,
        "hint": "Reuse the core approach from the main task, but adjust the preprocessing to cover the new requirement.",
    }


def _describe_line(line: str) -> Optional[str]:
    if line.startswith("if not"):
        return "Check for empty input early to avoid errors."
    if "round(" in line:
        return "Round the computed result to the required precision."
    if "sum(" in line and "/" in line:
        return "Compute the aggregate by dividing the sum by the count of items."
    if "append" in line:
        return "Collect results into a list to return from the function."
    if line.startswith("return"):
        return "Return the final value once all edge cases are handled."
    return None


async def _update_progress_after_submission(
    db: Session,
    mongo_db: AsyncIOMotorDatabase,
    *,
    user: User,
    domain_key: str,
    payload: CodingQuestionPayload,
    run_response: CodingRunResponse,
    xp_awarded: int,
) -> Dict[str, Any]:
    progress = db.query(Progress).filter(Progress.user_id == user.id).with_for_update().first()
    if not progress:
        progress = Progress(user_id=user.id, data={})
        db.add(progress)
        db.flush()

    data = progress.data if isinstance(progress.data, dict) else {}
    coding_state = data.setdefault("coding", {})
    stats = coding_state.setdefault("stats", {})

    now = datetime.now(timezone.utc)
    stats["last_submission_at"] = now.isoformat()
    stats["last_skill"] = payload.skill

    attempts = stats.get("total_attempts", 0) + 1
    stats["total_attempts"] = attempts
    if run_response.passed:
        stats["total_passed"] = stats.get("total_passed", 0) + 1

    total_passed = stats.get("total_passed", 0)
    stats["pass_rate"] = round(total_passed / attempts, 3) if attempts else 0.0
    stats["mastery_rate"] = stats.get("mastery_rate", 0.0)

    skills = coding_state.setdefault("skills", [])
    updated = False
    for entry in skills:
        if isinstance(entry, dict) and entry.get("name") == payload.skill:
            entry["lastPracticeAt"] = now.isoformat()
            entry["attempts"] = entry.get("attempts", 0) + 1
            if run_response.passed:
                entry["mastery"] = min(1.0, entry.get("mastery", 0.0) + 0.1)
            updated = True
            break
    if not updated:
        skills.append(
            {
                "name": payload.skill,
                "lastPracticeAt": now.isoformat(),
                "attempts": 1,
                "mastery": 0.1 if run_response.passed else 0.0,
                "level": payload.difficulty,
            }
        )

    coding_state.setdefault("xp", 0)
    if run_response.passed and xp_awarded:
        coding_state["xp"] += xp_awarded
        data["xp"] = data.get("xp", 0) + xp_awarded

    progress.data = data
    db.commit()

    await mongo_db.coding_submissions.insert_one(
        {
            "user_id": user.id,
            "domain": domain_key,
            "question_id": payload.question_id,
            "passed": run_response.passed,
            "score": run_response.score,
            "results": [result.model_dump() for result in run_response.results],
            "created_at": now,
        }
    )

    return {
        "xp_awarded": xp_awarded if run_response.passed else 0,
        "streak": coding_state.get("streak"),
        "total_xp": data.get("xp", 0),
        "coding_xp": coding_state.get("xp", 0),
    }


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


@router.get("/coding/overview", response_model=CodingOverviewResponse)
async def get_coding_overview(
    domain: Optional[str] = Query(None, description="Domain override"),
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
    db: Session = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_label = domain or user.domain or "AI/ML"
    domain_key = normalize_domain(domain_label)

    progress_model = db.query(Progress).filter(Progress.user_id == user.id).first()
    progress = _extract_progress(progress_model)
    coding_state = (progress_model.data or {}).get("coding", {}) if progress_model and isinstance(progress_model.data, dict) else {}

    next_skills = await _fetch_next_skills(mongo_db, user_id=user.id, domain_key=domain_key)
    skill_progress = _extract_skill_progress(coding_state)
    current_track = await _load_track_for_domain(
        mongo_db,
        domain_key=domain_key,
        level=coding_state.get("ladder_level") if isinstance(coding_state, dict) else None,
    )
    weekly_challenge = await _build_weekly_challenge(mongo_db, domain_key=domain_key)

    return CodingOverviewResponse(
        domain=domain_label,
        focus_skill=next_skills[0].skill if next_skills else None,
        progress=progress,
        next_skills=next_skills,
        skill_progress=skill_progress,
        current_track=_sanitize_document(current_track) if current_track else None,
        weekly_challenge=weekly_challenge,
    )


@router.get("/coding/lesson", response_model=CodingLessonResponse)
async def get_coding_lesson(
    skill: str = Query(..., description="Skill identifier"),
    domain: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_label = domain or user.domain or "AI/ML"
    domain_key = normalize_domain(domain_label)

    lesson = await _fetch_lesson(mongo_db, domain_key=domain_key, skill=skill)
    if not lesson:
        fallback_task = await mongo_db.coding_tasks.find_one({"domain": domain_key, "skill_tag": skill})
        if not fallback_task:
            fallback_task = await mongo_db.coding_tasks.find_one({"domain": domain_key}) or {}
        return CodingLessonResponse(
            skill=skill,
            title=f"{skill} essentials",
            markdown="<p>Start by reviewing the starter code and constraints from today’s adaptive challenge. Focus on understanding the inputs, outputs, and edge cases.</p>",
            demo_code=fallback_task.get("starter_code", ""),
            estimated_minutes=int(fallback_task.get("estimate_minutes", 15)) if fallback_task else 15,
            steps=[
                "Review the starter code provided in the coding challenge.",
                "Reproduce the solution without looking at hints.",
                "Write at least one additional test case covering an edge scenario.",
            ],
            references=[
                {
                    "label": "StackOverflow discussion",
                    "url": "https://stackoverflow.com/questions/tagged/python",
                }
            ],
        )

    return CodingLessonResponse(
        skill=lesson.get("skill", skill),
        title=lesson.get("title", f"{skill} Lesson"),
        markdown=lesson.get("markdown", ""),
        demo_code=lesson.get("demo_code"),
        estimated_minutes=int(lesson.get("estimated_minutes", 15)),
        steps=lesson.get("steps", []),
        references=lesson.get("references", []),
    )


@router.get("/coding/question/next", response_model=CodingQuestionResponse)
async def get_next_question(
    domain: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_label = domain or user.domain or "AI/ML"
    domain_key = normalize_domain(domain_label)

    question_response = await _pick_next_question(
        mongo_db,
        user=user,
        domain_key=domain_key,
        level=level or None,
    )
    return question_response


@router.post("/coding/submit", response_model=CodingRunResponse)
async def submit_coding_question(
    payload: CodingQuestionSubmission,
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
    db: Session = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_key = normalize_domain(user.domain or "AI/ML")
    question_payload = await _get_question_payload(
        mongo_db, question_id=payload.question_id, domain_key=domain_key
    )
    if not question_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")

    if (payload.language or question_payload.language).lower() != "python":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only Python questions are supported right now.")

    outcomes = _evaluate_python(payload.code, question_payload)
    run_response = _build_run_results(outcomes)

    xp_award = 50 if run_response.passed else 0
    summary = await _update_progress_after_submission(
        db,
        mongo_db,
        user=user,
        domain_key=domain_key,
        payload=question_payload,
        run_response=run_response,
        xp_awarded=xp_award,
    )

    run_response.xp_awarded = summary.get("xp_awarded", 0)
    run_response.streak = summary.get("streak")
    run_response.total_xp = summary.get("total_xp")
    run_response.coding_xp = summary.get("coding_xp")

    return run_response


@router.post("/coding/hint", response_model=CodingHintResponse)
async def get_contextual_hint(
    request: CodingHintRequest,
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_key = normalize_domain(user.domain or "AI/ML")
    question = await _get_question_payload(
        mongo_db, question_id=request.question_id, domain_key=domain_key
    )
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found.")

    hint_index = min(max((request.attempt or 1) - 1, 0), len(question.hints) - 1)
    hint_text = question.hints[hint_index] if question.hints else "Focus on decomposing the problem into smaller steps."

    resources = [ref.get("url") for ref in question.resources if isinstance(ref, dict) and ref.get("url")]
    return CodingHintResponse(hint=hint_text, additional_resources=resources)


@router.post("/coding/track", response_model=CodingTrackProgressResponse)
async def update_track_progress(
    request: CodingTrackProgressRequest,
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
    db: Session = Depends(get_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    track = await mongo_db.coding_tracks.find_one({"track_id": request.track_id})
    if not track or track.get("domain") != normalize_domain(user.domain or ""):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track not found for user domain.")

    progress = db.query(Progress).filter(Progress.user_id == user.id).with_for_update().first()
    if not progress:
        progress = Progress(user_id=user.id, data={})
        db.add(progress)
        db.flush()

    data = progress.data if isinstance(progress.data, dict) else {}
    coding_state = data.setdefault("coding", {})
    tracks_state = coding_state.setdefault("tracks", {})
    track_state = tracks_state.setdefault(request.track_id, {"completed_task_ids": [], "status": "not-started"})

    completed_items = set(track_state.get("completed_task_ids", []))
    if request.status == "completed":
        completed_items.add(request.item_id)
    elif request.item_id in completed_items and request.status == "not-started":
        completed_items.remove(request.item_id)

    track_state["completed_task_ids"] = list(completed_items)
    track_state["status"] = request.status
    track_state["updated_at"] = datetime.now(timezone.utc).isoformat()

    progress.data = data
    db.commit()

    return CodingTrackProgressResponse(
        track_id=request.track_id,
        item_id=request.item_id,
        status=request.status,
        completed_items=track_state["completed_task_ids"],
        updated_at=datetime.now(timezone.utc),
    )


@router.get("/coding/track", response_model=Dict[str, Any])
async def get_track_content(
    domain: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    mongo_db: AsyncIOMotorDatabase = Depends(get_mongo_db),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    domain_label = domain or user.domain or "AI/ML"
    domain_key = normalize_domain(domain_label)

    track = await _load_track_for_domain(mongo_db, domain_key=domain_key, level=level)
    if not track:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track not found yet for this domain.")

    # Merge curated items with trending playlist suggestions if available
    playlist = await _latest_document(
        mongo_db.coding_playlists_daily,
        query={"domain": domain_key, "level": track.get("level")},
        sort_key="date",
    )
    playlist_questions = playlist.get("question_ids", []) if playlist else []

    return {
        "track": _sanitize_document(track),
        "playlist": playlist_questions,
    }
