from pathlib import Path

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import (
    health,
    auth,
    user,
    auth_ext,
    resume,
    recommendations,
    progress,
    quiz,
    coding_experience,
    intelligence,
    leaderboard,
    learning,
    projects,
)
from app.db import get_mongo_db


def create_app() -> FastAPI:
    app = FastAPI(title="GradGear API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://0.0.0.0:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(user.router, prefix="/api/user", tags=["user"])
    app.include_router(auth_ext.router, prefix="/api", tags=["auth"])
    app.include_router(resume.router, prefix="/api/resume", tags=["resume"])
    app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])
    app.include_router(progress.router, prefix="/api/user", tags=["progress"])
    app.include_router(quiz.router, prefix="/api", tags=["quiz"])
    app.include_router(coding_experience.router, prefix="/api", tags=["coding"])
    app.include_router(intelligence.router, prefix="/api/intelligence", tags=["intelligence"])
    app.include_router(leaderboard.router, prefix="/api", tags=["leaderboard"])
    app.include_router(learning.router, prefix="/api", tags=["learning"])
    app.include_router(projects.router, prefix="/api", tags=["projects"])

    static_root = Path(__file__).resolve().parent.parent / "static"
    resumes_dir = static_root / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    return app


app = create_app()


@app.on_event("startup")
async def verify_mongo_connection() -> None:
    logger = logging.getLogger("gradgear.mongo")
    try:
        db = get_mongo_db()
        await db.command("ping")
        logger.info("Connected to MongoDB Atlas successfully.")
    except Exception as exc:  # pragma: no cover - startup guard
        logger.error("MongoDB connection failed: %s", exc)


