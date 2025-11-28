from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, auth, user, auth_ext, resume, recommendations, progress, quiz, coding, leaderboard


def create_app() -> FastAPI:
    app = FastAPI(title="GradGear API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        # Explicit origins for dev, plus a regex to catch variations
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://0.0.0.0:5173",
            "http://localhost",
            "http://127.0.0.1",
        ],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],  # includes Authorization
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
    app.include_router(coding.router, prefix="/api", tags=["coding"])
    app.include_router(leaderboard.router, prefix="/api", tags=["leaderboard"])

    return app


app = create_app()


