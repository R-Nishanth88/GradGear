GradGear: AI-Driven Career Intelligence and Upskilling Platform

Overview
GradGear is a full-stack platform that blends modern web UX with a big data and AI backend to deliver academic assistance, career intelligence, and personalized upskilling paths. It includes an academic dashboard, AI study assistant, research manager, notes organizer, performance tracker, career hub, planner, collaboration, secure auth, and cloud integrations.

Monorepo Structure
- apps/frontend — React (Vite), TailwindCSS, Framer Motion, Recharts
- apps/backend — FastAPI, PostgreSQL, MongoDB, Celery, Redis
- apps/spark — Apache Spark jobs (PySpark), MLlib, Spark NLP
- packages/shared — Shared types, schemas, and utilities
- infra/docker — Dockerfiles and docker-compose
- infra/k8s — Kubernetes manifests
- docs — Architecture, ADRs, API references
- scripts — Local dev scripts and helpers

Core Technologies
- Frontend: React (Vite), TailwindCSS, Framer Motion, Recharts/Chart.js
- Backend: FastAPI, SQLAlchemy, Pydantic, Celery, Redis, PostgreSQL, MongoDB
- Data/AI: Apache Spark (MLlib, Spark NLP), HDFS/S3, OpenAI/Gemini/local LLM
- Auth: JWT (optionally Firebase Auth)
- Integrations: Google Drive, GitHub, Google Calendar, Notion

Getting Started
1) Prereqs: Python 3.11+, Node 18+, Docker, Java (for Spark)
2) Copy .env.example to .env and fill secrets
3) Backend: `cd apps/backend && pip install -r requirements.txt && uvicorn app.main:app --reload`
4) Frontend: `cd apps/frontend && npm install && npm run dev`
5) Spark: see apps/spark/README.md

Features Implemented ✅
- Secure JWT authentication (register, login, persistent sessions)
- Domain-based personalization (Cybersecurity, AI/ML, Data Science, Web Dev, Cloud, Robotics, IoT)
- Academic Dashboard with skill progress charts and analytics
- AI Resume Builder (upload & analyze, auto-generate with AI)
- Skill Dashboard with progress tracking, badges, and Leaderboard
- Domain-specific recommendations (courses, tutorials, projects)
- Quiz system with domain-based MCQs
- Coding skill assessment with challenges
- Learning Path with personalized resources
- Profile management with domain preferences

Roadmap (Future)
- Real-time chat with AI tutor
- PDF summarization and flashcards
- Calendar sync and task planner
- Plagiarism checker
- Advanced Spark ML pipelines

License
Proprietary — for internal development unless otherwise specified.


