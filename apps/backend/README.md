GradGear Backend (FastAPI)

Quickstart
1) Create virtualenv and install requirements: pip install -r requirements.txt
2) Run dev: uvicorn app.main:app --reload --port 8000
3) Health: GET http://localhost:8000/api/health
4) Auth (stub): POST http://localhost:8000/api/auth/login

Environment
- JWT_SECRET, DATABASE_URL (Postgres), MONGODB_URI, REDIS_URL

Structure
- app/main.py — application factory and router mounting
- app/routes — health, auth, and feature routers
- future: app/core (config), app/models, app/db, app/services


