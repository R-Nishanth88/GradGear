GradGear Architecture

High-Level
GradGear uses a hybrid architecture: a modern React (Vite) frontend, a FastAPI backend, and an Apache Spark analytics layer. PostgreSQL stores relational data (users, courses, goals, skills, recommendations), MongoDB stores unstructured data (chat history, resume JSON, notes). Spark processes bulk data in S3/HDFS for market trends, clustering, and skill-gap detection. LLMs (OpenAI/Gemini/local) power natural language features.

Modules
- Academic Dashboard: aggregates schedules, grades, tasks, predictions
- AI Study Assistant: chat, document QA, summarization, flashcards
- Project & Research Manager: literature review, methodology, templates, GitHub sync
- Notes Organizer: OCR, tagging, semantic search, export
- Grade & Performance: GPA/CGPA, visualizations, predictive estimator
- Career & Placement Hub: resume builder, job alerts, interview prep
- Task & Goal Planner: Kanban, Pomodoro, productivity analytics
- Collaboration: shared workspaces, real-time editing (stubs), group calendar
- Security & Auth: JWT, RBAC (Student/Mentor/Admin), encrypted storage
- Integrations: Google Drive, GitHub, Calendar, Notion, Email

Backend Services
- API Gateway (FastAPI): REST + WebSocket endpoints
- Worker (Celery): async tasks (OCR, PDF parsing, embeddings, LLM calls)
- Vector Store: optional pgvector or external (e.g., Qdrant) for embeddings
- Model Adapters: OpenAI/Gemini/local LLM abstraction

Data Stores
- PostgreSQL: accounts, enrollments, subjects, tasks, events, skills, recs
- MongoDB: chats, notes JSON, resume JSON, extracted entities
- Object Storage (S3/GCS): uploaded docs, processed datasets
- Kafka (optional): event stream for analytics

Spark Layer
- Batch jobs: job market trends, clustering, skill-gap scoring, GPA prediction
- ML pipelines: feature engineering, model training, evaluation
- Outputs: recommendation tables written back to PostgreSQL

Security
- JWT with refresh tokens, RBAC, audit logs
- Secrets via environment variables and secret manager (K8s)
- File encryption at rest for sensitive uploads

Deployment
- Frontend: Vercel
- Backend: Render/Heroku (or container on ECS/K8s)
- Spark: EMR/Databricks or K8s cluster
- DB: Managed Postgres/Mongo services

Observability
- Logging: structured JSON logs
- Metrics: Prometheus/Grafana (optional)
- Tracing: OpenTelemetry (optional)


