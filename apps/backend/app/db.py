<<<<<<< HEAD
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
=======
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from motor.motor_asyncio import AsyncIOMotorClient

>>>>>>> 1c14d9e200a05891a5ee3c222d804cb3085955f3
from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


<<<<<<< HEAD
=======
@lru_cache
def get_mongo_client() -> AsyncIOMotorClient:
    if not settings.MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not configured. Set the environment variable to connect to MongoDB Atlas.")
    return AsyncIOMotorClient(settings.MONGODB_URI)


def get_mongo_db():
    client = get_mongo_client()
    db_name = settings.MONGODB_DB_NAME or "gradgear"
    return client[db_name]

>>>>>>> 1c14d9e200a05891a5ee3c222d804cb3085955f3
