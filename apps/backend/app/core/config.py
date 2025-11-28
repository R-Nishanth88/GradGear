import os


class Settings:
    APP_NAME = "GradGear API"
    JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME")
    JWT_ALG = "HS256"
    ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", "60"))

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gradgear.db")
<<<<<<< HEAD
=======
    MONGODB_URI = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "gradgear")
>>>>>>> 1c14d9e200a05891a5ee3c222d804cb3085955f3
    
    # AI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    AI_MODEL = os.getenv("AI_MODEL", "gemini")  # "openai" or "gemini"
    
    # External APIs
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
    COURSERA_API_KEY = os.getenv("COURSERA_API_KEY", "")


settings = Settings()


