import asyncio
from typing import List, Dict, Any

from app.db import get_mongo_db

TRACKS: List[Dict[str, Any]] = [
    {
        "domain": "ai_ml",
        "track_id": "ai-ml-track-1",
        "name": "Python Refreshers & Data Structures",
        "description": "Brush up on Python constructs you will reuse across ML pipelines.",
        "order": 1,
        "difficulty": "Beginner",
        "xp": 120,
        "badge": "Python Primer",
    },
    {
        "domain": "ai_ml",
        "track_id": "ai-ml-track-2",
        "name": "Data Wrangling with NumPy & Pandas",
        "description": "Practice realistic data cleaning patterns on CSV, JSON, and parquet sources.",
        "order": 2,
        "difficulty": "Beginner",
        "xp": 180,
        "badge": "Data Wrangler",
    },
    {
        "domain": "data_science",
        "track_id": "ds-track-1",
        "name": "SQL Fundamentals for Analytics",
        "description": "Solve SQL queries modelled on product analytics and BI interviews.",
        "order": 1,
        "difficulty": "Beginner",
        "xp": 150,
        "badge": "SQL Sprinter",
    },
    {
        "domain": "web_development",
        "track_id": "web-track-1",
        "name": "JavaScript Foundations & DOM Patterns",
        "description": "Modern JS syntax, DOM APIs, and asynchronous workflows to warm up for frontend challenges.",
        "order": 1,
        "difficulty": "Beginner",
        "xp": 150,
        "badge": "JS Kickoff",
    },
]


async def seed() -> None:
    db = get_mongo_db()

    for track in TRACKS:
        await db.coding_tracks.update_one(
            {"track_id": track["track_id"]},
            {"$set": track},
            upsert=True,
        )

    print(f"Seeded {len(TRACKS)} coding tracks.")


if __name__ == "__main__":
    asyncio.run(seed())

