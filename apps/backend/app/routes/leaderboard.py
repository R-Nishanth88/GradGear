from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Progress, User

router = APIRouter()


@router.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db), limit: int = 10):
    progs = db.query(Progress).all()
    scores = []
    for p in progs:
        d = p.data or {}
        total = (d.get('streak', 0) * 5) + (d.get('quizScore', 0) * 0.5) + (d.get('codingScore', 0) * 0.5)
        scores.append({
            'userId': p.user_id,
            'name': p.user.name,
            'streak': d.get('streak', 0),
            'quizScore': d.get('quizScore', 0),
            'codingScore': d.get('codingScore', 0),
            'total': int(total),
        })
    scores.sort(key=lambda x: x['total'], reverse=True)
    return {"items": scores[:limit]}

