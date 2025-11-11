from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Progress, User
from app.schemas import ProgressPayload
from app.routes.auth_ext import get_current_user

router = APIRouter()


def _get_user(current: User) -> User:
    return current


@router.get("/progress")
def get_progress(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    prog = db.query(Progress).filter(Progress.user_id == user.id).first()
    return {"data": prog.data if prog else {}}


@router.post("/progress")
def update_progress(payload: ProgressPayload, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    prog = db.query(Progress).filter(Progress.user_id == user.id).first()
    if not prog:
        prog = Progress(user_id=user.id, data=payload.data)
        db.add(prog)
    else:
        prog.data = payload.data
    db.commit()
    db.refresh(prog)
    return {"data": prog.data}


