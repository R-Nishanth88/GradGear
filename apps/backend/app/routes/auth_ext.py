from datetime import datetime, timedelta, timezone
import jwt
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import hash_password, verify_password
from app.db import get_db, Base, engine
from app.models import User
from app.schemas import RegisterRequest, LoginRequest, TokenResponse, UserProfile


router = APIRouter()


def create_access_token(sub: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.ACCESS_TTL_MIN)).timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    password_hash = hash_password(req.password)
    
    user = User(
        name=req.name,
        email=req.email,
        password_hash=password_hash,
        college=req.college,
        year=req.year,
        domain=req.domain,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Registration successful. You can now log in."}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Verify password
    password_valid = verify_password(req.password, user.password_hash)
    
    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token(sub=str(user.id))
    return TokenResponse(access_token=token)


def get_current_user(db: Session = Depends(get_db), authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        uid = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).get(uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/user/profile", response_model=UserProfile)
def profile(user: User = Depends(get_current_user)):
    return user
