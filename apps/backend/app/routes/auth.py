from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

JWT_SECRET = "CHANGE_ME"  # replace via env
JWT_ALG = "HS256"
ACCESS_TTL_MIN = 30


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


router = APIRouter()


def _create_access_token(subject: str, ttl_minutes: int = ACCESS_TTL_MIN) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    # NOTE: stub only; replace with DB user lookup + password verify
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = _create_access_token(subject=req.email)
    return TokenResponse(access_token=token)


