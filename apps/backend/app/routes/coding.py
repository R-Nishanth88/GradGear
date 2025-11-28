from fastapi import APIRouter, Depends
from app.routes.auth_ext import get_current_user
from app.models import User

router = APIRouter()

_CHALLENGES = [
    {
        "id": 1,
        "title": "Two Sum",
        "prompt": "Given an array and target, return indices of two numbers adding to target.",
        "signature": "def two_sum(nums, target):",
        "tests": [
            {"in": {"nums": [2,7,11,15], "target": 9}, "out": [0,1]},
        ],
    }
]


@router.get("/codingtest")
def get_challenge(user: User = Depends(get_current_user)):
    return {"items": _CHALLENGES}


@router.post("/codingtest/submit")
def submit_solution(user: User = Depends(get_current_user)):
    # For safety we do not execute user code here. In production, evaluate in a sandbox.
    return {"result": "received", "score": 0.8, "level": "Intermediate", "next_steps": ["Study algorithms: hash maps", "Practice 2-sum variants"]}


