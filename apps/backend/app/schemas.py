from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    college: Optional[str] = None
    year: Optional[str] = None
    domain: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserProfile(BaseModel):
    id: int
    name: str
    email: EmailStr
    college: Optional[str]
    year: Optional[str]
    domain: Optional[str]

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ResumeAnalyzeRequest(BaseModel):
    resume_id: Optional[int] = None


class ResumeGenerateRequest(BaseModel):
    name: str
    domain: str
    skills: List[str]
    achievements: List[str]
    # Extended fields
    personal: Optional[Dict[str, str]] = None
    education: Optional[List[Dict[str, str]]] = None
    experience: Optional[List[Dict[str, Any]]] = None
    projects: Optional[List[Dict[str, str]]] = None


class ResumeGenerateResponse(BaseModel):
    content: str
    ats_score: Optional[int] = None
    plagiarism_safe: Optional[bool] = None


class ResumeAnalysis(BaseModel):
    ats_score: int
    weak_points: List[str]
    skill_gaps: List[str]
    suggestions: List[str]


class ATSScoreResponse(BaseModel):
    ats_score: int
    faults: List[Dict[str, Any]]
    components: Dict[str, int]  # scores by component


class PlagiarismResponse(BaseModel):
    plagiarism_score: float
    plagiarism_safe: bool
    flagged_passages: List[Dict[str, Any]]


class CourseRecommendation(BaseModel):
    title: str
    source: str  # 'YouTube', 'Coursera', 'Udemy'
    url: str
    estimated_time: str
    level: str
    reason: Optional[str] = None


class SkillRecommendation(BaseModel):
    skill: str
    courses: List[CourseRecommendation]


class RecommendationItem(BaseModel):
    title: str
    url: str
    source: str


class ProgressPayload(BaseModel):
    data: Dict[str, Any]


class AnalyzeResponse(BaseModel):
    ats_score: int
    faults: List[Dict[str, Any]]
    recommendations: List[SkillRecommendation]
    categorized_skills: Dict[str, List[str]]
    plagiarism: PlagiarismResponse
