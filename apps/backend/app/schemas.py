from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, EmailStr


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


class ResumeContact(BaseModel):
    email: EmailStr
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class ResumeEducationEntry(BaseModel):
    institution: str
    degree: str
    year: Optional[str] = None
    gpa: Optional[str] = None
    highlights: Optional[List[str]] = None


class ResumeProjectEntry(BaseModel):
    title: str
    description: Optional[str] = None
    impact: Optional[str] = None
    tools: Optional[List[str]] = None
    link: Optional[str] = None


class ResumeExperienceEntry(BaseModel):
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    achievements: Optional[List[str]] = None


class AIResumeGenerateRequest(BaseModel):
    name: str
    domain: str
    contact: ResumeContact
    skills: List[str]
    education: List[ResumeEducationEntry]
    projects: Optional[List[ResumeProjectEntry]] = None
    experience: Optional[List[ResumeExperienceEntry]] = None
    summary: Optional[str] = None


class AIResumeSection(BaseModel):
    heading: str
    items: List[Dict[str, Any]]


class AIResumeGenerateResponse(BaseModel):
    resume_id: int
    ats_score: int
    structured_resume: Dict[str, Any]
    preview_html: str
    generated_pdf_url: Optional[str] = None
    generated_docx_url: Optional[str] = None


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


class ProjectResource(BaseModel):
    title: str
    url: str
    kind: str
    provider: Optional[str] = None


class ProjectStepSchema(BaseModel):
    id: str
    title: str
    description: str
    xp_reward: int
    resources: List[ProjectResource] = []


class ProjectSummary(BaseModel):
    id: str
    domain: str
    title: str
    difficulty: str
    time_commitment: str
    estimated_time: str
    tech_stack: List[str]
    tags: List[str]
    summary: str
    market_alignment: str
    progress_status: Optional[str] = None
    completed_steps: Optional[int] = 0
    total_steps: Optional[int] = 0


class ProjectDetail(ProjectSummary):
    why_it_matters: str
    description: str
    dataset: Optional[str] = None
    expected_output: Optional[str] = None
    github_template: Optional[str] = None
    resources: List[ProjectResource]
    steps: List[ProjectStepSchema]


class ProjectProgressEntry(BaseModel):
    project_id: str
    status: str
    completed_steps: List[str]
    total_steps: int
    github_link_submitted: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectProgressResponse(BaseModel):
    project: ProjectDetail
    progress: Optional[ProjectProgressEntry] = None


class ProjectStartRequest(BaseModel):
    overwrite: bool = False


class ProjectStepUpdateRequest(BaseModel):
    step_id: str


class ProjectSubmitRequest(BaseModel):
    github_url: str


class CodingTestCase(BaseModel):
    name: str
    hint: Optional[str] = None
    input: Any | None = None
    kwargs: Optional[Dict[str, Any]] = None
    expected: Any | None = None


class CodingTaskSummary(BaseModel):
    task_id: str
    track_id: str
    title: str
    difficulty: str
    estimate_minutes: int
    skill_tag: Optional[str] = None
    category: str = "Beginner"
    order: int = 0
    xp: int = 0
    status: str = "not-started"


class CodingTaskListResponse(BaseModel):
    tasks: List[CodingTaskSummary]


class CodingTaskDetail(CodingTaskSummary):
    prompt: Optional[str] = None
    description: Optional[str] = None
    language: str
    starter_code: str
    entry_function: Optional[str] = None
    hints: List[str] = []
    resources: List[Dict[str, Any]] = []
    sample_cases: List[Dict[str, Any]] = []
    tests: List[CodingTestCase] = []
    is_project: bool = False


class CodingRunRequest(BaseModel):
    code: str


class CodingRunTestResult(BaseModel):
    name: str
    passed: bool
    hint: Optional[str] = None
    expected: Any | None = None
    received: Any | None = None
    message: Optional[str] = None
    error: Optional[str] = None


class TestResult(CodingRunTestResult):
    """Backward compatible alias used by legacy endpoints."""


class CodingRunResponse(BaseModel):
    passed: bool
    score: float
    total_tests: int
    passed_tests: int
    results: List[CodingRunTestResult]
    execution_time_ms: Optional[float] = None
    detail: Optional[str] = None
    xp_awarded: Optional[int] = 0
    streak: Optional[int] = None
    total_xp: Optional[int] = None
    coding_xp: Optional[int] = None
    badges: List[str] = []
    track_completed: bool = False
    awarded_badge: Optional[str] = None
    error: Optional[str] = None


class CodingSubmitResponse(CodingRunResponse):
    pass


class CodingTrack(BaseModel):
    track_id: str
    name: str
    description: Optional[str] = None
    order: int
    difficulty: str
    xp: int
    total_tasks: int
    completed_tasks: Optional[int] = 0
    badge: Optional[str] = None
    cover_image: Optional[str] = None


class CodingTrackListResponse(BaseModel):
    tracks: List[CodingTrack]


class CodingSkillProgress(BaseModel):
    name: str
    mastery: float
    attempts: int = 0
    level: str = "Beginner"
    last_practiced_at: Optional[str] = None


class CodingNextSkill(BaseModel):
    skill: str
    difficulty: str
    reason: Optional[str] = None
    sources: List[str] = []
    recommended_resources: List[Dict[str, Any]] = []


class CodingOverviewProgress(BaseModel):
    xp: int = 0
    streak_days: int = 0
    pass_rate: float = 0.0
    mastery_rate: float = 0.0
    ladder_level: str = "Beginner"
    interview_ready_score: Optional[int] = None
    time_on_task_minutes: Optional[int] = None


class CodingOverviewWeeklyChallenge(BaseModel):
    title: str
    description: str
    difficulty: str
    reward_xp: int
    trend_reason: Optional[str] = None
    reference_links: List[str] = []


class CodingOverviewResponse(BaseModel):
    domain: str
    focus_skill: Optional[str] = None
    progress: CodingOverviewProgress
    next_skills: List[CodingNextSkill] = []
    skill_progress: List[CodingSkillProgress] = []
    current_track: Optional[Dict[str, Any]] = None
    weekly_challenge: Optional[CodingOverviewWeeklyChallenge] = None


class CodingLessonResponse(BaseModel):
    skill: str
    title: str
    markdown: str
    demo_code: Optional[str] = None
    estimated_minutes: int = 10
    steps: List[str] = []
    references: List[Dict[str, Any]] = []


class CodingQuestionPayload(BaseModel):
    question_id: str
    skill: str
    title: str
    prompt: str
    difficulty: str
    language: str
    starter_code: str
    tests: List[Dict[str, Any]]
    hints: List[str] = []
    tags: List[str] = []
    estimated_minutes: Optional[int] = None
    entry_function: Optional[str] = None
    guided_steps: List[Dict[str, Any]] = []
    resources: List[Dict[str, Any]] = []
    walkthrough: List[str] = []
    practice_question: Optional[Dict[str, Any]] = None


class CodingQuestionResponse(BaseModel):
    question: CodingQuestionPayload
    playlist_source: Optional[str] = None
    suggested_followups: List[str] = []


class CodingHintRequest(BaseModel):
    question_id: str
    code_snapshot: Optional[str] = None
    attempt: Optional[int] = None


class CodingHintResponse(BaseModel):
    hint: str
    additional_resources: List[str] = []


class CodingTrackProgressRequest(BaseModel):
    track_id: str
    item_id: str
    status: str  # "not-started" | "in-progress" | "completed"


class CodingTrackProgressResponse(BaseModel):
    track_id: str
    item_id: str
    status: str
    completed_items: List[str]
    updated_at: datetime


class CodingQuestionSubmission(BaseModel):
    question_id: str
    code: str
    language: Optional[str] = None
    runtime_ms: Optional[float] = None
    metadata: Dict[str, Any] = {}
