from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Body
from typing import List, Dict, Optional
from app.schemas import ResumeGenerateRequest, ResumeAnalysis, ResumeAnalyzeRequest
from app.routes.auth_ext import get_current_user
from app.models import User, Resume
from app.core.ats_scorer import ATSScorer
from app.core.plagiarism import PlagiarismChecker
from app.core.skill_categorizer import SkillCategorizer
from app.core.course_recommender import CourseRecommender
from app.core.ai_service import AIService
from app.core.trending_skills import TrendingSkillsDetector
from sqlalchemy.orm import Session
from app.db import get_db
import PyPDF2
import hashlib
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
from io import BytesIO
from datetime import datetime

router = APIRouter()
def _infer_domain_from_text(text: str) -> str:
    """Infer domain by matching ATS domain keyword sets."""
    ats_tmp = ATSScorer()
    domain_scores: Dict[str, int] = {}
    candidates = [
        "AI/ML", "Cybersecurity", "Data Science", "Web Development", "Cloud Computing", "IoT", "Robotics"
    ]
    text_l = text.lower()
    for dom in candidates:
        ats_tmp.domain = dom
        ats_tmp.domain_keywords = ats_tmp._get_domain_keywords(dom)
        score = 0
        for kw in ats_tmp.domain_keywords:
            if kw in text_l:
                score += 1
        domain_scores[dom] = score
    return max(domain_scores, key=domain_scores.get) if domain_scores else "AI/ML"


def _extract_top_keywords(text: str, top_n: int = 15) -> List[str]:
    """Naive keyword extraction by frequency filtering stopwords and short tokens."""
    import re
    from collections import Counter
    stop = set("""
a an the and or for of to in on with without into from by at as is are was were be been being that this those these it its their his her you your our we they them i me my mine ours us
""".split())
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-\+\.#]{1,}", text.lower())
    tokens = [t for t in tokens if t not in stop and len(t) >= 3]
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]

# Initialize core modules
skill_categorizer = SkillCategorizer()
course_recommender = CourseRecommender()
ai_service = AIService()
trending_detector = TrendingSkillsDetector()

# Template-based HTML resume generation (matches uploaded resume structure)
RESUME_TEMPLATE = """
<header style="margin-bottom: 20px; border-bottom: 2px solid #1e293b; padding-bottom: 15px;">
    <h1 style="font-size: 28px; font-weight: bold; margin-bottom: 8px; color: #1e293b;">{name}</h1>
    <div style="color: #64748b; font-size: 14px; line-height: 1.6;">
        {email}{linkedin}{github}
    </div>
</header>

<section style="margin-bottom: 20px;">
    <h2 style="font-size: 18px; font-weight: bold; text-transform: uppercase; color: #1e293b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">About Me</h2>
    <p style="line-height: 1.6; color: #475569; margin: 0;">{summary}</p>
</section>

<section style="margin-bottom: 20px;">
    <h2 style="font-size: 18px; font-weight: bold; text-transform: uppercase; color: #1e293b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">Education</h2>
    {education_html}
</section>

<section style="margin-bottom: 20px;">
    <h2 style="font-size: 18px; font-weight: bold; text-transform: uppercase; color: #1e293b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">Tools and Technologies</h2>
    <div style="line-height: 2;">
        {skills_html}
    </div>
</section>

<section style="margin-bottom: 20px;">
    <h2 style="font-size: 18px; font-weight: bold; text-transform: uppercase; color: #1e293b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">Projects</h2>
    {projects_html}
</section>

{achievements_html}

<section style="margin-bottom: 20px;">
    <h2 style="font-size: 18px; font-weight: bold; text-transform: uppercase; color: #1e293b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">Core Skills</h2>
    <div style="line-height: 1.8; color: #475569;">
        {core_skills_html}
    </div>
</section>

{languages_html}
"""


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        # Primary: PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        text = (text or "").strip()

        # Fallback: PyMuPDF if available and text seems too short
        if (not text or len(text) < 200) and fitz is not None:
            try:
                doc = fitz.open(stream=file_content, filetype="pdf")
                txt = []
                for pg in doc:
                    txt.append(pg.get_text("text") or "")
                fallback_text = "\n".join(txt).strip()
                if len(fallback_text) > len(text):
                    text = fallback_text
            except Exception:
                pass

        if not text or len(text) < 50:
            raise Exception("No extractable text found")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract DOCX text: {str(e)}")


def parse_resume_sections(text: str) -> Dict:
    """Parse resume text into structured sections (matching template)."""
    sections = {
        'about_me': '',
        'education': [],
        'tools_technologies': [],
        'projects': [],
        'achievements': [],
        'core_skills': [],
        'languages': [],
    }
    
    text_lower = text.lower()
    
    # Extract About Me / Summary
    if 'about me' in text_lower or 'summary' in text_lower:
        # Simple extraction (in production, use NLP)
        about_start = text_lower.find('about me')
        if about_start != -1:
            about_section = text[about_start:about_start+500].split('\n')[1:3]
            sections['about_me'] = ' '.join(about_section)
    
    # Extract Education (look for degree patterns)
    education_pattern = r'(b\.tech|bachelor|master|phd|degree).*?(\d{4}|\d{4}-\d{4})'
    import re
    education_matches = re.findall(education_pattern, text_lower)
    sections['education'] = [{'degree': m[0], 'period': m[1]} for m in education_matches]
    
    return sections


@router.post("/upload", response_model=ResumeAnalysis)
async def upload_resume(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and analyze resume."""
    try:
        if not file.filename or not file.filename.lower().endswith((".pdf", ".doc", ".docx")):
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or DOCX.")
        
        content = await file.read()
        
        # Extract text
        try:
            if file.filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(content)
            elif file.filename.lower().endswith((".doc", ".docx")):
                text = extract_text_from_docx(content)
            else:
                text = content.decode('utf-8', errors='ignore')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

        # Basic sanity check to ensure we're not storing empty/static content
        if not text or len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume text too short — parsing likely failed. Please upload a text-based PDF/DOCX.")
        
        # Store in database (with error handling)
        try:
            resume = Resume(
                user_id=user.id,
                name=file.filename or "uploaded_resume",
                raw_text=text[:50000] if text else "",  # Limit text size
                parsed_data=parse_resume_sections(text) if text else {},
                domain=user.domain or 'AI/ML',
            )
            db.add(resume)
            db.commit()
            db.refresh(resume)
        except Exception as db_error:
            db.rollback()
            # If Resume table doesn't exist, create it
            from app.db import Base, engine
            try:
                Base.metadata.create_all(bind=engine)
                db.add(resume)
                db.commit()
                db.refresh(resume)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e2)}")
        
        # Categorize skills
        categorized_skills = skill_categorizer.categorize(text)
        resume.categorized_skills = categorized_skills
        db.commit()
        
        # Calculate ATS score
        ats_scorer = ATSScorer(domain=user.domain or 'AI/ML')
        ats_score, faults = ats_scorer.score(text, resume.parsed_data)
        resume.ats_score = ats_score
        db.commit()
        
        # Check plagiarism
        plagiarism_checker = PlagiarismChecker()
        plagiarism_result = plagiarism_checker.check(text)
        resume.plagiarism_score = plagiarism_result['plagiarism_score']
        resume.plagiarism_safe = plagiarism_result['plagiarism_safe']
        db.commit()
        
        # Find missing skills
        missing_skills = skill_categorizer.find_missing_skills(categorized_skills, user.domain or 'AI/ML')
        
        # Generate suggestions
        suggestions = []
        for fault in faults:
            suggestions.append(fault.get('suggestion', ''))
        
        # Add skill recommendations
        if missing_skills:
            suggestions.append(f"Consider adding these trending skills: {', '.join(missing_skills[:5])}")
        
        return ResumeAnalysis(
            ats_score=ats_score,
            weak_points=[f.get('message', '') for f in faults],
            skill_gaps=missing_skills,
            suggestions=suggestions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/analyze")
async def analyze_resume(
    req: Optional[ResumeAnalyzeRequest] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deep analyze resume with full AI intelligence.
    
    Accepts JSON body: {"resume_id": int} or {} (uses latest resume)
    
    Returns comprehensive analysis with:
    - ATS score with sub-scores
    - AI-generated remarks and improvements
    - Trending skills detection
    - Certification recommendations
    - Course/video resources
    """
    resume_id = req.resume_id if req else None
    
    # Get resume
    if resume_id:
        resume = db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user.id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        text = resume.raw_text or ''
    else:
        # Get latest resume for user
        resume = db.query(Resume).filter(Resume.user_id == user.id).order_by(Resume.created_at.desc()).first()
        if not resume:
            raise HTTPException(status_code=404, detail="No resume found")
        text = resume.raw_text or ''
    
    inferred_domain = _infer_domain_from_text(text)
    domain = resume.domain or user.domain or inferred_domain or 'AI/ML'
    
    # 1. ATS Score with sub-scores
    ats_scorer = ATSScorer(domain=domain)
    ats_score, faults = ats_scorer.score(text, resume.parsed_data)
    
    # Calculate sub-scores (components)
    # Re-score to get component breakdown
    sub_scores = {
        'keyword': int((ats_scorer._score_keywords(text.lower())[0]) * 0.4),
        'section': int((ats_scorer._score_sections(text.lower(), resume.parsed_data)[0]) * 0.2),
        'format': int((ats_scorer._score_format(text.lower())[0]) * 0.15),
        'impact': int((ats_scorer._score_impact(text.lower())[0]) * 0.15),
        'conciseness': int((ats_scorer._score_length(text)[0]) * 0.10),
    }
    
    # 2. Plagiarism Check
    plagiarism_checker = PlagiarismChecker()
    plagiarism_result = plagiarism_checker.check(text)
    
    # 3. Skill Categorization
    categorized_skills = resume.categorized_skills or skill_categorizer.categorize(text)
    
    # 4. AI-Powered Analysis
    ai_analysis = await ai_service.generate_resume_analysis(
        resume_text=text,
        ats_score=ats_score,
        domain=domain,
        categorized_skills=categorized_skills,
        faults=faults
    )
    
    # 5. Trending Skills Detection
    all_current_skills = (
        categorized_skills.get('technical', []) +
        categorized_skills.get('tools', []) +
        categorized_skills.get('soft', [])
    )
    trending_data = trending_detector.get_trending_skills(domain, all_current_skills)
    
    # 6. Certification Recommendations
    certifications = trending_detector.get_recommended_certifications(domain)
    
    # 7. Missing Skills & Course Recommendations
    missing_skills = trending_data.get('missing', []) or skill_categorizer.find_missing_skills(categorized_skills, domain)
    recommended_resources = []
    
    for skill in missing_skills[:5]:  # Top 5 missing skills
        courses = await course_recommender.recommend(skill, domain)
        recommended_resources.append({
            'skill': skill,
            'resources': courses,
        })
    
    # Build comprehensive response
    improvements = ai_analysis.get('improvements', [])
    if not improvements:
        improvements = [f.get('suggestion', '') for f in faults[:3]]

    # Resume-specific keyword signals for personalization
    top_keywords = _extract_top_keywords(text, top_n=20)
    domain_keywords = ATSScorer(domain).domain_keywords
    found_keywords = [kw for kw in domain_keywords if kw in text.lower()]
    missing_keywords = [kw for kw in domain_keywords if kw not in text.lower()][:10]
    
    # Compute unique hash of current resume text (first 2KB window) to verify uniqueness end-to-end
    unique_hash = hashlib.md5((text[:2048] or "").encode('utf-8', errors='ignore')).hexdigest()

    return {
        'ats_score': ats_score,
        'sub_scores': sub_scores,
        'remarks': ai_analysis.get('remarks', f'Resume analysis for {domain} domain. Current ATS score: {ats_score}/100.'),
        'faults': faults,
        'improvements': improvements,
        'missing_skills': missing_skills[:5],
        'trending_skills': {
            'high_demand': trending_data.get('high_demand', [])[:5],
            'emerging': trending_data.get('emerging', [])[:3],
        },
        'suggested_certifications': [
            {'name': c['name'], 'provider': c['provider'], 'level': c['level']}
            for c in certifications
        ],
        'recommended_resources': recommended_resources,
        'categorized_skills': categorized_skills,
        'plagiarism': {
            'plagiarism_score': plagiarism_result['plagiarism_score'],
            'plagiarism_safe': plagiarism_result['plagiarism_safe'],
            'flagged_passages': plagiarism_result.get('flagged_passages', [])[:5],
        },
        'domain': domain,
        'explanation': ai_analysis.get('explanation', f'Adding metrics and domain certs can raise ATS by 10-15%.'),
        'unique_hash': unique_hash,
        'found_keywords': found_keywords[:15],
        'missing_keywords': missing_keywords[:15],
        'top_keywords': top_keywords,
    }


@router.post("/generate")
async def generate_resume(req: ResumeGenerateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Generate AI-powered resume matching template structure."""
    
    personal = req.personal or {}
    education = req.education or []
    experience = req.experience or []
    projects = req.projects or []
    
    # Prepare user data for AI generation
    user_data = {
        'personal': personal,
        'education': education,
        'experience': experience,
        'projects': projects,
        'skills': {
            'technical': req.skills[:len(req.skills)//2] if req.skills else [],
            'tools': req.skills[len(req.skills)//2:] if req.skills else [],
        },
        'achievements': req.achievements,
    }
    
    # Try AI-powered generation first
    try:
        ai_content = await ai_service.generate_resume_content(
            user_data=user_data,
            domain=req.domain,
            template_structure="Header, About Me, Education, Tools & Technologies, Projects, Achievements, Core Skills, Languages"
        )
        if ai_content:
            # Use AI-generated content
            html_content = ai_content
        else:
            raise Exception("AI generation failed, using template")
    except Exception as e:
        print(f"AI generation error, using template: {e}")
        # Fallback to template-based generation
        html_content = await _generate_template_resume(req, user, personal, education, experience, projects)
    
    # Calculate ATS score
    ats_scorer = ATSScorer(domain=req.domain)
    ats_score, _ = ats_scorer.score(html_content, {'education': education, 'projects': projects})
    
    # Check plagiarism
    plagiarism_checker = PlagiarismChecker()
    plagiarism_result = plagiarism_checker.check(html_content)
    
    # Track progress
    try:
        import json
        from app.models import Progress
        progress = db.query(Progress).filter(Progress.user_id == user.id).first()
        if not progress:
            progress = Progress(user_id=user.id, data={})
            db.add(progress)
        
        if 'activities' not in progress.data:
            progress.data['activities'] = []
        
        progress.data['activities'].append({
            'type': 'resume_generated',
            'data': {'ats_score': ats_score, 'domain': req.domain},
            'timestamp': datetime.utcnow().isoformat(),
        })
        
        # Award badge if ATS score >= 85
        badges = progress.data.get('badges', [])
        if ats_score >= 85 and 'Resume Optimized' not in badges:
            badges.append('Resume Optimized')
            progress.data['badges'] = badges
        
        db.commit()
    except Exception as e:
        print(f"Progress tracking error: {e}")
        pass  # Don't fail if tracking fails
    
    return {
        "content": html_content,
        "ats_score": ats_score,
        "plagiarism_safe": plagiarism_result['plagiarism_safe'],
    }


async def _generate_template_resume(req, user, personal, education, experience, projects):
    """Template-based resume generation (fallback)."""
    # Build resume HTML matching template
    # Header
    email = personal.get('email') or user.email
    linkedin = f" | <a href='{personal.get('linkedin', '')}' target='_blank'>LinkedIn</a>" if personal.get('linkedin') else ''
    github = f" | <a href='{personal.get('github', '')}' target='_blank'>GitHub</a>" if personal.get('github') else ''
    
    # Summary
    summary = personal.get('summary', '') or f"Motivated {req.domain} professional with experience in {', '.join(req.skills[:3])}. Proven track record of delivering impactful projects."
    
    # Education HTML
    education_html = ''.join([
        f"<div style='margin-bottom: 12px;'><strong>{edu.get('degree', '')}</strong> {edu.get('year', '')}<br>{edu.get('institution', '')}"
        + (f" CGPA: {edu.get('gpa', '')}" if edu.get('gpa') else '') + "</div>"
        for edu in education
    ]) or "<div>No education entries</div>"
    
    # Skills HTML (Tools & Technologies section)
    all_skills = req.skills if isinstance(req.skills, list) else []
    skills_html = f"<div><strong>Programming Languages:</strong> {', '.join(all_skills[:5])}</div>"
    if len(all_skills) > 5:
        skills_html += f"<div><strong>Frameworks & Tools:</strong> {', '.join(all_skills[5:10])}</div>"
    
    # Projects HTML
    projects_html = ''.join([
        f"<div style='margin-bottom: 15px;'><strong>{proj.get('title', '')}</strong><br>"
        f"<div style='color: #64748b; font-size: 14px; margin-top: 4px;'>{proj.get('description', '')}</div>"
        + (f"<div style='color: #64748b; font-size: 13px; margin-top: 4px;'>Tech Stack: {proj.get('tools', '')}</div>" if proj.get('tools') else '') + "</div>"
        for proj in projects
    ]) or "<div>No projects</div>"
    
    # Achievements HTML
    achievements_html = ''
    if req.achievements:
        achievements_html = f"""
<section style="margin-bottom: 20px;">
    <h2 style="font-size: 18px; font-weight: bold; text-transform: uppercase; color: #1e293b; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px;">Achievements</h2>
    <ul style="padding-left: 20px; line-height: 1.8;">
        {''.join(f'<li style="color: #475569; margin-bottom: 8px;">{ach}</li>' for ach in req.achievements)}
    </ul>
</section>
"""
    
    # Core Skills HTML
    core_skills_html = ', '.join(all_skills[:15])
    
    # Languages HTML
    languages_html = ''
    
    # Generate HTML using template
    return RESUME_TEMPLATE.format(
        name=req.name,
        email=email,
        linkedin=linkedin,
        github=github,
        summary=summary,
        education_html=education_html,
        skills_html=skills_html,
        projects_html=projects_html,
        achievements_html=achievements_html,
        core_skills_html=core_skills_html,
        languages_html=languages_html,
    )


@router.get("/recommendations/skills")
async def get_skill_recommendations(
    domain: str,
    skills: str = None,
    current_user: User = Depends(get_current_user)
):
    """Get course/video recommendations for missing skills."""
    current_skills_list = skills.split(',') if skills else []
    current_skills_dict = {
        'technical': current_skills_list,
        'tools': [],
        'soft': [],
        'languages': [],
    }
    
    missing_skills = skill_categorizer.find_missing_skills(current_skills_dict, domain)
    recommendations = []
    
    for skill in missing_skills[:5]:
        courses = course_recommender.recommend(skill, domain)
        recommendations.append({
            'skill': skill,
            'courses': courses,
        })
    
    return {'recommendations': recommendations}


@router.post("/track-progress")
async def track_resume_progress(
    activity_type: str = Form(...),  # 'course_clicked', 'quiz_passed', 'resume_generated'
    activity_data: str = Form(...),  # JSON string
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Track user activity for progress and badges."""
    import json
    try:
        data = json.loads(activity_data)
    except:
        data = {}
    
    # Get or create progress
    from app.models import Progress
    progress = db.query(Progress).filter(Progress.user_id == current_user.id).first()
    if not progress:
        progress = Progress(user_id=current_user.id, data={})
        db.add(progress)
    
    # Update progress data
    if 'activities' not in progress.data:
        progress.data['activities'] = []
    
    progress.data['activities'].append({
        'type': activity_type,
        'data': data,
        'timestamp': datetime.utcnow().isoformat(),
    })
    
    # Award badges logic
    badges = progress.data.get('badges', [])
    
    # Resume Optimized badge
    if activity_type == 'resume_generated' and data.get('ats_score', 0) >= 85:
        if 'Resume Optimized' not in badges:
            badges.append('Resume Optimized')
            progress.data['badges'] = badges
    
    # Quiz badges
    if activity_type == 'quiz_passed':
        quiz_score = data.get('score', 0)
        if quiz_score >= 90 and 'Quiz Master' not in badges:
            badges.append('Quiz Master')
        elif quiz_score >= 80 and 'Quiz Expert' not in badges:
            badges.append('Quiz Expert')
        elif quiz_score >= 70 and 'Quiz Scholar' not in badges:
            badges.append('Quiz Scholar')
        progress.data['badges'] = badges
    
    db.commit()
    
    return {'success': True, 'badges': badges}
