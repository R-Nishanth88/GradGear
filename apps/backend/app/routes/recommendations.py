from fastapi import APIRouter, Depends, Query
from app.schemas import RecommendationItem, SkillRecommendation, CourseRecommendation
from app.routes.auth_ext import get_current_user
from app.models import User
from typing import List

router = APIRouter()


@router.get("/recommendations/{domain:path}")
async def get_recommendations(domain: str, user: User = Depends(get_current_user)) -> dict:
    """Get domain-specific recommendations."""
    # Decode domain if URL encoded (e.g., AI%2FML -> AI/ML)
    import urllib.parse
    domain = urllib.parse.unquote(domain)
    
    # Stubbed recommendations; replace with YouTube/Coursera APIs + Spark filters
    items = [
        RecommendationItem(title=f"{domain} Crash Course", url="https://www.youtube.com/", source="YouTube"),
        RecommendationItem(title=f"{domain} Specialization", url="https://www.coursera.org/", source="Coursera"),
        RecommendationItem(title=f"{domain} Projects", url="https://github.com/search", source="GitHub"),
    ]
    return {"domain": domain, "items": [i.model_dump() for i in items]}


@router.get("/resume/recommendations/skills", response_model=List[SkillRecommendation])
async def get_skill_recommendations(
    domain: str = Query("ai_ml", description="Domain to get skill recommendations for"),
    user: User = Depends(get_current_user)
) -> List[SkillRecommendation]:
    """Get skill recommendations for resume building."""
    # Decode domain if URL encoded
    import urllib.parse
    domain = urllib.parse.unquote(domain)
    
    # Map domain to trending skills
    skill_map = {
        "ai_ml": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision"],
        "data": ["Python", "Pandas", "SQL", "Machine Learning", "Data Visualization", "Statistics"],
        "cybersec": ["Penetration Testing", "SIEM", "Incident Response", "Threat Analysis", "Cloud Security", "Firewall"],
        "web": ["React", "Node.js", "TypeScript", "Next.js", "GraphQL", "REST API"],
        "cloud": ["AWS", "Azure", "Kubernetes", "Docker", "Terraform", "CI/CD"],
        "iot": ["Embedded Systems", "MQTT", "Arduino", "Raspberry Pi", "Sensor Networks", "Edge Computing"]
    }
    
    # Normalize domain input
    domain_normalized = domain.lower().replace(" ", "_").replace("/", "_")
    if domain_normalized not in skill_map:
        domain_normalized = "ai_ml"
    
    skills = skill_map[domain_normalized]
    
    # Create skill recommendations with courses
    recommendations = []
    for skill in skills:
        courses = [
            CourseRecommendation(
                title=f"{skill} Fundamentals",
                source="YouTube",
                url="https://www.youtube.com/",
                estimated_time="2 hours",
                level="Beginner",
                reason=f"Essential skill for {domain}"
            ),
            CourseRecommendation(
                title=f"Advanced {skill}",
                source="Coursera",
                url="https://www.coursera.org/",
                estimated_time="4 weeks",
                level="Advanced",
                reason=f"Deep dive into {skill}"
            )
        ]
        recommendations.append(SkillRecommendation(skill=skill, courses=courses))
    
    return recommendations


