"""
AI Service for OpenAI/Gemini Integration
"""
import os
import json
from typing import Dict, List, Optional
import httpx


class AIService:
    """Unified AI service supporting OpenAI and Gemini."""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.preferred_model = os.getenv("AI_MODEL", "gemini")  # "openai" or "gemini"
    
    async def generate_resume_analysis(
        self,
        resume_text: str,
        ats_score: int,
        domain: str,
        categorized_skills: Dict[str, List[str]],
        faults: List[Dict]
    ) -> Dict:
        """
        Generate comprehensive AI analysis of resume.
        
        Returns:
        {
            "remarks": str,
            "improvements": List[str],
            "explanation": str,
            "trending_skills": List[str],
            "suggested_certifications": List[str]
        }
        """
        
        # Build comprehensive prompt
        prompt = self._build_analysis_prompt(resume_text, ats_score, domain, categorized_skills, faults)
        
        try:
            if self.preferred_model == "openai" and self.openai_key:
                return await self._call_openai(prompt)
            elif self.gemini_key:
                return await self._call_gemini(prompt)
            else:
                # Fallback to rule-based analysis that uses the actual resume text
                return self._fallback_analysis(ats_score, domain, categorized_skills, faults, resume_text)
        except Exception as e:
            print(f"AI service error: {e}")
            return self._fallback_analysis(ats_score, domain, categorized_skills, faults, resume_text)
    
    async def generate_resume_content(
        self,
        user_data: Dict,
        domain: str,
        template_structure: str
    ) -> str:
        """
        Generate AI-powered resume content matching template structure.
        
        Args:
            user_data: Dict with personal, education, experience, projects, skills, achievements
            domain: Target domain (AI/ML, Cybersecurity, etc.)
            template_structure: Description of template structure
        
        Returns:
            HTML resume content
        """
        
        prompt = self._build_resume_generation_prompt(user_data, domain, template_structure)
        
        try:
            if self.preferred_model == "openai" and self.openai_key:
                content = await self._call_openai_text(prompt)
                return self._format_as_html(content, user_data)
            elif self.gemini_key:
                content = await self._call_gemini_text(prompt)
                return self._format_as_html(content, user_data)
            else:
                return self._fallback_resume_generation(user_data, domain)
        except Exception as e:
            print(f"AI resume generation error: {e}")
            return self._fallback_resume_generation(user_data, domain)
    
    def _build_analysis_prompt(
        self,
        resume_text: str,
        ats_score: int,
        domain: str,
        categorized_skills: Dict[str, List[str]],
        faults: List[Dict]
    ) -> str:
        """Build comprehensive analysis prompt."""
        
        return f"""You are GradGear AI, an expert resume analyst and career advisor.

Analyze this resume for a {domain} professional:

RESUME TEXT:
{resume_text[:3000]}

CURRENT ATS SCORE: {ats_score}/100
DOMAIN: {domain}
CURRENT SKILLS: {', '.join(categorized_skills.get('technical', [])[:10])}

DETECTED FAULTS:
{json.dumps([f.get('message', '') for f in faults[:5]], indent=2)}

Provide a comprehensive analysis in JSON format:
{{
    "remarks": "2-3 sentence overall assessment focusing on strengths and key gaps",
    "improvements": [
        "Specific actionable improvement 1",
        "Specific actionable improvement 2",
        "Specific actionable improvement 3"
    ],
    "explanation": "Brief explanation of how to raise ATS score by X% with specific steps",
    "trending_skills": ["Skill 1", "Skill 2", "Skill 3"],
    "suggested_certifications": ["Cert 1", "Cert 2"]
}}

Focus on:
1. Domain-specific keywords and trends for {domain}
2. Measurable improvements (quantify potential ATS increase)
3. Trending skills in {domain} job market
4. Industry-standard certifications for {domain}
5. Actionable, specific recommendations

Return ONLY valid JSON, no markdown or extra text.
"""
    
    def _build_resume_generation_prompt(self, user_data: Dict, domain: str, template_structure: str) -> str:
        """Build resume generation prompt."""
        
        personal = user_data.get('personal', {})
        education = user_data.get('education', [])
        experience = user_data.get('experience', [])
        projects = user_data.get('projects', [])
        skills = user_data.get('skills', {})
        achievements = user_data.get('achievements', [])
        
        return f"""You are GradGear AI. Generate a plagiarism-free, ATS-optimized resume
for a {domain} professional using this structure:
- Header: Name + Contact (email, LinkedIn, GitHub)
- About Me: 2-3 line professional summary
- Education: Degrees with institutions, years, CGPA
- Tools & Technologies: Categorized skills
- Projects: Title, description, tech stack, impact
- Achievements: Quantified accomplishments
- Core Skills: Technical skills list
- Languages: If provided

USER DATA:
Name: {personal.get('name', '')}
Email: {personal.get('email', '')}
LinkedIn: {personal.get('linkedin', '')}
GitHub: {personal.get('github', '')}

Education: {json.dumps(education, indent=2)}
Experience: {json.dumps(experience, indent=2)}
Projects: {json.dumps(projects, indent=2)}
Skills: {json.dumps(skills, indent=2)}
Achievements: {json.dumps(achievements, indent=2)}

Requirements:
1. Follow the template structure exactly (match Nishanth_final_resume.pdf layout)
2. Use domain-specific keywords for {domain}
3. Start bullet points with action verbs (Led, Developed, Improved, etc.)
4. Include quantifiable metrics (numbers, percentages)
5. Professional, concise, ATS-friendly formatting
6. Plagiarism-free, original content
7. Return structured text that can be formatted as HTML (not HTML tags, just structured text)

Generate the resume content now:
"""
    
    async def _call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API for analysis."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are an expert resume analyst. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1500
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                # Extract JSON from response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
                return json.loads(content)
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _call_openai_text(self, prompt: str) -> str:
        """Call OpenAI API for text generation."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are an expert resume writer. Generate professional, ATS-optimized resume content."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def _call_gemini(self, prompt: str) -> Dict:
        """Call Gemini API for analysis."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 1500
                        }
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                # Extract JSON
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
                return json.loads(content)
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def _call_gemini_text(self, prompt: str) -> str:
        """Call Gemini API for text generation."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 2000
                        }
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _fallback_analysis(
        self,
        ats_score: int,
        domain: str,
        categorized_skills: Dict[str, List[str]],
        faults: List[Dict],
        resume_text: str
    ) -> Dict:
        """Fallback rule-based analysis when AI is unavailable. Uses resume text for personalization."""
        
        # Basic strength extraction from detected skills
        strengths = []
        for cat in ("technical", "tools", "soft"):
            strengths.extend(categorized_skills.get(cat, [])[:5])
        strengths = strengths[:8]

        remarks = f"ATS {ats_score}/100 for {domain}. Strong in: {', '.join(strengths[:3]) if strengths else 'foundational skills'}."
        if ats_score >= 80:
            remarks += " Resume is solid; focus on impact metrics and domain certs."
        elif ats_score >= 60:
            remarks += " Improve keyword density and quantify achievements with metrics."
        else:
            remarks += " Strengthen sections, add domain keywords, and quantify outcomes."
        
        # Improvements from faults + text heuristics
        improvements = [f.get('suggestion', '') for f in faults if f.get('suggestion')][:5]
        # Heuristic: suggest metrics if numbers are scarce
        import re
        if len(re.findall(r"\b(\d+%|\d+\s+(users|projects|models|datasets))\b", resume_text.lower())) < 2:
            improvements.append("Add quantifiable metrics (e.g., 'Improved inference latency by 25% for 10k users').")
        if len(re.findall(r"\b(led|developed|implemented|optimized|designed)\b", resume_text.lower())) < 3:
            improvements.append("Start bullets with strong action verbs (Led, Developed, Implemented, Optimized).")
        improvements = improvements[:5]
        
        explanation = (
            f"Increase ATS by 10-20% by adding {domain} keywords, quantifying outcomes, and including relevant certifications."
        )
        
        # Domain-specific trending skills
        trending_map = {
            'AI/ML': ['LLM', 'MLOps', 'Vector Databases', 'LangChain', 'Hugging Face'],
            'Cybersecurity': ['Incident Response', 'Threat Analysis', 'SIEM', 'Penetration Testing', 'Cloud Security'],
            'Data Science': ['Data Engineering', 'MLOps', 'Feature Engineering', 'A/B Testing', 'Time Series'],
            'Web Development': ['Next.js', 'TypeScript', 'GraphQL', 'Microservices', 'Serverless'],
            'Cloud Computing': ['Kubernetes', 'Terraform', 'Serverless', 'Multi-cloud', 'DevOps'],
        }
        
        # Domain-specific certifications
        cert_map = {
            'AI/ML': ['TensorFlow Developer Certificate', 'AWS Machine Learning Specialty'],
            'Cybersecurity': ['CEH', 'CompTIA Security+', 'CISSP', 'AWS Security Specialty'],
            'Data Science': ['AWS Certified Data Analytics', 'Google Data Analytics Certificate'],
            'Web Development': ['AWS Certified Developer', 'Google Cloud Professional Cloud Developer'],
            'Cloud Computing': ['AWS Solutions Architect', 'Kubernetes Administrator', 'Terraform Associate'],
        }
        
        return {
            "remarks": remarks,
            "improvements": improvements[:5],
            "explanation": explanation,
            "trending_skills": trending_map.get(domain, [])[:5],
            "suggested_certifications": cert_map.get(domain, [])[:3],
        }
    
    def _fallback_resume_generation(self, user_data: Dict, domain: str) -> str:
        """Fallback resume generation without AI."""
        # Use template-based generation (already implemented in resume.py)
        return ""
    
    def _format_as_html(self, ai_content: str, user_data: Dict) -> str:
        """Format AI-generated content as HTML matching template."""
        # Parse AI response and format according to template
        # This is a simplified version - in production, use more sophisticated parsing
        return ai_content  # Will be processed by template formatter

    async def generate_project_bullet(
        self,
        title: str,
        summary: str,
        tech_stack: List[str],
        impact_summary: str,
    ) -> str:
        """
        Generate a resume-ready bullet point for a completed portfolio project.
        """
        prompt = f"""You are GradGear AI, an expert resume writer.

Create ONE concise resume bullet (max 28 words) for the project "{title}".
Context:
- Project summary: {summary}
- Tech stack: {', '.join(tech_stack)}
- Impact: {impact_summary}

Constraints:
- Start with an action verb.
- Mention tech stack and quantifiable or directional impact (fabricate a realistic metric if none given).
- Past tense, ATS-optimised, no first person, no filler.
- Return only the bullet text without quotes or bullet characters.
"""
        try:
            if self.preferred_model == "openai" and self.openai_key:
                return (await self._call_openai_text(prompt)).strip().lstrip("-• ")
            if self.gemini_key:
                return (await self._call_gemini_text(prompt)).strip().lstrip("-• ")
        except Exception as exc:
            print(f"AI project bullet generation error: {exc}")
        return self._fallback_project_bullet(title, tech_stack, impact_summary)

    def _fallback_project_bullet(self, title: str, tech_stack: List[str], impact: str) -> str:
        stack = ", ".join(tech_stack[:3]) if tech_stack else "modern tooling"
        impact_phrase = impact.rstrip(".")
        return (
            f"Delivered {title.lower()} using {stack}, translating {impact_phrase.lower()} into a production-ready deliverable adopted during stakeholder review."
        )

