"""
Advanced Skill Categorization with NLP
"""
import re
from typing import Dict, List


class SkillCategorizer:
    """Categorize skills into Technical, Tools, Soft, Languages."""
    
    # Skill dictionaries (in production, use ML model + embeddings)
    TECHNICAL_SKILLS = {
        'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala', 'r'],
        'algorithms': ['data structures', 'algorithms', 'oop', 'design patterns', 'system design'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch'],
        'ml_ai': ['machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision', 'reinforcement learning'],
    }
    
    TOOLS_FRAMEWORKS = {
        'web': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'express', 'spring', 'laravel'],
        'ml': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'opencv'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'github actions'],
        'devops': ['git', 'ci/cd', 'jenkins', 'gitlab', 'github', 'jira', 'confluence'],
    }
    
    SOFT_SKILLS = [
        'leadership', 'communication', 'teamwork', 'problem-solving', 'critical thinking',
        'collaboration', 'adaptability', 'time management', 'project management', 'mentoring',
    ]
    
    LANGUAGES = [
        'english', 'hindi', 'spanish', 'french', 'mandarin', 'german', 'japanese', 'korean',
        'telugu', 'tamil', 'bengali', 'marathi', 'gujarati', 'kannada',
    ]
    
    def categorize(self, resume_text: str) -> Dict[str, List[str]]:
        """
        Extract and categorize skills from resume text.
        
        Returns:
        {
            'technical': List[str],
            'tools': List[str],
            'soft': List[str],
            'languages': List[str]
        }
        """
        text_lower = resume_text.lower()
        
        # Extract skills using keyword matching
        technical = self._extract_technical_skills(text_lower)
        tools = self._extract_tools(text_lower)
        soft = self._extract_soft_skills(text_lower)
        languages = self._extract_languages(text_lower)
        
        return {
            'technical': list(set(technical)),
            'tools': list(set(tools)),
            'soft': list(set(soft)),
            'languages': list(set(languages)),
        }
    
    def _extract_technical_skills(self, text: str) -> List[str]:
        """Extract technical/programming skills."""
        skills = []
        
        for category, keywords in self.TECHNICAL_SKILLS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    skills.append(keyword.title())
        
        return skills
    
    def _extract_tools(self, text: str) -> List[str]:
        """Extract tools and frameworks."""
        skills = []
        
        for category, keywords in self.TOOLS_FRAMEWORKS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    skills.append(keyword.title())
        
        return skills
    
    def _extract_soft_skills(self, text: str) -> List[str]:
        """Extract soft skills."""
        skills = []
        
        for skill in self.SOFT_SKILLS:
            if skill.lower() in text:
                skills.append(skill.title())
        
        return skills
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract languages."""
        languages = []
        
        for lang in self.LANGUAGES:
            if lang.lower() in text:
                languages.append(lang.title())
        
        return languages
    
    def find_missing_skills(self, current_skills: Dict[str, List[str]], domain: str) -> List[str]:
        """Find missing trending skills for the domain."""
        # Domain-specific recommended skills (in production, pull from job trends DB)
        recommended_skills = {
            'AI/ML': ['TensorFlow', 'PyTorch', 'MLOps', 'LLM', 'Hugging Face', 'LangChain', 'Vector DB'],
            'Data Science': ['Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly', 'Tableau', 'Power BI'],
            'Web Development': ['React', 'Node.js', 'TypeScript', 'Next.js', 'GraphQL', 'REST APIs'],
            'Cybersecurity': ['Penetration Testing', 'SIEM', 'OWASP', 'Network Security', 'Ethical Hacking'],
            'Cloud Computing': ['AWS', 'Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'Serverless'],
        }
        
        domain_skills = recommended_skills.get(domain, [])
        all_current = current_skills.get('technical', []) + current_skills.get('tools', [])
        all_current_lower = [s.lower() for s in all_current]
        
        missing = [skill for skill in domain_skills if skill.lower() not in all_current_lower]
        
        return missing[:10]  # Top 10 missing skills

