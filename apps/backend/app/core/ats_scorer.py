"""
ATS Scoring Model - Comprehensive explainable ATS scoring
"""
import re
from typing import List, Dict, Tuple


class ATSScorer:
    """ATS Scoring Model with explainable components."""
    
    # Domain-specific keywords (in production, pull from job trends DB)
    DOMAIN_KEYWORDS = {
        'AI/ML': ['machine learning', 'deep learning', 'neural networks', 'tensorflow', 'pytorch', 'nlp', 'computer vision', 'ai', 'ml', 'scikit-learn', 'keras'],
        'Data Science': ['data analysis', 'sql', 'pandas', 'numpy', 'statistics', 'visualization', 'tableau', 'power bi', 'data modeling'],
        'Web Development': ['react', 'node.js', 'javascript', 'html', 'css', 'api', 'rest', 'frontend', 'backend', 'full stack'],
        'Cybersecurity': ['penetration testing', 'vulnerability assessment', 'siem', 'firewall', 'encryption', 'security', 'owasp'],
        'Cloud Computing': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'terraform', 'cloud', 'devops'],
    }
    
    REQUIRED_SECTIONS = ['summary', 'about me', 'education', 'skills', 'experience', 'projects']
    
    def __init__(self, domain: str = 'AI/ML'):
        self.domain = domain
        self.domain_keywords = self.DOMAIN_KEYWORDS.get(domain, [])
    
    def score(self, resume_text: str, structured_data: Dict = None) -> Tuple[int, List[Dict]]:
        """
        Calculate ATS score (0-100) and return faults.
        
        Components:
        1. Keyword Match (40%) - domain keywords in resume
        2. Section Presence (20%) - required sections present
        3. Format Safety (15%) - ATS-friendly formatting
        4. Action & Impact (15%) - quantifiable results
        5. Length & Conciseness (10%) - appropriate length
        """
        text_lower = resume_text.lower()
        faults = []
        scores = {}
        
        # 1. Keyword Match (40%)
        keyword_score, keyword_faults = self._score_keywords(text_lower)
        scores['keywords'] = keyword_score
        faults.extend(keyword_faults)
        
        # 2. Section Presence (20%)
        section_score, section_faults = self._score_sections(text_lower, structured_data)
        scores['sections'] = section_score
        faults.extend(section_faults)
        
        # 3. Format Safety (15%)
        format_score, format_faults = self._score_format(text_lower)
        scores['format'] = format_score
        faults.extend(format_faults)
        
        # 4. Action & Impact (15%)
        impact_score, impact_faults = self._score_impact(text_lower)
        scores['impact'] = impact_score
        faults.extend(impact_faults)
        
        # 5. Length & Conciseness (10%)
        length_score, length_faults = self._score_length(resume_text)
        scores['length'] = length_score
        faults.extend(length_faults)
        
        # Weighted total
        total_score = (
            keyword_score * 0.40 +
            section_score * 0.20 +
            format_score * 0.15 +
            impact_score * 0.15 +
            length_score * 0.10
        )
        
        return int(total_score), faults
    
    def _score_keywords(self, text_lower: str) -> Tuple[int, List[Dict]]:
        """Score keyword matching (0-100)."""
        found_keywords = []
        missing_keywords = []
        
        for keyword in self.domain_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Score: percentage of keywords found
        if not self.domain_keywords:
            return 100, []
        
        score = int((len(found_keywords) / len(self.domain_keywords)) * 100)
        faults = []
        
        if missing_keywords:
            faults.append({
                'type': 'missing_keywords',
                'severity': 'high',
                'message': f"Missing domain keywords: {', '.join(missing_keywords[:5])}",
                'suggestion': f"Add these keywords naturally: {', '.join(missing_keywords[:3])}. Include them in your projects, skills, or experience sections.",
                'example': f"Instead of 'worked on ML project', write 'Developed machine learning models using TensorFlow for...'"
            })
        
        return min(100, score), faults
    
    def _score_sections(self, text_lower: str, structured_data: Dict = None) -> Tuple[int, List[Dict]]:
        """Score section presence (0-100)."""
        found_sections = []
        missing_sections = []
        
        # Check for section headers or content
        section_patterns = {
            'summary': ['summary', 'about me', 'professional summary', 'overview'],
            'education': ['education', 'degree', 'university', 'college', 'cgpa'],
            'skills': ['skills', 'technical skills', 'tools', 'technologies'],
            'experience': ['experience', 'work experience', 'employment', 'internship'],
            'projects': ['projects', 'project', 'portfolio'],
        }
        
        for section_name, patterns in section_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                found_sections.append(section_name)
            else:
                missing_sections.append(section_name)
        
        # Also check structured data if available
        if structured_data:
            if structured_data.get('education'):
                if 'education' in missing_sections:
                    missing_sections.remove('education')
                    found_sections.append('education')
            if structured_data.get('projects'):
                if 'projects' in missing_sections:
                    missing_sections.remove('projects')
                    found_sections.append('projects')
        
        score = int((len(found_sections) / len(section_patterns)) * 100)
        faults = []
        
        if missing_sections:
            faults.append({
                'type': 'missing_sections',
                'severity': 'high',
                'message': f"Missing sections: {', '.join(missing_sections)}",
                'suggestion': f"Add a clear '{missing_sections[0].title()}' section with a heading like '## {missing_sections[0].title()}'",
                'example': f"## {missing_sections[0].title()}\n[Your content here]"
            })
        
        return score, faults
    
    def _score_format(self, text_lower: str) -> Tuple[int, List[Dict]]:
        """Score ATS-friendly formatting (0-100)."""
        score = 100
        faults = []
        
        # Check for tables (common ATS issue)
        if '|' in text_lower and text_lower.count('|') > 5:
            score -= 30
            faults.append({
                'type': 'tables',
                'severity': 'medium',
                'message': "Uses tables which may reduce ATS parse accuracy",
                'suggestion': "Convert tables to plain text with bullets. Use standard headings and bullet points instead.",
                'example': "Instead of a skills table, use: 'Skills: Python, React, SQL, Docker'"
            })
        
        # Check for images references
        if any(word in text_lower for word in ['image:', 'img:', '.jpg', '.png']):
            score -= 20
            faults.append({
                'type': 'images',
                'severity': 'low',
                'message': "References to images found",
                'suggestion': "Remove image references. ATS systems cannot parse images.",
            })
        
        # Check for fonts that might not parse well
        if any(font in text_lower for font in ['comic sans', 'wingdings', 'symbol']):
            score -= 15
            faults.append({
                'type': 'font',
                'severity': 'low',
                'message': "Non-standard fonts detected",
                'suggestion': "Use standard fonts: Arial, Calibri, Times New Roman",
            })
        
        return max(0, score), faults
    
    def _score_impact(self, text_lower: str) -> Tuple[int, List[Dict]]:
        """Score quantifiable impact statements (0-100)."""
        # Look for numbers, percentages, action verbs with results
        number_pattern = r'\b\d+(\.\d+)?%?\b'
        action_verbs = ['increased', 'decreased', 'improved', 'reduced', 'achieved', 'led', 'developed', 'implemented']
        
        numbers = re.findall(number_pattern, text_lower)
        action_with_results = sum(1 for verb in action_verbs if verb in text_lower)
        
        # Score based on presence of numbers and action verbs
        score = 0
        if numbers:
            score += 50  # Has quantifiable metrics
        if action_with_results >= 3:
            score += 50  # Has multiple action verbs
        elif action_with_results >= 1:
            score += 25
        
        faults = []
        if score < 50:
            faults.append({
                'type': 'no_quantifiable_impact',
                'severity': 'medium',
                'message': "Missing quantifiable achievements and impact metrics",
                'suggestion': "Add numbers, percentages, or measurable results to your bullet points. Use action verbs like 'Increased', 'Improved', 'Led'.",
                'example': "Instead of 'Worked on ML project', write 'Developed ML model that improved accuracy by 15% and reduced processing time by 30%'"
            })
        
        return min(100, score), faults
    
    def _score_length(self, resume_text: str) -> Tuple[int, List[Dict]]:
        """Score appropriate length (0-100)."""
        word_count = len(resume_text.split())
        char_count = len(resume_text)
        
        # Ideal: 400-600 words for entry level, 600-800 for mid-level
        ideal_min = 400
        ideal_max = 800
        
        score = 100
        faults = []
        
        if word_count < ideal_min:
            score = int((word_count / ideal_min) * 100)
            faults.append({
                'type': 'too_short',
                'severity': 'low',
                'message': f"Resume is too short ({word_count} words). Target: {ideal_min}-{ideal_max} words.",
                'suggestion': "Add more detail to your projects, achievements, and experience sections.",
            })
        elif word_count > ideal_max * 1.5:
            score = 70
            faults.append({
                'type': 'too_long',
                'severity': 'low',
                'message': f"Resume is quite long ({word_count} words). Consider condensing.",
                'suggestion': "Focus on most relevant experiences. Remove redundant information.",
            })
        
        return score, faults

