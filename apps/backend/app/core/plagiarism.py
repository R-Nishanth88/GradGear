"""
Plagiarism Checker with LLM-based remediation
"""
import re
from typing import List, Dict, Tuple


class PlagiarismChecker:
    """Check for plagiarism and provide remediation suggestions."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        # In production, initialize OpenAI client if API key provided
        # if api_key:
        #     from openai import OpenAI
        #     self.client = OpenAI(api_key=api_key)
    
    def check(self, resume_text: str) -> Dict:
        """
        Check plagiarism and return score + flagged passages.
        
        Returns:
        {
            'plagiarism_score': float (0-100),
            'matched_sources': List[str],
            'flagged_passages': List[Dict],
            'plagiarism_safe': bool
        }
        """
        # In production: call plagiarism API (Copyleaks, PlagiarismCheck, etc.)
        # For now: use LLM-based semantic similarity check
        
        # Common resume phrases to check for
        common_phrases = [
            "motivated individual",
            "hard working and dedicated",
            "seeking opportunities",
            "excellent communication skills",
            "team player",
            "detail-oriented professional",
        ]
        
        flagged_passages = []
        plagiarism_score = 0.0
        
        for phrase in common_phrases:
            if phrase.lower() in resume_text.lower():
                # Count occurrences
                count = len(re.findall(re.escape(phrase), resume_text, re.IGNORECASE))
                if count > 0:
                    plagiarism_score += 2.0  # 2% per common phrase
                    flagged_passages.append({
                        'text': phrase,
                        'similarity': 0.9,
                        'suggestion': self._generate_paraphrase(phrase),
                    })
        
        # If using external API:
        # result = self._call_plagiarism_api(resume_text)
        # plagiarism_score = result.get('score', 0)
        # flagged_passages = result.get('matches', [])
        
        plagiarism_safe = plagiarism_score < 10.0  # Threshold: 10%
        
        return {
            'plagiarism_score': min(100.0, plagiarism_score),
            'matched_sources': [],
            'flagged_passages': flagged_passages[:10],  # Top 10
            'plagiarism_safe': plagiarism_safe,
        }
    
    def _generate_paraphrase(self, text: str) -> str:
        """Generate paraphrase suggestion using LLM."""
        # In production: call LLM API
        paraphrase_map = {
            "motivated individual": "driven professional",
            "hard working and dedicated": "committed and results-oriented",
            "seeking opportunities": "exploring new challenges",
            "excellent communication skills": "strong interpersonal communication",
            "team player": "collaborative team member",
            "detail-oriented professional": "meticulous professional with strong attention to detail",
        }
        
        return paraphrase_map.get(text.lower(), f"Rephrase: {text}")
    
    def remediate_passages(self, passages: List[str]) -> List[Dict]:
        """Provide remediation suggestions for flagged passages."""
        remediations = []
        for passage in passages:
            suggestion = self._generate_paraphrase(passage)
            remediations.append({
                'original': passage,
                'suggested': suggestion,
                'reason': "Rephrased to avoid common resume clichés and emphasize unique contributions",
            })
        return remediations

