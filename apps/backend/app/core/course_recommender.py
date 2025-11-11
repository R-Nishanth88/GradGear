"""
Course and Resource Recommender (YouTube, Coursera, Udemy)
Enhanced with real API integration
"""
from typing import List, Dict
import os
import httpx


class CourseRecommender:
    """Recommend courses and resources for missing skills."""
    
    def __init__(self):
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY', '')
        self.coursera_api_key = os.getenv('COURSERA_API_KEY', '')
    
    async def recommend(self, skill: str, domain: str) -> List[Dict]:
        """
        Recommend courses/videos/resources for a skill.
        
        Returns:
        [
            {
                'title': str,
                'source': str,  # 'YouTube', 'Coursera', 'Udemy'
                'url': str,
                'estimated_time': str,
                'level': str,  # 'Beginner', 'Intermediate', 'Advanced'
                'reason': str  # Why this course fits
            }
        ]
        """
        recommendations = []
        
        # YouTube recommendations
        youtube_courses = await self._get_youtube_courses(skill, domain)
        recommendations.extend(youtube_courses[:3])
        
        # Coursera recommendations
        coursera_courses = await self._get_coursera_courses(skill, domain)
        recommendations.extend(coursera_courses[:2])
        
        # Curated fallback (if APIs unavailable)
        if not recommendations:
            recommendations = self._get_fallback_courses(skill, domain)
        
        return recommendations
    
    async def _get_youtube_courses(self, skill: str, domain: str) -> List[Dict]:
        """Get YouTube course recommendations via API."""
        if not self.youtube_api_key:
            return self._get_fallback_youtube(skill, domain)
        
        try:
            async with httpx.AsyncClient() as client:
                # Search for tutorials
                search_query = f"{skill} {domain} tutorial"
                response = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "key": self.youtube_api_key,
                        "q": search_query,
                        "part": "snippet",
                        "type": "video",
                        "maxResults": 3,
                        "order": "relevance",
                        "videoDuration": "medium",  # 4-20 minutes
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    courses = []
                    for item in data.get("items", []):
                        video_id = item["id"]["videoId"]
                        snippet = item["snippet"]
                        courses.append({
                            'title': snippet['title'],
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'source': 'YouTube',
                            'estimated_time': '10-20 minutes',
                            'level': 'Beginner',
                            'reason': f"Quick {skill} tutorial for {domain} professionals",
                        })
                    return courses
        except Exception as e:
            print(f"YouTube API error: {e}")
        
        return self._get_fallback_youtube(skill, domain)
    
    async def _get_coursera_courses(self, skill: str, domain: str) -> List[Dict]:
        """Get Coursera course recommendations."""
        # Coursera API requires partner access, use curated fallback
        return [
            {
                'title': f'{skill} Specialization',
                'url': f'https://www.coursera.org/search?query={skill.replace(" ", "+")}',
                'source': 'Coursera',
                'estimated_time': '4-6 weeks',
                'level': 'Intermediate',
                'reason': f'Professional {skill} specialization with industry certification',
            }
        ]
    
    def _get_fallback_youtube(self, skill: str, domain: str) -> List[Dict]:
        """Fallback YouTube recommendations."""
        youtube_map = {
            'tensorflow': [
                {'title': 'TensorFlow 2.0 Complete Course', 'url': 'https://www.youtube.com/results?search_query=tensorflow+complete+course', 'reason': 'Comprehensive TensorFlow tutorial'},
            ],
            'python': [
                {'title': 'Python Full Course for Beginners', 'url': 'https://www.youtube.com/results?search_query=python+full+course', 'reason': 'Complete Python programming guide'},
            ],
            'react': [
                {'title': 'React JS Full Course', 'url': 'https://www.youtube.com/results?search_query=react+full+course', 'reason': 'Modern React development'},
            ],
        }
        
        skill_lower = skill.lower()
        for key, courses in youtube_map.items():
            if key in skill_lower:
                return [{**c, 'source': 'YouTube', 'estimated_time': '8-12 hours', 'level': 'Intermediate'} for c in courses]
        
        return [{
            'title': f'{skill} Tutorial - {domain}',
            'url': f'https://www.youtube.com/results?search_query={skill.replace(" ", "+")}+{domain.replace("/", "")}+tutorial',
            'source': 'YouTube',
            'estimated_time': '5-8 hours',
            'level': 'Beginner',
            'reason': f'Comprehensive {skill} tutorial with {domain} applications',
        }]
    
    def _get_fallback_courses(self, skill: str, domain: str) -> List[Dict]:
        """Fallback curated course recommendations."""
        return [
            {
                'title': f'{skill} - Complete Guide',
                'url': f'https://www.udemy.com/search/?q={skill.replace(" ", "+")}',
                'source': 'Udemy',
                'estimated_time': '10-15 hours',
                'level': 'Beginner',
                'reason': f'Beginner-friendly {skill} course aligned with {domain} domain',
            },
            {
                'title': f'{skill} Tutorial - {domain} Focus',
                'url': f'https://www.youtube.com/results?search_query={skill.replace(" ", "+")}+{domain.replace("/", "")}',
                'source': 'YouTube',
                'estimated_time': '5-8 hours',
                'level': 'Intermediate',
                'reason': f'Practical {skill} tutorial with {domain} applications',
            },
        ]
