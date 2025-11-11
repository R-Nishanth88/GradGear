"""
Trending Skills & Certifications Detection
"""
from typing import Dict, List


class TrendingSkillsDetector:
    """Detect trending skills and certifications from job market data."""
    
    # Domain-specific trending skills (in production, fetch from Spark/job DB)
    TRENDING_SKILLS = {
        'AI/ML': {
            'high_demand': ['LLM', 'Large Language Models', 'Generative AI', 'MLOps', 'Vector Databases', 'LangChain', 'Hugging Face', 'Fine-tuning', 'RAG'],
            'emerging': ['Agentic AI', 'Multimodal Models', 'AI Safety', 'Prompt Engineering'],
        },
        'Cybersecurity': {
            'high_demand': ['Incident Response', 'Threat Analysis', 'SIEM', 'Penetration Testing', 'Cloud Security', 'Zero Trust', 'Identity Management'],
            'emerging': ['AI Security', 'Supply Chain Security', 'IoT Security'],
        },
        'Data Science': {
            'high_demand': ['Data Engineering', 'MLOps', 'Feature Engineering', 'A/B Testing', 'Time Series', 'Big Data', 'Real-time Analytics'],
            'emerging': ['Data Mesh', 'Data Products', 'Lakehouse Architecture'],
        },
        'Web Development': {
            'high_demand': ['Next.js', 'TypeScript', 'GraphQL', 'Microservices', 'Serverless', 'React', 'Node.js'],
            'emerging': ['Edge Computing', 'WebAssembly', 'Progressive Web Apps'],
        },
        'Cloud Computing': {
            'high_demand': ['Kubernetes', 'Terraform', 'Serverless', 'Multi-cloud', 'DevOps', 'CI/CD', 'Infrastructure as Code'],
            'emerging': ['FinOps', 'Cloud Security Posture', 'Serverless Patterns'],
        },
        'IoT': {
            'high_demand': ['Edge Computing', 'MQTT', 'IoT Security', 'Embedded Systems', 'Real-time Processing'],
            'emerging': ['Digital Twins', 'Industrial IoT'],
        },
        'Robotics': {
            'high_demand': ['ROS', 'Computer Vision', 'Motion Planning', 'Sensor Fusion', 'Robotic Control'],
            'emerging': ['Human-Robot Interaction', 'Autonomous Systems'],
        },
    }
    
    # Domain-specific certifications
    CERTIFICATIONS = {
        'AI/ML': [
            {'name': 'TensorFlow Developer Certificate', 'provider': 'Google', 'level': 'Intermediate'},
            {'name': 'AWS Machine Learning Specialty', 'provider': 'AWS', 'level': 'Advanced'},
            {'name': 'Azure AI Engineer Associate', 'provider': 'Microsoft', 'level': 'Intermediate'},
        ],
        'Cybersecurity': [
            {'name': 'Certified Ethical Hacker (CEH)', 'provider': 'EC-Council', 'level': 'Intermediate'},
            {'name': 'CompTIA Security+', 'provider': 'CompTIA', 'level': 'Beginner'},
            {'name': 'CISSP', 'provider': 'ISC²', 'level': 'Advanced'},
            {'name': 'AWS Security Specialty', 'provider': 'AWS', 'level': 'Advanced'},
        ],
        'Data Science': [
            {'name': 'AWS Certified Data Analytics', 'provider': 'AWS', 'level': 'Intermediate'},
            {'name': 'Google Data Analytics Certificate', 'provider': 'Google', 'level': 'Beginner'},
            {'name': 'Tableau Desktop Specialist', 'provider': 'Tableau', 'level': 'Intermediate'},
        ],
        'Web Development': [
            {'name': 'AWS Certified Developer Associate', 'provider': 'AWS', 'level': 'Intermediate'},
            {'name': 'Google Cloud Professional Cloud Developer', 'provider': 'Google', 'level': 'Intermediate'},
            {'name': 'Meta Front-End Developer Certificate', 'provider': 'Meta', 'level': 'Beginner'},
        ],
        'Cloud Computing': [
            {'name': 'AWS Solutions Architect Associate', 'provider': 'AWS', 'level': 'Intermediate'},
            {'name': 'Kubernetes Administrator (CKA)', 'provider': 'CNCF', 'level': 'Intermediate'},
            {'name': 'Terraform Associate', 'provider': 'HashiCorp', 'level': 'Intermediate'},
        ],
    }
    
    def get_trending_skills(self, domain: str, current_skills: List[str]) -> Dict[str, List[str]]:
        """
        Get trending skills not present in current resume.
        
        Returns:
        {
            'high_demand': List[str],
            'emerging': List[str],
            'missing': List[str]  # High demand skills user is missing
        }
        """
        domain_data = self.TRENDING_SKILLS.get(domain, self.TRENDING_SKILLS.get('AI/ML', {}))
        
        current_lower = [s.lower() for s in current_skills]
        
        high_demand = domain_data.get('high_demand', [])
        emerging = domain_data.get('emerging', [])
        
        # Find missing high-demand skills
        missing = [skill for skill in high_demand if not any(skill.lower() in curr or curr in skill.lower() for curr in current_lower)]
        
        return {
            'high_demand': high_demand[:10],
            'emerging': emerging[:5],
            'missing': missing[:5],  # Top 5 missing skills
        }
    
    def get_recommended_certifications(self, domain: str, experience_level: str = 'Intermediate') -> List[Dict]:
        """Get recommended certifications for domain and experience level."""
        certs = self.CERTIFICATIONS.get(domain, self.CERTIFICATIONS.get('AI/ML', []))
        
        # Filter by experience level if needed
        filtered = [c for c in certs if c['level'] == experience_level or experience_level == 'All']
        
        return filtered[:5]  # Top 5 certifications

