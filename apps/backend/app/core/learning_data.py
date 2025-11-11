from __future__ import annotations

from typing import Any, Dict, List, Optional

DomainKey = str
SkillTag = str


def _normalize_domain(domain: str) -> DomainKey:
    return domain.lower().replace(" ", "_").replace("/", "_")


LEARNING_PATHS: Dict[DomainKey, Dict[str, Any]] = {
    "ai_ml": {
        "headline": "Master foundational ML to ship production-ready models.",
        "steps": [
            {
                "skill": "Python for Data Science",
                "tag": "python_ds",
                "duration": "1 week",
                "mini_task": "Implement data wrangling utilities using pandas.",
                "video": {
                    "title": "NumPy & Pandas Crash Course",
                    "url": "https://www.youtube.com/watch?v=tRKeO-6UqD0",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Applied Data Science with Python",
                    "url": "https://www.coursera.org/specializations/data-science-python",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "Machine Learning Foundations",
                "tag": "ml_foundations",
                "duration": "2 weeks",
                "mini_task": "Train and evaluate a classification model on Kaggle Titanic dataset.",
                "video": {
                    "title": "Hands-on Machine Learning",
                    "url": "https://www.youtube.com/watch?v=qeHZOdmJvFU",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Machine Learning by Andrew Ng",
                    "url": "https://www.coursera.org/learn/machine-learning",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "Neural Networks",
                "tag": "neural_networks",
                "duration": "2 weeks",
                "mini_task": "Build an image classifier using PyTorch and report accuracy improvements.",
                "video": {
                    "title": "Neural Network Visual Explanation",
                    "url": "https://www.youtube.com/watch?v=aircAruvnKk",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Deep Learning Specialization",
                    "url": "https://www.coursera.org/specializations/deep-learning",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "MLOps & Deployment",
                "tag": "mlops",
                "duration": "1 week",
                "mini_task": "Package a model with FastAPI and deploy to Render or Railway.",
                "video": {
                    "title": "End-to-End MLOps Tutorial",
                    "url": "https://www.youtube.com/watch?v=0hKc8MyF8r8",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "MLOps Fundamentals",
                    "url": "https://www.udemy.com/course/mlops-fundamentals/",
                    "platform": "Udemy",
                },
            },
        ],
        "projects": [
            {
                "title": "Student Placement Predictor",
                "description": "Predict placements using scikit-learn, sharing model insights in a dashboard.",
                "url": "https://github.com/krishnaik06/StudentPlacement",
                "difficulty": "Intermediate",
                "duration": "2 weeks",
            },
            {
                "title": "Resume Keyword Classifier",
                "description": "Use NLP to classify resumes by job role and surface ATS keyword gaps.",
                "url": "https://github.com/abhishekkrthakur/approachingalmost",
                "difficulty": "Advanced",
                "duration": "3 weeks",
            },
        ],
    },
    "cybersecurity": {
        "headline": "Build blue-team fundamentals and pass Security+.",
        "steps": [
            {
                "skill": "Networking Basics",
                "tag": "networking",
                "duration": "1 week",
                "mini_task": "Diagram the OSI model with real tools mapped to each layer.",
                "video": {
                    "title": "Networking Fundamentals",
                    "url": "https://www.youtube.com/watch?v=qiQR5rTSshw",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Introduction to Networking",
                    "url": "https://www.coursera.org/learn/comptia-network-plus",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "Threat Identification",
                "tag": "threats",
                "duration": "1 week",
                "mini_task": "Perform a mini threat model for your campus network.",
                "video": {
                    "title": "Threat Modeling 101",
                    "url": "https://www.youtube.com/watch?v=li_ejMIxJdw",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "IBM Cybersecurity Analyst",
                    "url": "https://www.coursera.org/professional-certificates/ibm-cybersecurity-analyst",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "Security Operations",
                "tag": "siem",
                "duration": "2 weeks",
                "mini_task": "Set up a SIEM lab using Elastic Security and detect anomalies.",
                "video": {
                    "title": "SIEM Tutorial",
                    "url": "https://www.youtube.com/watch?v=ZE7qJUKFZ5s",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Security Operations & Automation",
                    "url": "https://www.udemy.com/course/siem-security-operations/",
                    "platform": "Udemy",
                },
            },
        ],
        "projects": [
            {
                "title": "Security Monitoring Lab",
                "description": "Deploy Elastic or Splunk to monitor simulated attacks.",
                "url": "https://github.com/C3ntryLabs/Blue-Team-Training",
                "difficulty": "Intermediate",
                "duration": "3 weeks",
            },
            {
                "title": "Incident Response Playbook",
                "description": "Document a detailed IR plan and automate triage with Python.",
                "url": "https://github.com/soxoj/maigret",
                "difficulty": "Advanced",
                "duration": "2 weeks",
            },
        ],
    },
    "data_science": {
        "headline": "Master analytics foundations and ship data products with measurable impact.",
        "steps": [
            {
                "skill": "Python for Analytics",
                "tag": "python_ds",
                "duration": "1 week",
                "mini_task": "Clean and analyze a Kaggle dataset using pandas and visualize findings.",
                "video": {
                    "title": "Python Data Analysis Crash Course",
                    "url": "https://www.youtube.com/watch?v=vmEHCJofslg",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "IBM Python for Data Science",
                    "url": "https://www.coursera.org/learn/python-for-data-science-ai",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "Exploratory Data Analysis",
                "tag": "eda",
                "duration": "1 week",
                "mini_task": "Deliver an EDA report highlighting anomalies and actionable metrics.",
                "video": {
                    "title": "EDA with pandas & seaborn",
                    "url": "https://www.youtube.com/watch?v=GcXcSZ0gQps",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Applied Plotting, Charting & Data Representation",
                    "url": "https://www.coursera.org/learn/python-plot-data",
                    "platform": "Coursera",
                },
            },
            {
                "skill": "Machine Learning Workflow",
                "tag": "ml_workflow",
                "duration": "2 weeks",
                "mini_task": "Train, tune, and evaluate a model with cross-validation and report metrics.",
                "video": {
                    "title": "Machine Learning in 100 Minutes",
                    "url": "https://www.youtube.com/watch?v=7eh4d6sabA0",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "Machine Learning with Scikit-Learn",
                    "url": "https://www.udemy.com/course/machine-learning-with-scikit-learn/",
                    "platform": "Udemy",
                },
            },
            {
                "skill": "Model Deployment",
                "tag": "model_deployment",
                "duration": "1 week",
                "mini_task": "Deploy a FastAPI endpoint serving a trained model and log predictions.",
                "video": {
                    "title": "Deploy ML Models with FastAPI",
                    "url": "https://www.youtube.com/watch?v=0sOvCWFmrtA",
                    "platform": "YouTube",
                },
                "course": {
                    "title": "MLOps with AWS",
                    "url": "https://www.coursera.org/learn/mlops-aws",
                    "platform": "Coursera",
                },
            },
        ],
        "projects": [
            {
                "title": "Customer Churn Insight Dashboard",
                "description": "Analyze churn drivers, build predictive models, and ship a Streamlit dashboard.",
                "url": "https://github.com/tirthajyoti/Machine-Learning-with-Python",
                "difficulty": "Intermediate",
                "duration": "2-3 weeks",
            },
            {
                "title": "AutoML Experiment Tracker",
                "description": "Automate feature engineering and track experiments with MLflow for reproducibility.",
                "url": "https://github.com/mlflow/mlflow",
                "difficulty": "Advanced",
                "duration": "3 weeks",
            },
        ],
    },
}


RESOURCE_CATALOG: List[Dict[str, Any]] = [
    {
        "domain": "ai_ml",
        "skillTag": "neural_networks",
        "videoUrl": "https://www.youtube.com/watch?v=aircAruvnKk",
        "videoTitle": "Neural Networks - 3Blue1Brown",
        "courseUrl": "https://www.coursera.org/learn/neural-networks-deep-learning",
        "courseTitle": "Neural Networks and Deep Learning",
        "difficulty": "Beginner",
        "duration": "2 hours video / 3 weeks course",
    },
    {
        "domain": "ai_ml",
        "skillTag": "mlops",
        "videoUrl": "https://www.youtube.com/watch?v=0hKc8MyF8r8",
        "videoTitle": "MLOps from Scratch",
        "courseUrl": "https://www.udemy.com/course/mlops-fundamentals/",
        "courseTitle": "MLOps Fundamentals",
        "difficulty": "Intermediate",
        "duration": "3 hours video / 5 weeks course",
    },
    {
        "domain": "cybersecurity",
        "skillTag": "siem",
        "videoUrl": "https://www.youtube.com/watch?v=ZE7qJUKFZ5s",
        "videoTitle": "SIEM Tutorial for Beginners",
        "courseUrl": "https://www.udemy.com/course/siem-security-operations/",
        "courseTitle": "Security Operations & Automation",
        "difficulty": "Intermediate",
        "duration": "90 mins video / 4 weeks course",
    },
    {
        "domain": "data_science",
        "skillTag": "python_ds",
        "videoUrl": "https://www.youtube.com/watch?v=vmEHCJofslg",
        "videoTitle": "Python Data Analysis Crash Course",
        "courseUrl": "https://www.coursera.org/learn/python-for-data-science-ai",
        "courseTitle": "Python for Data Science, AI & Development",
        "difficulty": "Beginner",
        "duration": "2 hours video / 3 weeks course",
    },
    {
        "domain": "data_science",
        "skillTag": "eda",
        "videoUrl": "https://www.youtube.com/watch?v=GcXcSZ0gQps",
        "videoTitle": "Exploratory Data Analysis with pandas",
        "courseUrl": "https://www.coursera.org/learn/python-plot-data",
        "courseTitle": "Applied Plotting, Charting & Data Representation",
        "difficulty": "Beginner",
        "duration": "90 mins video / 2 weeks course",
    },
    {
        "domain": "data_science",
        "skillTag": "ml_workflow",
        "videoUrl": "https://www.youtube.com/watch?v=7eh4d6sabA0",
        "videoTitle": "Machine Learning in 100 Minutes",
        "courseUrl": "https://www.udemy.com/course/machine-learning-with-scikit-learn/",
        "courseTitle": "Machine Learning with Scikit-Learn",
        "difficulty": "Intermediate",
        "duration": "100 mins video / 4 weeks course",
    },
    {
        "domain": "data_science",
        "skillTag": "model_deployment",
        "videoUrl": "https://www.youtube.com/watch?v=0sOvCWFmrtA",
        "videoTitle": "Deploy ML Models with FastAPI",
        "courseUrl": "https://www.coursera.org/learn/mlops-aws",
        "courseTitle": "MLOps on AWS",
        "difficulty": "Advanced",
        "duration": "75 mins video / 3 weeks course",
    },
]


CERTIFICATION_ROADMAP: Dict[DomainKey, List[Dict[str, Any]]] = {
    "cybersecurity": [
        {
            "name": "CompTIA Security+",
            "examCode": "SY0-701",
            "officialUrl": "https://www.comptia.org/certifications/securityplus",
            "prepCourseUrl": "https://www.udemy.com/course/securityplus/",
            "timeToPrepare": "6-10 weeks",
            "weightage": [
                {"topic": "General Security Concepts", "weight": "12%"},
                {"topic": "Threats, Vulnerabilities & Mitigations", "weight": "22%"},
                {"topic": "Security Operations", "weight": "28%"},
                {"topic": "Security Architecture", "weight": "18%"},
                {"topic": "Security Program Management", "weight": "20%"},
            ],
        },
        {
            "name": "Certified Ethical Hacker",
            "examCode": "CEH v12",
            "officialUrl": "https://www.eccouncil.org/programs/certified-ethical-hacker-ceh/",
            "prepCourseUrl": "https://www.udemy.com/course/certified-ethical-hacker-course/",
            "timeToPrepare": "8-12 weeks",
            "weightage": [
                {"topic": "Footprinting & Reconnaissance", "weight": "20%"},
                {"topic": "Scanning Networks", "weight": "15%"},
                {"topic": "Vulnerability Analysis", "weight": "12%"},
                {"topic": "System Hacking", "weight": "20%"},
                {"topic": "Malware Threats & Sniffing", "weight": "10%"},
                {"topic": "Social Engineering", "weight": "8%"},
                {"topic": "Cloud Security & IoT", "weight": "15%"},
            ],
        },
    ],
    "ai_ml": [
        {
            "name": "AWS Machine Learning Specialty",
            "examCode": "MLS-C01",
            "officialUrl": "https://aws.amazon.com/certification/certified-machine-learning-specialty/",
            "prepCourseUrl": "https://www.udemy.com/course/aws-machine-learning/",
            "timeToPrepare": "8-12 weeks",
            "weightage": [
                {"topic": "Data Engineering", "weight": "20%"},
                {"topic": "Exploratory Data Analysis", "weight": "24%"},
                {"topic": "Modeling", "weight": "36%"},
                {"topic": "Machine Learning Implementation & Operations", "weight": "20%"},
            ],
        },
        {
            "name": "TensorFlow Developer Certificate",
            "examCode": "TF-DEV",
            "officialUrl": "https://www.tensorflow.org/certificate",
            "prepCourseUrl": "https://www.coursera.org/professional-certificates/tensorflow-in-practice",
            "timeToPrepare": "4-6 weeks",
            "weightage": [
                {"topic": "TensorFlow for Deep Learning", "weight": "30%"},
                {"topic": "Computer Vision", "weight": "30%"},
                {"topic": "Natural Language Processing", "weight": "20%"},
                {"topic": "Time Series and Other Tasks", "weight": "20%"},
            ],
        },
    ],
    "data_science": [
        {
            "name": "AWS Certified Data Analytics – Specialty",
            "examCode": "DAS-C01",
            "officialUrl": "https://aws.amazon.com/certification/certified-data-analytics-specialty/",
            "prepCourseUrl": "https://www.udemy.com/course/aws-certified-data-analytics-specialty/",
            "timeToPrepare": "6–10 weeks",
            "weightage": [
                {"topic": "Collection", "weight": "18%"},
                {"topic": "Storage & Data Management", "weight": "22%"},
                {"topic": "Processing", "weight": "24%"},
                {"topic": "Analysis & Visualization", "weight": "18%"},
                {"topic": "Security", "weight": "18%"},
            ],
            "provider": "AWS",
            "level": "Intermediate",
        },
        {
            "name": "TensorFlow Developer Certificate",
            "examCode": "TF-DEV",
            "officialUrl": "https://www.tensorflow.org/certificate",
            "prepCourseUrl": "https://www.coursera.org/professional-certificates/tensorflow-in-practice",
            "timeToPrepare": "4–6 weeks",
            "weightage": [
                {"topic": "TensorFlow Fundamentals", "weight": "20%"},
                {"topic": "Computer Vision", "weight": "22%"},
                {"topic": "NLP with TensorFlow", "weight": "24%"},
                {"topic": "Time Series & Sequences", "weight": "18%"},
                {"topic": "Deployment", "weight": "16%"},
            ],
            "provider": "Google",
            "level": "Intermediate",
        },
        {
            "name": "Databricks Certified Data Engineer Associate",
            "examCode": "DB-DEA",
            "officialUrl": "https://www.databricks.com/learn/certification/data-engineer-associate",
            "prepCourseUrl": "https://www.databricks.com/learn/training/data-engineer",
            "timeToPrepare": "4–8 weeks",
            "weightage": [
                {"topic": "Data Engineering with Databricks", "weight": "20%"},
                {"topic": "Delta Lake", "weight": "25%"},
                {"topic": "ETL Pipelines", "weight": "20%"},
                {"topic": "Production & Automation", "weight": "15%"},
                {"topic": "Security & Governance", "weight": "20%"},
            ],
            "provider": "Databricks",
            "level": "Intermediate",
        },
    ],
}


def get_learning_path(domain: str) -> Optional[Dict[str, Any]]:
    return LEARNING_PATHS.get(_normalize_domain(domain))


def get_resources(domain: str, skill: Optional[str] = None) -> List[Dict[str, Any]]:
    domain_key = _normalize_domain(domain)
    items = [res for res in RESOURCE_CATALOG if res["domain"] == domain_key]
    if skill:
        skill_key = skill.lower().replace(" ", "_")
        items = [res for res in items if res["skillTag"] == skill_key or res["skillTag"] == skill]
    return items


def get_certifications(domain: str) -> List[Dict[str, Any]]:
    return CERTIFICATION_ROADMAP.get(_normalize_domain(domain), [])


def get_project_suggestions(domain: str) -> List[Dict[str, Any]]:
    learning_path = get_learning_path(domain) or {}
    return learning_path.get("projects", [])

