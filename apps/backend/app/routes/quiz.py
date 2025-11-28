from fastapi import APIRouter, Depends, HTTPException
from random import sample, shuffle
from app.routes.auth_ext import get_current_user
from app.models import User
from typing import List, Dict, Any

router = APIRouter()

_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "AI/ML": [
        {"q": "What is overfitting in machine learning?", "options": ["High bias", "High variance", "Low variance", "None"], "a": 1},
        {"q": "Which activation function is best for binary classification?", "options": ["ReLU", "Sigmoid", "Softmax", "Tanh"], "a": 1},
        {"q": "What does CNN stand for?", "options": ["Convolutional Neural Network", "Cascaded Neural Network", "Complex Neural Network", "Cyclic Neural Network"], "a": 0},
        {"q": "What is the purpose of dropout in neural networks?", "options": ["Increase model complexity", "Reduce overfitting", "Speed up training", "Reduce model parameters"], "a": 1},
        {"q": "Which algorithm is used for classification problems?", "options": ["Linear Regression", "K-Means", "Logistic Regression", "DBSCAN"], "a": 2},
        {"q": "What is a learning rate in gradient descent?", "options": ["The number of epochs", "The step size for parameter updates", "The batch size", "The number of layers"], "a": 1},
        {"q": "What is cross-validation used for?", "options": ["Speeding up training", "Evaluating model performance", "Reducing features", "Increasing accuracy"], "a": 1},
        {"q": "Which type of learning uses labeled data?", "options": ["Unsupervised Learning", "Supervised Learning", "Reinforcement Learning", "Semi-supervised Learning"], "a": 1},
        {"q": "What is the vanishing gradient problem?", "options": ["Gradients become too large", "Gradients become too small", "Gradients become zero", "Gradients become negative"], "a": 1},
        {"q": "What does RNN stand for?", "options": ["Random Neural Network", "Recurrent Neural Network", "Robust Neural Network", "Regularized Neural Network"], "a": 1},
        {"q": "What is the purpose of batch normalization?", "options": ["Reduce batch size", "Stabilize training", "Increase model size", "Reduce parameters"], "a": 1},
        {"q": "Which metric is best for imbalanced datasets?", "options": ["Accuracy", "F1-Score", "MAE", "R-squared"], "a": 1},
    ],
    "Cybersecurity": [
        {"q": "What is SQL injection?", "options": ["XSS attack", "Database vulnerability exploit", "CSRF attack", "MITM attack"], "a": 1},
        {"q": "TLS operates at which OSI layer?", "options": ["Application", "Transport", "Network", "Data Link"], "a": 1},
        {"q": "What is a DDoS attack?", "options": ["Data breach", "Distributed Denial of Service", "Database deletion", "Device disruption"], "a": 1},
        {"q": "What is the purpose of a firewall?", "options": ["Speed up network", "Block unauthorized access", "Encrypt data", "Store backups"], "a": 1},
        {"q": "What does VPN stand for?", "options": ["Virtual Private Network", "Very Private Network", "Verified Private Network", "Volatile Private Network"], "a": 0},
        {"q": "What is two-factor authentication?", "options": ["Two passwords", "Two-step verification", "Two accounts", "Two devices"], "a": 1},
        {"q": "What is a zero-day vulnerability?", "options": ["Known vulnerability", "Unknown vulnerability with no patch", "Old vulnerability", "Fixed vulnerability"], "a": 1},
        {"q": "What is encryption?", "options": ["Data compression", "Data encoding for security", "Data deletion", "Data backup"], "a": 1},
        {"q": "What is phishing?", "options": ["Network attack", "Social engineering attack", "Database attack", "System attack"], "a": 1},
        {"q": "What is the purpose of penetration testing?", "options": ["Attack systems", "Find vulnerabilities", "Steal data", "Disrupt services"], "a": 1},
    ],
    "Data Science": [
        {"q": "What is pandas used for in Python?", "options": ["Web development", "Data manipulation", "Machine learning", "Game development"], "a": 1},
        {"q": "What is the difference between correlation and causation?", "options": ["Same thing", "Correlation doesn't imply causation", "Causation implies correlation", "No relationship"], "a": 1},
        {"q": "What is feature engineering?", "options": ["Removing features", "Creating/modifying features", "Deleting data", "Adding noise"], "a": 1},
        {"q": "What is a confusion matrix used for?", "options": ["Visualize classification performance", "Plot data", "Store results", "Calculate accuracy"], "a": 0},
        {"q": "What is ETL?", "options": ["Extract, Transform, Load", "Enter, Transfer, Leave", "Export, Test, Launch", "Encode, Transfer, Load"], "a": 0},
        {"q": "What is overfitting?", "options": ["Model too simple", "Model memorizes training data", "Model too slow", "Model too fast"], "a": 1},
        {"q": "What is the purpose of cross-validation?", "options": ["Train faster", "Evaluate model generalization", "Reduce data", "Increase accuracy"], "a": 1},
        {"q": "What is a p-value?", "options": ["Probability of data", "Probability of hypothesis", "Probability of error", "Probability of success"], "a": 0},
    ],
    "Web Development": [
        {"q": "What is React?", "options": ["Backend framework", "JavaScript library for UI", "Database", "Server"], "a": 1},
        {"q": "What is REST API?", "options": ["Real-time API", "Representational State Transfer", "Remote API", "Rapid API"], "a": 1},
        {"q": "What is the purpose of useEffect in React?", "options": ["Styling", "Side effects and lifecycle", "Routing", "State management"], "a": 1},
        {"q": "What is JSX?", "options": ["JavaScript XML", "Java XML", "JSON XML", "JQuery XML"], "a": 0},
        {"q": "What is the virtual DOM?", "options": ["Real DOM", "JavaScript representation of DOM", "Database", "Server"], "a": 1},
        {"q": "What is Node.js?", "options": ["Frontend framework", "JavaScript runtime", "Database", "Browser"], "a": 1},
        {"q": "What is the purpose of state in React?", "options": ["Store data", "Manage component data", "Style components", "Route pages"], "a": 1},
    ],
    "Cloud Computing": [
        {"q": "What is AWS?", "options": ["Amazon Web Services", "Azure Web Services", "Application Web Services", "Advanced Web Services"], "a": 0},
        {"q": "What is Docker?", "options": ["Database", "Containerization platform", "Cloud provider", "Programming language"], "a": 1},
        {"q": "What is Kubernetes?", "options": ["Container orchestrator", "Container platform", "Cloud service", "Database"], "a": 0},
        {"q": "What is serverless computing?", "options": ["No servers", "Managed server infrastructure", "Local servers", "Private servers"], "a": 1},
        {"q": "What is IaaS?", "options": ["Infrastructure as a Service", "Internet as a Service", "Integration as a Service", "Interface as a Service"], "a": 0},
    ],
    "IoT": [
        {"q": "What does IoT stand for?", "options": ["Internet of Things", "Internet of Technology", "Integration of Things", "Interface of Things"], "a": 0},
        {"q": "What is MQTT?", "options": ["Message Queue Telemetry Transport", "Machine Query Transport", "Mobile Query Transport", "Message Queue Technology"], "a": 0},
        {"q": "What is edge computing?", "options": ["Cloud computing", "Processing near data source", "Centralized computing", "Remote computing"], "a": 1},
    ],
    "Robotics": [
        {"q": "What does ROS stand for?", "options": ["Robot Operating System", "Robot Operating Software", "Remote Operating System", "Robotic Operating System"], "a": 0},
        {"q": "What is SLAM in robotics?", "options": ["Simultaneous Localization and Mapping", "Single Localization and Mapping", "System Localization and Mapping", "Smart Localization and Mapping"], "a": 0},
        {"q": "What is the purpose of sensors in robotics?", "options": ["Control actuators", "Gather environmental data", "Store data", "Process data"], "a": 1},
    ],
}


@router.get("/quiz/{domain:path}")
async def quiz(domain: str, user: User = Depends(get_current_user)):
    """
    Get domain-specific quiz questions.
    
    Args:
        domain: Domain name (e.g., "AI/ML", "Cybersecurity", etc.)
        user: Authenticated user
        
    Returns:
        Dictionary with domain and randomized quiz items (max 10 questions)
    """
    try:
        # Decode domain if URL encoded (e.g., AI%2FML -> AI/ML)
        import urllib.parse
        domain_decoded = urllib.parse.unquote(domain)
        
        # Get question bank for domain, fallback to AI/ML
        bank = _QUESTIONS.get(domain_decoded, _QUESTIONS.get("AI/ML", []))
        
        if not bank:
            raise HTTPException(status_code=404, detail=f"No questions available for domain: {domain_decoded}")
        
        # Select up to 10 random questions and shuffle options
        selected = sample(bank, min(10, len(bank))) if len(bank) > 10 else bank
        
        # Shuffle options for each question (but keep track of correct answer)
        for q in selected:
            original_answer = q['options'][q['a']]
            options_copy = q['options'].copy()
            shuffle(options_copy)
            q['options'] = options_copy
            q['a'] = options_copy.index(original_answer)
        
        return {
            "domain": domain_decoded,
            "items": selected,
            "total_questions": len(selected)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")
