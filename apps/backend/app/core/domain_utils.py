_DOMAIN_ALIASES: dict[str, str] = {
    "ai": "ai_ml",
    "ml": "ai_ml",
    "machine_learning": "ai_ml",
    "artificial_intelligence": "ai_ml",
    "data": "data_science",
    "datascience": "data_science",
    "analytics": "data_science",
    "cyber": "cybersecurity",
    "security": "cybersecurity",
    "cloud": "cloud_devops",
    "devops": "cloud_devops",
    "web": "web_development",
    "frontend": "web_development",
    "fullstack": "web_development",
    "iot": "iot",
    "robotics": "iot",
}


def normalize_domain(domain: str | None) -> str:
    if not domain:
        return ""
    normalized = (
        domain.strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return _DOMAIN_ALIASES.get(normalized, normalized)


