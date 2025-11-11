from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable
import hashlib


DomainKey = str
Difficulty = str
TimeCommitment = str


def _normalize_domain(domain: str) -> DomainKey:
    return (
        domain.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


def _make_project_id(domain: str, title: str) -> str:
    slug_src = f"{domain}-{title}".lower()
    slug = (
        slug_src.replace(" ", "-")
        .replace("/", "-")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )
    digest = hashlib.md5(slug_src.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}"


@dataclass
class ProjectResource:
    title: str
    url: str
    kind: str  # video, course, article, repo
    provider: Optional[str] = None


@dataclass
class ProjectStep:
    id: str
    title: str
    description: str
    xp_reward: int = 25
    resources: List[ProjectResource] = field(default_factory=list)


@dataclass
class ProjectDefinition:
    id: str
    domain: str
    title: str
    difficulty: Difficulty
    time_commitment: TimeCommitment
    estimated_time: str
    tech_stack: List[str]
    tags: List[str]
    summary: str
    why_it_matters: str
    description: str
    dataset: Optional[str]
    market_alignment: str
    steps: List[ProjectStep]
    resources: List[ProjectResource]
    github_template: Optional[str] = None
    expected_output: Optional[str] = None


def _resource(
    title: str,
    url: str,
    kind: str,
    provider: Optional[str] = None,
) -> ProjectResource:
    return ProjectResource(title=title, url=url, kind=kind, provider=provider)


def _step(
    id: str,
    title: str,
    description: str,
    xp_reward: int,
    resources: Optional[List[ProjectResource]] = None,
) -> ProjectStep:
    return ProjectStep(
        id=id,
        title=title,
        description=description,
        xp_reward=xp_reward,
        resources=resources or [],
    )


PROJECT_CATALOG: Dict[DomainKey, List[ProjectDefinition]] = {}


def _register_project(domain_label: str, payload: Dict) -> None:
    domain_key = _normalize_domain(domain_label)
    project_id = payload.get("id") or _make_project_id(domain_key, payload["title"])
    PROJECT_CATALOG.setdefault(domain_key, []).append(
        ProjectDefinition(
            id=project_id,
            domain=domain_key,
            title=payload["title"],
            difficulty=payload["difficulty"],
            time_commitment=payload["time_commitment"],
            estimated_time=payload["estimated_time"],
            tech_stack=payload["tech_stack"],
            tags=payload.get("tags", []),
            summary=payload["summary"],
            why_it_matters=payload["why_it_matters"],
            description=payload["description"],
            dataset=payload.get("dataset"),
            market_alignment=payload["market_alignment"],
            steps=payload["steps"],
            resources=payload["resources"],
            github_template=payload.get("github_template"),
            expected_output=payload.get("expected_output"),
        )
    )


def _project(
    domain: str,
    title: str,
    *,
    difficulty: Difficulty,
    time_commitment: TimeCommitment,
    estimated_time: str,
    tech_stack: Iterable[str],
    tags: Iterable[str],
    summary: str,
    why_it_matters: str,
    description: str,
    market_alignment: str,
    dataset: Optional[str],
    steps: List[ProjectStep],
    resources: List[ProjectResource],
    github_template: Optional[str] = None,
    expected_output: Optional[str] = None,
) -> None:
    _register_project(
        domain,
        {
            "title": title,
            "difficulty": difficulty,
            "time_commitment": time_commitment,
            "estimated_time": estimated_time,
            "tech_stack": list(tech_stack),
            "tags": list(tags),
            "summary": summary,
            "why_it_matters": why_it_matters,
            "description": description,
            "dataset": dataset,
            "market_alignment": market_alignment,
            "steps": steps,
            "resources": resources,
            "github_template": github_template,
            "expected_output": expected_output,
        },
    )


# -- Project Seed Data ----------------------------------------------------- #

_project(
    "AI/ML",
    "Real-Time Object Detection App",
    difficulty="Intermediate",
    time_commitment="medium",
    estimated_time="10-14 days",
    tech_stack=["TensorFlow Lite", "OpenCV", "Python", "Raspberry Pi"],
    tags=["edge-ai", "computer-vision", "iot"],
    summary="Build an edge-deployable object detection application optimised for Raspberry Pi.",
    why_it_matters="Edge vision workloads are exploding in smart retail, robotics, and manufacturing. Recruiters evaluate your ability to compress, quantise, and deploy models outside the cloud.",
    description="You will train an object detection model, convert it to TensorFlow Lite, and deploy it to Raspberry Pi with a lightweight UI that highlights detections in real-time.",
    market_alignment="Edge deployment and computer vision are trending on most AI job breakdowns (+38% YoY across Indeed / LinkedIn postings).",
    dataset="https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection",
    steps=[
        _step(
            "collect",
            "Assemble and label dataset",
            "Collect or download a safety gear dataset, clean labels, and split into train/validation/test sets.",
            xp_reward=40,
            resources=[
                _resource(
                    "LabelImg quickstart",
                    "https://github.com/tzutalin/labelImg",
                    "article",
                    "GitHub",
                )
            ],
        ),
        _step(
            "train",
            "Train and evaluate detection model",
            "Fine-tune an SSD MobileNet model and report mAP + FPS benchmarks.",
            xp_reward=60,
            resources=[
                _resource(
                    "TensorFlow Object Detection Tutorial",
                    "https://www.youtube.com/watch?v=pDXdlXlaCco",
                    "video",
                    "YouTube",
                )
            ],
        ),
        _step(
            "optimise",
            "Quantise & convert to TFLite",
            "Export the model to TensorFlow Lite and evaluate accuracy vs. latency trade-offs.",
            xp_reward=50,
            resources=[
                _resource(
                    "TensorFlow Lite Model Optimization",
                    "https://www.tensorflow.org/lite/performance/model_optimization",
                    "article",
                    "TensorFlow",
                )
            ],
        ),
        _step(
            "deploy",
            "Deploy to Raspberry Pi with live overlay",
            "Implement a Python script with OpenCV overlays and run real-time inference on the device.",
            xp_reward=70,
            resources=[
                _resource(
                    "Deploying TFLite on Raspberry Pi",
                    "https://www.youtube.com/watch?v=aimSGOAUI8Y",
                    "video",
                    "YouTube",
                )
            ],
        ),
        _step(
            "report",
            "Document and demo impact",
            "Record a demo video, publish benchmarks, and note future improvements in README.",
            xp_reward=30,
        ),
    ],
    resources=[
        _resource(
            "Edge Vision in Production",
            "https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch10.html",
            "article",
        ),
        _resource(
            "Edge AI Product Tips",
            "https://www.youtube.com/watch?v=Tq6C99C7JtY",
            "video",
            "YouTube",
        ),
        _resource(
            "Sample Raspberry Pi Vision Repo",
            "https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi",
            "repo",
            "GitHub",
        ),
    ],
    github_template="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi",
    expected_output="Short demo clip + README describing edge inference metrics (FPS, precision, power draw).",
)

_project(
    "AI/ML",
    "Titanic Survival Sprint",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="2-3 days",
    tech_stack=["Python", "Pandas", "scikit-learn"],
    tags=["classification", "feature-engineering", "kaggle"],
    summary="Build and submit a Kaggle-ready baseline model for the Titanic survival dataset.",
    why_it_matters="Interviewers love to see how you scope, baseline, and iterate quickly on classic datasets while telling a clear story.",
    description="Own the full mini-project: data cleaning, feature engineering, model selection, cross-validation, and competition submission notes.",
    market_alignment="Rapid prototyping and storytelling with tabular data is a day-one expectation for AI product roles.",
    dataset="https://www.kaggle.com/c/titanic/data",
    steps=[
        _step(
            "profile",
            "Profile data & identify gaps",
            "Audit missing values, outliers, and column semantics. Publish quick EDA visuals.",
            xp_reward=20,
            resources=[
                _resource(
                    "EDA in 15 minutes",
                    "https://www.youtube.com/watch?v=rlF5g9cxZV4",
                    "video",
                    "StatQuest",
                )
            ],
        ),
        _step(
            "features",
            "Engineer predictive features",
            "Create family size, title, and ticket group features; document intuition and impact.",
            xp_reward=25,
        ),
        _step(
            "model",
            "Train & validate baseline model",
            "Compare logistic regression vs. gradient boosting with cross-validation.",
            xp_reward=30,
        ),
        _step(
            "explain",
            "Explain results & next steps",
            "Summarise feature importances, error analysis, and how you'd improve with more time.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Kaggle Titanic Starter Guide",
            "https://www.kaggle.com/code/startupsci/titanic-data-science-solutions",
            "article",
        ),
        _resource(
            "scikit-learn Model Evaluation",
            "https://scikit-learn.org/stable/modules/model_evaluation.html",
            "article",
        ),
    ],
    github_template="https://github.com/awesomedata/awesome-public-datasets/tree/master/Datasets/titanic",
    expected_output="Notebook with narrative EDA, model comparisons, and Kaggle submission CSV.",
)

_project(
    "AI/ML",
    "LLM Prompt Quality Analyzer",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="3-4 weeks",
    tech_stack=["Python", "LangChain", "OpenAI API", "FastAPI", "PostgreSQL"],
    tags=["llm", "prompt-engineering", "evaluation"],
    summary="Ship a SaaS-style evaluator that scores prompts with quantitative metrics and qualitative feedback.",
    why_it_matters="LLM evaluation is a top growth area: OpenAI, Anthropic, and enterprise teams assess prompt robustness before deployment.",
    description="Engineer a system that benchmarks prompts against rubrics (toxicity, hallucination risk, sentiment) and suggests improvements using LLM self-critique patterns.",
    market_alignment="Prompt evaluation + LLMOps roles surged 52% in postings this quarter (LinkedIn Emerging Jobs report).",
    dataset=None,
    steps=[
        _step(
            "design",
            "Design evaluation rubric",
            "Draft the scoring criteria and capture baseline prompts to test against.",
            xp_reward=40,
        ),
        _step(
            "llm_eval",
            "Implement automated evaluation pipeline",
            "Use LangChain evaluators or custom OpenAI calls to score prompts.",
            xp_reward=70,
            resources=[
                _resource(
                    "LangChain Evaluators Guide",
                    "https://python.langchain.com/docs/guides/evaluation/",
                    "article",
                )
            ],
        ),
        _step(
            "backend",
            "Expose REST API + persistence",
            "Create FastAPI endpoints with PostgreSQL models for prompt submissions and reports.",
            xp_reward=50,
            resources=[
                _resource(
                    "FastAPI & SQLModel Tutorial",
                    "https://www.youtube.com/watch?v=0sOvCWFmrtA",
                    "video",
                    "freeCodeCamp",
                )
            ],
        ),
        _step(
            "frontend",
            "Build reporter dashboard",
            "Render charts summarising prompt scores and highlight actionable recommendations.",
            xp_reward=50,
        ),
        _step(
            "synthesis",
            "Ship launch story",
            "Write a blog or README documenting learnings, metrics, and next steps.",
            xp_reward=30,
        ),
    ],
    resources=[
        _resource(
            "Prompt Evaluation Playbook",
            "https://www.anthropic.com/index/prompt-evaluation-playbook",
            "article",
        ),
        _resource(
            "OpenAI Evals Quickstart",
            "https://github.com/openai/evals",
            "repo",
            "GitHub",
        ),
    ],
    github_template="https://github.com/langchain-ai/langchain-evals-template",
    expected_output="Dashboard screenshots + benchmark table comparing prompts before/after optimisation.",
)

_project(
    "AI/ML",
    "MLOps Drift Watchdog",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4-5 weeks",
    tech_stack=["Python", "FastAPI", "Evidently AI", "Prometheus", "Grafana"],
    tags=["mlops", "monitoring", "observability"],
    summary="Build an end-to-end monitoring system that detects model/data drift and auto-notifies stakeholders.",
    why_it_matters="Hiring teams expect production ML engineers to pair models with reliable monitoring and rollback stories.",
    description="Instrument a deployed model with drift checks, statistical testing, alerting, and dashboards tied to business KPIs.",
    market_alignment="87% of ML teams report drift incidents—platform roles now assess observability rigor.",
    dataset="https://github.com/evidentlyai/datasets/blob/main/README.md",
    steps=[
        _step(
            "baseline",
            "Ship reference scoring API",
            "Expose a FastAPI endpoint with a persisted baseline model and prediction logging.",
            xp_reward=50,
        ),
        _step(
            "monitor",
            "Implement data & concept drift checks",
            "Use Evidently or custom stats to compare live data against training reference windows.",
            xp_reward=70,
            resources=[
                _resource(
                    "Evidently Drift Tutorial",
                    "https://docs.evidentlyai.com/tutorials/drift-detection",
                    "article",
                )
            ],
        ),
        _step(
            "alert",
            "Wire alerting + on-call playbook",
            "Trigger Slack/email/pager alerts with contextual lineage and recommended mitigations.",
            xp_reward=60,
        ),
        _step(
            "dashboard",
            "Publish observability dashboards",
            "Create Grafana dashboards showing drift metrics, SLA trends, and experiment history.",
            xp_reward=50,
        ),
        _step(
            "retro",
            "Run ops retrospective",
            "Document a simulated incident, response timeline, and backlog for reliability upgrades.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "ML Observability Checklist",
            "https://www.tecton.ai/blog/ml-observability-checklist/",
            "article",
        ),
        _resource(
            "Prometheus FastAPI Exporter",
            "https://github.com/stephenhillier/starlette_exporter",
            "repo",
            "GitHub",
        ),
    ],
    github_template="https://github.com/evidentlyai/evidently/tree/main/examples/fastapi",
    expected_output="Monitoring dashboard screenshots + incident runbook PDF + demo video.",
)

_project(
    "AI/ML",
    "LLM Guardrails Spike",
    difficulty="Advanced",
    time_commitment="short",
    estimated_time="3 days",
    tech_stack=["Python", "LangChain", "OpenAI API", "Guardrails AI"],
    tags=["llm", "safety", "prompt-engineering"],
    summary="Prototype a guardrails layer that filters toxic or off-policy LLM responses with automated tests.",
    why_it_matters="Platform teams expect rapid prototyping of safety layers before greenlighting LLM features.",
    description="Wrap an LLM endpoint with guardrails, create adversarial prompts, and document test coverage plus failure analysis.",
    market_alignment="Responsible AI interviews emphasise how fast you can detect and block jailbreak attempts.",
    dataset=None,
    steps=[
        _step(
            "plan",
            "Define misuse scenarios",
            "Catalogue risky categories (PII, self-harm, toxicity) and success metrics.",
            xp_reward=30,
        ),
        _step(
            "implement",
            "Implement guardrails policies",
            "Use Guardrails AI or custom regex/classifier to enforce structured, policy-compliant answers.",
            xp_reward=40,
        ),
        _step(
            "attack",
            "Script adversarial prompts",
            "Automate red-teaming with curated jailbreak prompts; log failures.",
            xp_reward=35,
        ),
        _step(
            "report",
            "Publish risk review",
            "Summarise residual risks, QA coverage, and playbook for product stakeholders.",
            xp_reward=25,
        ),
    ],
    resources=[
        _resource(
            "Guardrails AI Quickstart",
            "https://docs.guardrailsai.com/quickstart",
            "article",
        ),
        _resource(
            "OpenAI Safety Best Practices",
            "https://platform.openai.com/docs/guides/safety-best-practices",
            "article",
        ),
    ],
    github_template="https://github.com/shreyashankar/gptguard",
    expected_output="Repo with guardrails config, adversarial prompt scripts, and risk memo PDF.",
)

_project(
    "AI/ML",
    "Few-Shot Classifier Workshop",
    difficulty="Intermediate",
    time_commitment="short",
    estimated_time="2 days",
    tech_stack=["Python", "PyTorch", "Hugging Face", "Weights & Biases"],
    tags=["few-shot", "transfer-learning", "mlops"],
    summary="Fine-tune a pretrained transformer with fewer than 100 labelled samples and compare uplift vs zero-shot baselines.",
    why_it_matters="Hiring panels want to see you adapt foundation models quickly while controlling cost and overfitting.",
    description="Experiment with parameter-efficient fine-tuning, track metrics, and synthesise findings for stakeholders.",
    market_alignment="Rapid adaptation of foundation models is a top requirement in AI product and platform roles.",
    dataset="https://huggingface.co/datasets/SetFit/sst2",
    steps=[
        _step(
            "baseline",
            "Establish zero-shot benchmark",
            "Evaluate pre-trained model without fine-tuning; log accuracy, precision/recall, and failure cases.",
            xp_reward=20,
        ),
        _step(
            "fewshot",
            "Curate balanced few-shot sample",
            "Select representative samples, perform data augmentation, and document selection heuristics.",
            xp_reward=20,
        ),
        _step(
            "finetune",
            "Run parameter-efficient fine-tuning",
            "Apply LoRA/Adapters, compare with full fine-tuning, and track metrics in W&B.",
            xp_reward=30,
            resources=[
                _resource(
                    "PEFT Quickstart",
                    "https://huggingface.co/docs/peft/index",
                    "article",
                )
            ],
        ),
        _step(
            "report",
            "Publish experiment report",
            "Create a report summarising results, trade-offs, and next iteration ideas.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Few-Shot Text Classification",
            "https://huggingface.co/blog/setfit",
            "article",
        ),
        _resource(
            "Weights & Biases Reports",
            "https://docs.wandb.ai/guides/reports",
            "article",
        ),
    ],
    github_template="https://github.com/huggingface/setfit",
    expected_output="Notebook + W&B report comparing zero-shot, few-shot, and full fine-tuning results.",
)

_project(
    "AI/ML",
    "Autonomous Experiment Orchestrator",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4-5 weeks",
    tech_stack=["Prefect", "Metaflow", "Docker", "PostgreSQL", "React"],
    tags=["mlops", "pipeline", "platform"],
    summary="Design a self-service ML experimentation platform with orchestration, scheduling, and experiment dashboards.",
    why_it_matters="Staff-level ML engineers are expected to build internal platforms that accelerate model delivery.",
    description="Implement workflow orchestration, metadata store, UI dashboards, and governance guardrails for experimentation.",
    market_alignment="Companies are investing heavily in ML productivity platforms, making platform experience highly sought after.",
    dataset=None,
    steps=[
        _step(
            "blueprint",
            "Produce platform architecture blueprint",
            "Define personas, workflow lifecycle, metadata schemas, and security constraints.",
            xp_reward=60,
        ),
        _step(
            "orchestrate",
            "Implement orchestration backbone",
            "Wire Prefect/Metaflow flows, containerize workloads, and store run lineage.",
            xp_reward=70,
        ),
        _step(
            "ui",
            "Ship run analytics UI",
            "Build React dashboard for run status, metrics comparison, and experiment notes.",
            xp_reward=60,
        ),
        _step(
            "govern",
            "Add governance & approvals",
            "Enforce cost ceilings, data residency checks, and review workflows before execution.",
            xp_reward=50,
        ),
        _step(
            "handoff",
            "Document rollout & ops guide",
            "Create onboarding docs, SRE playbook, and demo for stakeholders.",
            xp_reward=50,
        ),
    ],
    resources=[
        _resource(
            "Metaflow Architecture Guide",
            "https://docs.metaflow.org/getting-started/architecture",
            "article",
        ),
        _resource(
            "Prefect Orchestration Patterns",
            "https://docs.prefect.io/latest/guides/flows/",
            "article",
        ),
        _resource(
            "Building ML Platforms",
            "https://fullstackdeeplearning.com/seminar/building-ml-platforms",
            "video",
            "Full Stack Deep Learning",
        ),
    ],
    github_template="https://github.com/Netflix/metaflow",
    expected_output="Platform repo + architecture deck + demo recording showcasing experiment lifecycle.",
)

_project(
    "Data Science",
    "Customer Churn Prediction Dashboard",
    difficulty="Intermediate",
    time_commitment="medium",
    estimated_time="12-16 days",
    tech_stack=["Python", "scikit-learn", "Pandas", "Plotly Dash"],
    tags=["dashboard", "machine-learning", "retention"],
    summary="Predict churn and embed insights in an interactive executive dashboard.",
    why_it_matters="Retention analytics projects demonstrate end-to-end thinking—data prep, modelling, storytelling.",
    description="You will build a churn model, explain the drivers via SHAP, and expose insights in a Dash dashboard tailored for business stakeholders.",
    market_alignment="Customer retention analytics appear in 40% of entry-level data roles across fintech, SaaS, and telecom.",
    dataset="https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    steps=[
        _step(
            "ingest",
            "Ingest & profile data",
            "Load the dataset, handle missing values, and publish a data dictionary.",
            xp_reward=30,
            resources=[
                _resource(
                    "Pandas Profiling Tutorial",
                    "https://www.youtube.com/watch?v=1d4EFsHC63s",
                    "video",
                    "Data Professor",
                )
            ],
        ),
        _step(
            "model",
            "Train churn classifier with explainability",
            "Test multiple models, pick the best, and compute SHAP feature importances.",
            xp_reward=60,
            resources=[
                _resource(
                    "Interpretable ML Book - SHAP",
                    "https://christophm.github.io/interpretable-ml-book/shap.html",
                    "article",
                )
            ],
        ),
        _step(
            "dashboard",
            "Build Plotly Dash insights app",
            "Communicate metrics, segment drill-downs, and SHAP plots in a responsive dashboard.",
            xp_reward=50,
        ),
        _step(
            "recommend",
            "Frame business recommendations",
            "Write actionable retention strategies tied to dashboard metrics.",
            xp_reward=30,
        ),
        _step(
            "ship",
            "Deploy & demo",
            "Deploy the dashboard (Render/Heroku) and record a walkthrough video.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "Plotly Dash Crash Course",
            "https://www.youtube.com/watch?v=hSPmj7mK6ng",
            "video",
            "Charming Data",
        ),
        _resource(
            "ML Explainability Guide",
            "https://www.kaggle.com/code/learnai/interpretability",
            "article",
        ),
    ],
    github_template="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-churn",
    expected_output="Hosted dashboard URL + README summarising churn insights and recommended actions.",
)

_project(
    "Data Science",
    "Retail A/B Testing Insights",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="3 days",
    tech_stack=["Python", "Pandas", "SciPy", "Plotly"],
    tags=["experimentation", "ab-testing", "visualisation"],
    summary="Design and analyse an A/B test for a retail landing page, presenting actionable insights.",
    why_it_matters="Product analysts need to move from raw experiment logs to confident recommendations fast.",
    description="Clean experiment logs, verify assumptions, run statistical tests, and craft executive-ready visuals.",
    market_alignment="Experimentation literacy is now a core requirement in data science job postings.",
    dataset="https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
    steps=[
        _step(
            "ingest",
            "Clean experiment dataset",
            "Load CSVs, handle bots/anomalies, ensure randomisation integrity, and publish data dictionary.",
            xp_reward=25,
        ),
        _step(
            "assess",
            "Validate test assumptions",
            "Check sample ratio mismatch, variance equality, and independence assumptions.",
            xp_reward=25,
        ),
        _step(
            "test",
            "Run hypothesis tests",
            "Use t-test or non-parametric alternative; compute uplift and confidence intervals.",
            xp_reward=30,
        ),
        _step(
            "story",
            "Tell the experiment story",
            "Create a one-pager with charts, recommendations, and next experiments.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "AB Testing Crash Course",
            "https://www.youtube.com/watch?v=4I2W7l45cLM",
            "video",
            "Luke Barousse",
        ),
        _resource(
            "Practical Guide to p-values",
            "https://www.evanmiller.org/how-not-to-run-an-ab-test.html",
            "article",
        ),
    ],
    github_template="https://github.com/andreyvoronkov/ab-testing-case-study",
    expected_output="Interactive notebook + stakeholder slide deck with recommendation and next steps.",
)

_project(
    "Data Science",
    "Operational SQL Pulse",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="1-2 days",
    tech_stack=["PostgreSQL", "dbt", "Metabase"],
    tags=["analytics", "sql", "dashboard"],
    summary="Stand up a lightweight analytics stack that ships daily KPIs, anomaly alerts, and exec-ready visuals.",
    why_it_matters="Modern data teams expect analysts to deliver business insights fast with pragmatic SQL tooling.",
    description="Model core tables with dbt, build KPI dashboards in Metabase, and automate notifications for anomalies.",
    market_alignment="Companies scaling analytics teams prize engineers who can bootstrap analytics in days, not weeks.",
    dataset="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
    steps=[
        _step(
            "model",
            "Model core entities with dbt",
            "Create staging + mart models for orders, customers, revenue, and retention.",
            xp_reward=20,
        ),
        _step(
            "dashboard",
            "Visualise KPIs",
            "Build Metabase dashboards covering revenue trend, retention cohorts, and refunds.",
            xp_reward=20,
        ),
        _step(
            "alert",
            "Automate anomaly detection",
            "Schedule SQL-based alerts that trigger when metrics deviate from historical bounds.",
            xp_reward=20,
        ),
        _step(
            "communicate",
            "Summarise business impact",
            "Record a Loom walkthrough explaining insights, risks, and recommended actions.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "dbt Fundamentals",
            "https://docs.getdbt.com/guides/manual-install",
            "article",
        ),
        _resource(
            "Metabase Dashboard Guide",
            "https://www.metabase.com/learn/dashboards/metrics-dashboards",
            "article",
        ),
    ],
    github_template="https://github.com/dbt-labs/jaffle_shop",
    expected_output="dbt project + Metabase export + insight summary deck.",
)

_project(
    "Data Science",
    "Streaming Anomaly Detection Pipeline",
    difficulty="Advanced",
    time_commitment="medium",
    estimated_time="2 weeks",
    tech_stack=["Apache Kafka", "Spark Structured Streaming", "Python", "Docker"],
    tags=["streaming", "anomaly-detection", "mlops"],
    summary="Detect anomalies in IoT sensor streams with latency SLAs and automatic incident dispatch.",
    why_it_matters="Real-time analytics roles expect fluency with streaming architectures and online inference.",
    description="Ingest live sensor data, maintain rolling features, flag anomalies, and push alerts to downstream systems.",
    market_alignment="Streaming ML roles grew 31% YoY as enterprises instrument factories and fleets.",
    dataset="https://github.com/numenta/NAB",
    steps=[
        _step(
            "ingest",
            "Provision streaming stack",
            "Spin up Kafka + Spark containers; define schemas and partitions for sensor topics.",
            xp_reward=45,
        ),
        _step(
            "features",
            "Compute rolling features",
            "Implement sliding windows, z-score, and EWMA metrics for anomaly scoring.",
            xp_reward=50,
        ),
        _step(
            "model",
            "Train & deploy online detector",
            "Train isolation forest / LSTM, serialise model, and expose Spark UDF for scoring.",
            xp_reward=55,
        ),
        _step(
            "alert",
            "Publish incidents",
            "Trigger Slack/webhook alerts with contextual metrics and recommended actions.",
            xp_reward=40,
        ),
        _step(
            "observe",
            "Monitor pipeline health",
            "Instrument lag, throughput, and anomaly rate dashboards via Prometheus + Grafana.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "Spark Streaming Guide",
            "https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html",
            "article",
        ),
        _resource(
            "Kafka Streams Crash Course",
            "https://www.youtube.com/watch?v=EiDLKECLcZw",
            "video",
            "Confluent",
        ),
    ],
    github_template="https://github.com/confluentinc/kafka-streams-examples",
    expected_output="Docker-compose project + Grafana dashboards + incident playbook.",
)

_project(
    "Data Science",
    "Marketing Mix Optimiser",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4 weeks",
    tech_stack=["Python", "PyMC", "Prophet", "Dash"],
    tags=["media-mix", "bayesian", "forecasting"],
    summary="Model multi-channel marketing spend vs conversions and deliver an interactive planning dashboard for executives.",
    why_it_matters="Revenue teams rely on data scientists to translate modeling into budget allocation guidance.",
    description="Fit a Bayesian MMM, simulate scenarios, and ship a dashboard that recommends optimal spend allocations.",
    market_alignment="Privacy changes have revived MMM demand, making hands-on experience a differentiator in interviews.",
    dataset="https://github.com/facebookexperimental/Robyn/tree/main/data",
    steps=[
        _step(
            "prepare",
            "Prepare spend & conversion dataset",
            "Cleanse channel spend, conversions, seasonality, and external drivers into model-ready format.",
            xp_reward=40,
        ),
        _step(
            "model",
            "Fit Bayesian regression",
            "Estimate channel coefficients with PyMC/Stan; evaluate diagnostics and model fit.",
            xp_reward=60,
        ),
        _step(
            "simulate",
            "Simulate budget scenarios",
            "Build tooling to test budget reallocations and quantify expected lift.",
            xp_reward=50,
        ),
        _step(
            "dashboard",
            "Launch planning dashboard",
            "Expose Dash app summarising channel efficiency, saturation curves, and recommendations.",
            xp_reward=50,
        ),
        _step(
            "present",
            "Deliver CMO-ready narrative",
            "Create executive slides capturing insights, risks, and next experiments.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "PyMC Marketing Mix Tutorial",
            "https://www.pymc.io/projects/docs/en/stable/pymc-examples/examples/case_studies/marketing.html",
            "article",
        ),
        _resource(
            "Facebook Robyn",
            "https://facebookexperimental.github.io/Robyn/",
            "article",
        ),
    ],
    github_template="https://github.com/facebookexperimental/Robyn",
    expected_output="MMM notebook + Dash app + executive recommendation deck.",
)

_project(
    "Web Development",
    "Real-Time Chat App with Typing Indicator",
    difficulty="Intermediate",
    time_commitment="medium",
    estimated_time="10-12 days",
    tech_stack=["React", "Node.js", "Socket.io", "Redis"],
    tags=["real-time", "websockets", "fullstack"],
    summary="Develop a production-ready chat app with presence, typing status, and message persistence.",
    why_it_matters="Real-time collaboration skills are in demand for product engineering interviews (Slack, Linear, Notion).",
    description="Implement a Socket.io-powered app with reliable delivery, offline support, and observable metrics.",
    market_alignment="Realtime collaboration tooling is now a staple whiteboard challenge for frontend/fullstack roles.",
    dataset=None,
    steps=[
        _step(
            "setup",
            "Design UI & component architecture",
            "Create responsive chat UI components with dark/light themes and accessibility.",
            xp_reward=40,
        ),
        _step(
            "ws",
            "Implement WebSocket events",
            "Handle join/leave/typing events, message delivery, and ack retries.",
            xp_reward=60,
        ),
        _step(
            "storage",
            "Persist conversations & presence",
            "Use Redis or Mongo to store history and presence metadata, with TTL policies.",
            xp_reward=40,
        ),
        _step(
            "observability",
            "Add instrumentation & tests",
            "Add integration tests plus metrics (message latency, dropped events).",
            xp_reward=30,
        ),
        _step(
            "deploy",
            "Deploy to Render/ Railway",
            "Containerise and deploy both client & server; document CI/CD pipeline.",
            xp_reward=30,
        ),
    ],
    resources=[
        _resource(
            "Socket.io Chat Tutorial",
            "https://socket.io/get-started/chat",
            "article",
        ),
        _resource(
            "React Query + WebSockets Patterns",
            "https://tanstack.com/query/latest/docs/framework/react/guides/subscriptions",
            "article",
        ),
    ],
    github_template="https://github.com/socketio/chat-example",
    expected_output="Screen recording demonstrating typing indicator, offline handling, and deployment link.",
)

_project(
    "Web Development",
    "Personal Landing Page Sprint",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="2 days",
    tech_stack=["React", "Tailwind CSS", "Framer Motion"],
    tags=["frontend", "portfolio", "ui-ux"],
    summary="Ship a responsive, accessible personal landing page with motion micro-interactions.",
    why_it_matters="Junior engineers are often asked to prove they can deliver polished UX quickly.",
    description="Design, build, and deploy a landing page highlighting projects, skills, and contact channels.",
    market_alignment="Portfolios remain a differentiator—recruiters skim UI craft before reading resumes.",
    dataset=None,
    steps=[
        _step(
            "wireframe",
            "Sketch layout & content hierarchy",
            "Create a quick wireframe, define sections, and gather copy/assets.",
            xp_reward=15,
        ),
        _step(
            "build",
            "Implement responsive layout",
            "Code hero, about, projects, and contact sections with mobile-first Tailwind.",
            xp_reward=25,
        ),
        _step(
            "animate",
            "Add motion & accessibility polish",
            "Use Framer Motion for entrance animations and ensure WCAG-friendly semantics.",
            xp_reward=20,
        ),
        _step(
            "deploy",
            "Deploy & collect feedback",
            "Ship to Vercel/Netlify, gather peer feedback, and iterate on copy.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Tailwind Landing Page Tips",
            "https://www.youtube.com/watch?v=sKFW3wekld8",
            "video",
            "Fireship",
        ),
        _resource(
            "Accessible React Patterns",
            "https://www.smashingmagazine.com/2021/08/complete-guide-accessible-front-end-components/",
            "article",
        ),
    ],
    github_template="https://github.com/tailwindtoolbox/landing-page",
    expected_output="Deployed URL + Loom walkthrough highlighting motion details and accessibility tweaks.",
)

_project(
    "Web Development",
    "Accessibility Bug Bash",
    difficulty="Intermediate",
    time_commitment="short",
    estimated_time="2 days",
    tech_stack=["React", "Storybook", "Testing Library", "axe-core"],
    tags=["accessibility", "frontend", "quality"],
    summary="Audit an existing React component system for accessibility gaps and ship fixes with automated tests.",
    why_it_matters="Engineering teams expect developers to spot and fix a11y issues without waiting on design audits.",
    description="Run automated scans, implement fixes for keyboard navigation and semantics, and document guidelines for contributors.",
    market_alignment="Enterprise contracts increasingly require WCAG 2.1 AA compliance, making accessibility expertise highly marketable.",
    dataset=None,
    steps=[
        _step(
            "audit",
            "Capture baseline audit",
            "Run axe-core and Storybook a11y scans; triage critical issues.",
            xp_reward=20,
        ),
        _step(
            "fix",
            "Patch component issues",
            "Resolve aria roles, focus traps, and contrast problems with regression tests.",
            xp_reward=25,
        ),
        _step(
            "docs",
            "Publish accessibility checklist",
            "Document best practices and code examples for future contributions.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Deque axe DevTools",
            "https://www.deque.com/axe/",
            "article",
        ),
        _resource(
            "Testing Library A11y Queries",
            "https://testing-library.com/docs/dom-testing-library/api-queries#byrole",
            "article",
        ),
    ],
    github_template="https://github.com/storybookjs/storybook/tree/next/code/lib/components",
    expected_output="Accessibility audit report + patched component library + contributor guide.",
)

_project(
    "Web Development",
    "Realtime Whiteboard Collaboration Suite",
    difficulty="Advanced",
    time_commitment="medium",
    estimated_time="2-3 weeks",
    tech_stack=["Next.js", "WebRTC", "Liveblocks", "Redis", "TypeScript"],
    tags=["collaboration", "webrtc", "fullstack"],
    summary="Create a Figma-style whiteboard with multiplayer drawing, presence, and comment threads.",
    why_it_matters="Senior frontend interviews probe ability to blend realtime sync, performance, and DX polish.",
    description="Implement shared canvas, OT/CRDT syncing, video huddles, and persistence with audits.",
    market_alignment="Collaboration SaaS hiring managers look for evidence you can ship low-latency UX.",
    dataset=None,
    steps=[
        _step(
            "canvas",
            "Implement drawing engine",
            "Support shapes, freehand, text, and undo/redo with performant rendering.",
            xp_reward=55,
        ),
        _step(
            "sync",
            "Add realtime presence & CRDT",
            "Leverage Liveblocks or custom CRDT to sync strokes, cursors, and comments.",
            xp_reward=60,
        ),
        _step(
            "media",
            "Enable WebRTC huddles",
            "Embed lightweight audio/video rooms and status indicators.",
            xp_reward=45,
        ),
        _step(
            "persist",
            "Persist sessions",
            "Store boards, revisions, and access controls in Redis/Postgres.",
            xp_reward=40,
        ),
        _step(
            "polish",
            "Ship product polish & tests",
            "Add keyboard shortcuts, accessibility, end-to-end tests, and analytics events.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "CRDTs for Beginners",
            "https://josephg.com/blog/crdts-go-brrr/",
            "article",
        ),
        _resource(
            "Liveblocks Multiplayer Guide",
            "https://liveblocks.io/docs/guides/nextjs-multiplayer-whiteboard",
            "article",
        ),
    ],
    github_template="https://github.com/liveblocks/nextjs-whiteboard-example",
    expected_output="Deployed demo + product walkthrough video + architectural memo.",
)

_project(
    "Web Development",
    "SaaS Product Analytics Platform",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4 weeks",
    tech_stack=["Next.js", "tRPC", "Prisma", "PostgreSQL", "ClickHouse", "Tailwind"],
    tags=["analytics", "fullstack", "product-led-growth"],
    summary="Build a product analytics platform that ingests events, computes cohorts, and surfaces PLG metrics with alerts.",
    why_it_matters="Senior fullstack roles often test ability to blend backend data modeling with polished growth analytics UX.",
    description="Create event ingestion SDK, scalable warehouse, cohort explorer, activation dashboards, and alerting workflows.",
    market_alignment="Product-led growth tooling is in high demand; demonstrating hands-on experience sets candidates apart.",
    dataset=None,
    steps=[
        _step(
            "sdk",
            "Author tracking SDK",
            "Ship TypeScript SDK that batches events, retries, and supports user identity resolution.",
            xp_reward=60,
        ),
        _step(
            "ingest",
            "Implement ingestion service",
            "Use tRPC APIs + ClickHouse to persist and query millions of events efficiently.",
            xp_reward=70,
        ),
        _step(
            "cohorts",
            "Build cohort explorer",
            "Enable segmentation by plan, persona, lifecycle stage, and behaviour.",
            xp_reward=60,
        ),
        _step(
            "dashboards",
            "Design PLG dashboards",
            "Visualise activation, retention, and feature adoption with Tailwind + charting libs.",
            xp_reward=50,
        ),
        _step(
            "alerts",
            "Automate insights",
            "Trigger email/webhook alerts and CSV exports when metrics cross thresholds.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "ClickHouse Analytics Patterns",
            "https://clickhouse.com/blog",
            "article",
        ),
        _resource(
            "PostHog Growth Guides",
            "https://posthog.com/tutorials",
            "article",
        ),
    ],
    github_template="https://github.com/vercel/nextjs-postgres-auth-starter",
    expected_output="Deployed analytics app + SDK package + executive growth memo.",
)

_project(
    "Cybersecurity",
    "SIEM Log Analysis Dashboard",
    difficulty="Intermediate",
    time_commitment="medium",
    estimated_time="12-16 days",
    tech_stack=["ELK Stack", "Python", "Docker"],
    tags=["siem", "log-analysis", "blue-team"],
    summary="Create an ELK dashboard that surfaces suspicious patterns from simulated enterprise logs.",
    why_it_matters="Security analysts need to convert noisy logs into investigation-ready visualisations.",
    description="Spin up a SIEM stack, ingest logs from open datasets, map alerts to MITRE ATT&CK, and automate triage scripts.",
    market_alignment="SOC analyst postings emphasise SIEM tooling (Splunk/ELK) as a foundational skill.",
    dataset="https://github.com/OTRF/detection-hackathon-data",
    steps=[
        _step(
            "provision",
            "Provision ELK stack locally",
            "Containerise Elasticsearch, Logstash, Kibana, and Fleet for ingest.",
            xp_reward=40,
        ),
        _step(
            "ingest",
            "Ingest diverse log sources",
            "Feed Windows event logs, firewall logs, and AWS CloudTrail into ELK.",
            xp_reward=50,
        ),
        _step(
            "visualise",
            "Build detection dashboards",
            "Create visualisations for suspicious logins, privilege escalation, lateral movement.",
            xp_reward=50,
        ),
        _step(
            "automate",
            "Automate alerting workflow",
            "Write Python scripts or use ElastAlert to notify analysts with enrichment context.",
            xp_reward=40,
        ),
        _step(
            "report",
            "Document incident playbook",
            "Summarise findings, ATT&CK mapping, and response steps in a security playbook.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "Elastic Security Hands-on Lab",
            "https://www.elastic.co/security-labs",
            "article",
        ),
        _resource(
            "MITRE ATT&CK Mapping Guide",
            "https://attack.mitre.org/resources/working-with-attack/",
            "article",
        ),
    ],
    github_template="https://github.com/elastic/examples",
    expected_output="Kibana dashboard screenshots + playbook PDF referencing ATT&CK tactics.",
)

_project(
    "Cybersecurity",
    "Purple Team Attack Simulation",
    difficulty="Advanced",
    time_commitment="medium",
    estimated_time="2 weeks",
    tech_stack=["Atomic Red Team", "Azure Sentinel", "PowerShell", "Python"],
    tags=["purple-team", "incident-response", "automation"],
    summary="Execute adversary emulation plans and validate detections across the kill chain.",
    why_it_matters="Advanced security roles expect familiarity with red/blue collaboration and detection engineering.",
    description="Plan, execute, and document simulated attacks while tuning SIEM rules and response playbooks.",
    market_alignment="Purple teaming is now standard for enterprises hardening detection coverage.",
    dataset=None,
    steps=[
        _step(
            "plan",
            "Select adversary technique set",
            "Map Atomic Red Team tests to MITRE ATT&CK and define success metrics.",
            xp_reward=50,
        ),
        _step(
            "execute",
            "Run controlled attack simulations",
            "Trigger persistence, credential access, and lateral movement techniques.",
            xp_reward=60,
        ),
        _step(
            "detect",
            "Tune detection content",
            "Iterate Azure Sentinel rules/ML models to raise high-fidelity alerts.",
            xp_reward=55,
        ),
        _step(
            "respond",
            "Automate response playbooks",
            "Use Logic Apps/Python to contain compromised accounts or hosts automatically.",
            xp_reward=45,
        ),
        _step(
            "retro",
            "Produce after-action review",
            "Summarise gaps, successes, and roadmap for leadership.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "Atomic Red Team Docs",
            "https://atomicredteam.io/",
            "article",
        ),
        _resource(
            "Microsoft Sentinel Tutorials",
            "https://learn.microsoft.com/en-us/azure/sentinel/",
            "article",
        ),
    ],
    github_template="https://github.com/redcanaryco/atomic-red-team",
    expected_output="Attack runbook + tuned detection rules + leadership debrief deck.",
)

_project(
    "Cybersecurity",
    "Packet Capture Mini-Lab",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="3 days",
    tech_stack=["Wireshark", "Python", "Scapy"],
    tags=["networking", "traffic-analysis", "blue-team"],
    summary="Capture, inspect, and document suspicious network traffic patterns in a home lab environment.",
    why_it_matters="Security interviews probe for hands-on familiarity with packet analysis and tooling basics.",
    description="Spin up a small lab, capture HTTP/SSH sessions, identify anomalies, and document findings.",
    market_alignment="Foundational packet analysis remains a core SOC competency per CompTIA+ exams.",
    dataset=None,
    steps=[
        _step(
            "setup",
            "Provision mini lab",
            "Create two VMs and simulate traffic between them; enable packet capture.",
            xp_reward=20,
        ),
        _step(
            "capture",
            "Capture diverse protocols",
            "Record HTTP, DNS, and SSH sessions; export PCAPs for analysis.",
            xp_reward=25,
        ),
        _step(
            "analyze",
            "Identify anomalies",
            "Flag suspicious payloads, failed logins, and unusual ports using Wireshark filters.",
            xp_reward=25,
        ),
        _step(
            "report",
            "Write analyst report",
            "Summarise findings with ATT&CK mapping and remediation recommendations.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Wireshark 101",
            "https://www.youtube.com/watch?v=TkCSr30UojM",
            "video",
            "Hak5",
        ),
        _resource(
            "Scapy Packet Crafting Guide",
            "https://scapy.readthedocs.io/en/latest/introduction.html",
            "article",
        ),
    ],
    github_template="https://github.com/secdev/scapy/tree/master/doc/notebooks",
    expected_output="PCAP samples + analyst PDF with annotated screenshots & remediation plan.",
)

_project(
    "Cybersecurity",
    "Password Hygiene Blitz",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="2 days",
    tech_stack=["Python", "Have I Been Pwned API", "Hashcat"],
    tags=["identity", "security-awareness", "automation"],
    summary="Audit organisational password hygiene, detect breached credentials, and propose remediation playbooks.",
    why_it_matters="Identity hardening remains the most requested blue-team task for entry-level analysts.",
    description="Automate checks against breach corpuses, score password strength, and deliver remediation actions to leadership.",
    market_alignment="Credential theft is still the top breach vector, so proactive hygiene audits are highly valued.",
    dataset=None,
    steps=[
        _step(
            "collect",
            "Gather anonymised credential sample",
            "Generate hashes or obtain salted dump for analysis while preserving privacy.",
            xp_reward=15,
        ),
        _step(
            "check",
            "Check breach exposure",
            "Use Have I Been Pwned API and Hashcat rules to flag compromised or weak credentials.",
            xp_reward=25,
        ),
        _step(
            "recommend",
            "Draft remediation plan",
            "Summarise policy updates, MFA rollout, and employee education programme.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Have I Been Pwned API Docs",
            "https://haveibeenpwned.com/API/v3",
            "article",
        ),
        _resource(
            "OWASP Authentication Cheat Sheet",
            "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
            "article",
        ),
    ],
    github_template="https://github.com/andrew-d/static-bcrypt-cracker",
    expected_output="Automation scripts + breach exposure report + remediation checklist.",
)

_project(
    "Cybersecurity",
    "Cloud Threat Hunting Platform",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4 weeks",
    tech_stack=["AWS CloudTrail", "ElasticSearch", "Neo4j", "Python", "Terraform"],
    tags=["threat-hunting", "cloud-security", "graph-analytics"],
    summary="Build a hunt engineering platform that correlates cloud events, maps attack paths, and automates remediation.",
    why_it_matters="Cloud security teams expect engineers to operationalise hunts across sprawling multi-account estates.",
    description="Ingest CloudTrail, construct access graphs, codify hunt queries, and automate incident playbooks.",
    market_alignment="Cloud breach response is in high demand—graph-driven hunt tooling is a major interviewing topic.",
    dataset=None,
    steps=[
        _step(
            "ingest",
            "Automate event ingestion",
            "Stream CloudTrail into ElasticSearch/Neo4j with Terraform-based pipelines.",
            xp_reward=60,
        ),
        _step(
            "graph",
            "Model access graph",
            "Create relationships for principals, resources, and API calls to surface dangerous paths.",
            xp_reward=60,
        ),
        _step(
            "hunt",
            "Codify hunt queries",
            "Write saved searches detecting privilege escalation, persistence, and exfil patterns.",
            xp_reward=55,
        ),
        _step(
            "respond",
            "Automate response playbooks",
            "Trigger remediation workflows (IAM lockdown, ticketing) when hunts fire.",
            xp_reward=50,
        ),
        _step(
            "brief",
            "Deliver exec briefing",
            "Summarise findings, residual risk, and roadmap for CISO stakeholders.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "AWS CloudTrail Lake Guide",
            "https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-lake.html",
            "article",
        ),
        _resource(
            "BloodHound AWS",
            "https://github.com/BloodHoundAD/AWSBloodhound",
            "repo",
            "GitHub",
        ),
    ],
    github_template="https://github.com/salesforce/policy_sentry",
    expected_output="Terraform IaC + hunt playbooks + executive briefing deck.",
)

_project(
    "Cloud Computing",
    "Docker → Jenkins → Kubernetes CI/CD Pipeline",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="3-4 weeks",
    tech_stack=["Docker", "Jenkins", "Kubernetes", "Helm", "Grafana"],
    tags=["cicd", "devops", "kubernetes"],
    summary="Engineer a full CI/CD pipeline from Docker build to automated deployments on Kubernetes.",
    why_it_matters="DevOps engineers are assessed on pipeline automation, observability, and rollback strategies.",
    description="Create a modular pipeline that builds, tests, secures, and deploys a sample app to Kubernetes with rollbacks & monitoring.",
    market_alignment="Job surveys list Kubernetes + Jenkins as top DevOps stack demands (+44% YoY).",
    dataset=None,
    steps=[
        _step(
            "pipeline",
            "Design multi-stage Jenkins pipeline",
            "Author Jenkinsfile covering build, test, security scan, and packaging steps.",
            xp_reward=60,
        ),
        _step(
            "k8s",
            "Provision Kubernetes cluster",
            "Use kind or managed service; configure namespaces, RBAC, ingress.",
            xp_reward=50,
        ),
        _step(
            "deploy",
            "Automate Helm deployment",
            "Parameterise Helm charts for environment toggles and implement canary strategy.",
            xp_reward=70,
        ),
        _step(
            "observe",
            "Wire monitoring & logging",
            "Integrate Prometheus/Grafana dashboards with alerting for deployment health.",
            xp_reward=50,
        ),
        _step(
            "document",
            "Document pipeline & rollback playbook",
            "Write runbook covering rollback, incident response, and cost optimisations.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "Jenkins + Kubernetes CI/CD",
            "https://www.youtube.com/watch?v=dw7ZZf6h5iM",
            "video",
            "TechWorld with Nana",
        ),
        _resource(
            "Helm Best Practices",
            "https://helm.sh/docs/topics/charts/",
            "article",
        ),
    ],
    github_template="https://github.com/jenkinsci/kubernetes-cd-demo",
    expected_output="CI/CD pipeline diagram + Grafana dashboard screenshot + Jenkins build logs.",
)

_project(
    "Cloud Computing",
    "FinOps Cost Radar",
    difficulty="Intermediate",
    time_commitment="short",
    estimated_time="2-3 days",
    tech_stack=["AWS Cost Explorer", "Python", "QuickSight", "Terraform"],
    tags=["finops", "cost-optimization", "cloud"],
    summary="Build an automated dashboard that monitors cloud spend, flags anomalies, and recommends savings opportunities.",
    why_it_matters="Cloud teams want engineers who can pair delivery speed with ongoing cost governance.",
    description="Automate cost data extraction, detect cost spikes/idle resources, and publish executive-friendly insights.",
    market_alignment="FinOps skills now appear across cloud job descriptions as companies watch budgets closely.",
    dataset=None,
    steps=[
        _step(
            "ingest",
            "Automate cost ingestion",
            "Use Cost Explorer API to fetch tagged spend data into a warehouse.",
            xp_reward=25,
        ),
        _step(
            "analyze",
            "Detect anomalies & waste",
            "Highlight unusual spend patterns, idle resources, and rightsizing candidates.",
            xp_reward=25,
        ),
        _step(
            "visualise",
            "Publish savings dashboard",
            "Deploy QuickSight/Looker dashboard with savings tracker and weekly report.",
            xp_reward=25,
        ),
    ],
    resources=[
        _resource(
            "AWS Cost Explorer API",
            "https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_Operations_AWS_Cost_Explorer_Service.html",
            "article",
        ),
        _resource(
            "FinOps Foundation Guides",
            "https://www.finops.org/introduction/what-is-finops/",
            "article",
        ),
    ],
    github_template="https://github.com/aws-samples/aws-cost-explorer-report",
    expected_output="Cost dashboard + anomaly report + savings recommendation memo.",
)

_project(
    "Cloud Computing",
    "Multi-Region Serverless Commerce",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4 weeks",
    tech_stack=["AWS Lambda", "DynamoDB", "CloudFront", "Terraform"],
    tags=["serverless", "multi-region", "resilience"],
    summary="Design a resilient serverless storefront deployed across two AWS regions with failover.",
    why_it_matters="Cloud teams now assess your ability to balance latency, cost, and resilience in distributed systems.",
    description="Provision IaC, implement blue/green deploys, simulate region failure, and benchmark latency.",
    market_alignment="Resilience design interviews increasing with multi-region SLAs for commerce and fintech teams.",
    dataset=None,
    steps=[
        _step(
            "design",
            "Architect multi-region topology",
            "Diagram traffic flow, data replication, and failover strategies.",
            xp_reward=50,
        ),
        _step(
            "infra",
            "Provision IaC with Terraform",
            "Automate Lambda, API Gateway, DynamoDB global tables, and CloudFront distribution.",
            xp_reward=60,
        ),
        _step(
            "deploy",
            "Implement blue/green deploy",
            "Add weighted routing for canary releases and automated rollback.",
            xp_reward=60,
        ),
        _step(
            "resilience",
            "Run resilience game day",
            "Simulate region outage, capture metrics, and document lessons learned.",
            xp_reward=50,
        ),
        _step(
            "observability",
            "Create operations dashboard",
            "Expose latency, error, and failover metrics in CloudWatch dashboards.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "Serverless Multi-Region Patterns",
            "https://aws.amazon.com/blogs/architecture/active-active-multi-region-serverless-architectures/",
            "article",
        ),
        _resource(
            "Terraform AWS Lambda Example",
            "https://github.com/terraform-aws-modules/terraform-aws-lambda",
            "repo",
            "GitHub",
        ),
    ],
    github_template="https://github.com/aws-samples/serverless-patterns/tree/main/lambda",
    expected_output="Terraform repo + CloudWatch dashboards + chaos game day report.",
)

_project(
    "IoT",
    "Smart Agriculture IoT Sensor System",
    difficulty="Intermediate",
    time_commitment="medium",
    estimated_time="14-18 days",
    tech_stack=["ESP32", "MQTT", "Python", "InfluxDB", "Grafana"],
    tags=["iot", "sensors", "agritech"],
    summary="Design an IoT solution for monitoring soil moisture, temperature, and humidity with alerting.",
    why_it_matters="IoT deployments in agriculture are a top growth sector; recruiters expect full-stack IoT capability.",
    description="Prototype sensor firmware, IoT broker, data storage, and analytics dashboard with automation rules.",
    market_alignment="IoT agricultural solutions are a $25B market; employers value end-to-end prototyping and cloud integration skills.",
    dataset=None,
    steps=[
        _step(
            "hardware",
            "Assemble and program sensors",
            "Configure ESP32 with soil moisture + DHT sensors; calibrate readings and publish via MQTT.",
            xp_reward=50,
        ),
        _step(
            "ingest",
            "Build MQTT ingestion + storage",
            "Deploy MQTT broker (Mosquitto) and write ingestion service storing metrics in InfluxDB.",
            xp_reward=50,
        ),
        _step(
            "dashboard",
            "Visualise trends & thresholds",
            "Create Grafana dashboard with alert rules for critical thresholds.",
            xp_reward=40,
        ),
        _step(
            "automation",
            "Automate irrigation rule",
            "Trigger webhook or relay when soil moisture stays low for X minutes.",
            xp_reward=40,
        ),
        _step(
            "reporting",
            "Document deployment playbook",
            "Create README or Notion doc with wiring diagrams, calibration tips, and future roadmap.",
            xp_reward=30,
        ),
    ],
    resources=[
        _resource(
            "ESP32 Sensor Tutorial",
            "https://www.youtube.com/watch?v=BFF3R0jHKVY",
            "video",
        ),
        _resource(
            "MQTT Essentials",
            "https://www.hivemq.com/mqtt-essentials/",
            "article",
        ),
    ],
    github_template="https://github.com/shreyaspapi/IoT-smart-agriculture",
    expected_output="Grafana dashboard screenshot + automation diagram demonstrating irrigation trigger.",
)

_project(
    "IoT",
    "Home Energy Monitoring Quickstart",
    difficulty="Beginner",
    time_commitment="short",
    estimated_time="2 days",
    tech_stack=["ESP8266", "MQTT", "Home Assistant", "Python"],
    tags=["smart-home", "energy", "iot"],
    summary="Prototype an energy monitoring setup that tracks household power usage and identifies quick savings wins.",
    why_it_matters="IoT employers love to see scrappy proofs-of-concept that connect hardware, data, and actionable insights within a weekend.",
    description="Flash sensor firmware, stream metrics into Home Assistant, visualise consumption, and recommend behavioural/device changes.",
    market_alignment="Sustainability mandates are fuelling demand for engineers who can deliver end-to-end energy insights.",
    dataset=None,
    steps=[
        _step(
            "hardware",
            "Assemble energy sensor",
            "Configure ESP device with CT clamp or smart plug firmware and validate readings.",
            xp_reward=20,
        ),
        _step(
            "ingest",
            "Integrate with Home Assistant",
            "Set up MQTT bridge, create dashboards, and log history for time-of-day analysis.",
            xp_reward=20,
        ),
        _step(
            "insights",
            "Generate savings insights",
            "Identify peak usage devices and craft top three recommendations for reduction.",
            xp_reward=20,
        ),
    ],
    resources=[
        _resource(
            "Home Assistant Energy Setup",
            "https://www.home-assistant.io/docs/energy/",
            "article",
        ),
        _resource(
            "ESPHome Power Monitoring",
            "https://esphome.io/components/sensor/total_daily_energy.html",
            "article",
        ),
    ],
    github_template="https://github.com/esphome/esphome",
    expected_output="Home Assistant dashboard screenshots + energy savings action plan.",
)

_project(
    "IoT",
    "Autonomous Drone Swarm Simulator",
    difficulty="Advanced",
    time_commitment="long",
    estimated_time="4-5 weeks",
    tech_stack=["ROS2", "Gazebo", "PX4", "Python", "OpenCV"],
    tags=["robotics", "simulation", "computer-vision"],
    summary="Simulate a coordinated multi-drone mission with perception, collision avoidance, and mission planning.",
    why_it_matters="Robotics employers expect experience orchestrating autonomous fleets before granting hardware access.",
    description="Build a ROS2/Gazebo simulation that coordinates multiple drones to scan an area, avoid obstacles, and report findings.",
    market_alignment="Swarm coordination is a rising theme in logistics, defense, and industrial automation interviews.",
    dataset=None,
    steps=[
        _step(
            "sim",
            "Set up ROS2 + Gazebo environment",
            "Create world, spawn drones, and define mission scripts with PX4 offboard control.",
            xp_reward=60,
        ),
        _step(
            "perception",
            "Implement onboard perception",
            "Process camera feeds for obstacle detection and mapping.",
            xp_reward=60,
        ),
        _step(
            "coordination",
            "Coordinate swarm behaviour",
            "Develop formation control and task allocation to cover the mission area efficiently.",
            xp_reward=60,
        ),
        _step(
            "failsafe",
            "Add safety & recovery routines",
            "Simulate communication loss and collisions with graceful recovery strategies.",
            xp_reward=50,
        ),
        _step(
            "demo",
            "Deliver mission demo",
            "Record mission walkthrough and author technical design document.",
            xp_reward=40,
        ),
    ],
    resources=[
        _resource(
            "PX4 ROS2 Offboard Guide",
            "https://docs.px4.io/main/en/ros/ros2_offboard_control.html",
            "article",
        ),
        _resource(
            "ROS2 Navigation Stack",
            "https://navigation.ros.org/",
            "article",
        ),
    ],
    github_template="https://github.com/PX4/PX4-SITL_gazebo",
    expected_output="Simulation repo + mission video + technical architecture report.",
)


def list_projects(
    domain: str,
    difficulty: Optional[str] = None,
    time_commitment: Optional[str] = None,
) -> List[ProjectDefinition]:
    domain_key = _normalize_domain(domain)
    projects = PROJECT_CATALOG.get(domain_key, [])

    def _matches(project: ProjectDefinition) -> bool:
        diff_ok = not difficulty or project.difficulty.lower() == difficulty.lower()
        time_ok = not time_commitment or project.time_commitment.lower() == time_commitment.lower()
        return diff_ok and time_ok

    return [p for p in projects if _matches(p)]


def get_project(project_id: str) -> Optional[ProjectDefinition]:
    for projects in PROJECT_CATALOG.values():
        for project in projects:
            if project.id == project_id:
                return project
    return None


def all_projects_for_user(domain: str) -> List[ProjectDefinition]:
    return list_projects(domain)

