import asyncio
from typing import Any, Dict, List

from app.db import get_mongo_db

TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "ai-ml-track-1-task-1",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "title": "Compute Feature Mean",
        "prompt": "Given a list of numeric feature values, return the arithmetic mean rounded to two decimals.",
        "language": "python",
        "entry_point": "feature_mean",
        "starter_code": "def feature_mean(values: list[float]) -> float:\n    # TODO: compute mean rounded to two decimals\n    pass\n",
        "solution_code": "def feature_mean(values: list[float]) -> float:\n    if not values:\n        return 0.0\n    return round(sum(values) / len(values), 2)\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 1,
        "estimate_minutes": 10,
        "skill_tag": "Python Fundamentals",
        "xp": 40,
        "hints": [
            "Use sum(values) and len(values) to compute the average.",
            "Remember to round the result to two decimal places with round(value, 2).",
        ],
        "sample_cases": [
            {"input": [1.0, 2.0, 3.0], "output": 2.0},
            {"input": [4.5, 5.5], "output": 5.0},
        ],
        "tests": [
            {
                "name": "Three values",
                "input": [[1.0, 2.0, 3.0]],
                "expected": 2.0,
                "hint": "Divide sum by count.",
            },
            {
                "name": "Empty list",
                "input": [[]],
                "expected": 0.0,
                "hint": "Return 0 when the list is empty.",
            },
            {
                "name": "Rounded mean",
                "input": [[0.2, 0.2, 0.2]],
                "expected": 0.2,
            },
        ],
    },
    {
        "task_id": "ai-ml-track-1-task-2",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "title": "Standardize Feature Values",
        "prompt": "Given a list of numeric values, return their z-score standardisation rounded to two decimals.",
        "language": "python",
        "entry_point": "standardize",
        "starter_code": "def standardize(values: list[float]) -> list[float]:\n    # TODO: compute z-score for each value\n    pass\n",
        "solution_code": "def standardize(values: list[float]) -> list[float]:\n    if not values:\n        return []\n    mean = sum(values) / len(values)\n    variance = sum((v - mean) ** 2 for v in values) / len(values)\n    std = variance ** 0.5\n    if std == 0:\n        return [0.0 for _ in values]\n    return [round((v - mean) / std, 2) for v in values]\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 2,
        "estimate_minutes": 20,
        "skill_tag": "Data Preparation",
        "xp": 60,
        "hints": [
            "Compute the mean first, then the standard deviation.",
            "Handle the case where standard deviation is zero.",
        ],
        "sample_cases": [
            {"input": [1, 2, 3], "output": [-1.22, 0.0, 1.22]},
        ],
        "tests": [
            {
                "name": "Simple sequence",
                "input": [[1, 2, 3]],
                "expected": [-1.22, 0.0, 1.22],
                "hint": "Use population standard deviation.",
            },
            {
                "name": "Constant sequence",
                "input": [[5, 5, 5]],
                "expected": [0.0, 0.0, 0.0],
            },
        ],
    },
    {
        "task_id": "ai-ml-track-2-task-1",
        "track_id": "ai-ml-track-2",
        "domain": "ai_ml",
        "title": "Fill Missing Values",
        "prompt": "Replace None values in a dataset with a provided fallback value.",
        "language": "python",
        "entry_point": "fill_missing",
        "starter_code": "def fill_missing(values: list[float | None], fill_value: float) -> list[float]:\n    # TODO: replace None with fill_value\n    pass\n",
        "solution_code": "def fill_missing(values: list[float | None], fill_value: float) -> list[float]:\n    return [fill_value if v is None else v for v in values]\n",
        "difficulty": "Intermediate",
        "category": "Intermediate",
        "order": 1,
        "estimate_minutes": 12,
        "skill_tag": "Data Cleaning",
        "xp": 50,
        "hints": [
            "Iterate through the list and replace None with the fill value.",
        ],
        "sample_cases": [
            {"input": {"values": [1, None, 3], "fill_value": 0}, "output": [1, 0, 3]},
        ],
        "tests": [
            {
                "name": "Mix of numbers and None",
                "input": {"values": [1, None, 3], "fill_value": 0},
                "expected": [1, 0, 3],
            },
            {
                "name": "All None",
                "input": {"values": [None, None], "fill_value": 7.5},
                "expected": [7.5, 7.5],
            },
        ],
    },
    {
        "task_id": "ai-ml-track-1-task-3",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "title": "Filter Positive Numbers",
        "prompt": "Return only positive numbers from the given list.",
        "language": "python",
        "entry_point": "filter_positive",
        "starter_code": "def filter_positive(values: list[int]) -> list[int]:\n    # TODO: keep only numbers greater than zero\n    pass\n",
        "solution_code": "def filter_positive(values: list[int]) -> list[int]:\n    return [v for v in values if v > 0]\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 3,
        "estimate_minutes": 8,
        "skill_tag": "List Comprehension",
        "xp": 35,
        "hints": [
            "Use a list comprehension to filter elements.",
        ],
        "tests": [
            {"name": "Mixed values", "input": [[-1, 0, 5, 2]], "expected": [5, 2]},
            {"name": "All negative", "input": [[-3, -4]], "expected": []},
        ],
    },
    {
        "task_id": "ai-ml-track-1-task-4",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "title": "Flatten Nested Lists",
        "prompt": "Flatten a single-level nested list (list of lists) into a single list.",
        "language": "python",
        "entry_point": "flatten",
        "starter_code": "def flatten(values: list[list[int]]) -> list[int]:\n    # TODO: flatten nested lists into one list\n    pass\n",
        "solution_code": "def flatten(values: list[list[int]]) -> list[int]:\n    result = []\n    for sub in values:\n        result.extend(sub)\n    return result\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 4,
        "estimate_minutes": 15,
        "skill_tag": "Python Lists",
        "xp": 50,
        "hints": [
            "Iterate through each sub-list and extend the result.",
        ],
        "tests": [
            {"name": "Two lists", "input": [[[1, 2], [3, 4]]], "expected": [1, 2, 3, 4]},
            {"name": "Empty inner list", "input": [[[1], [], [2, 3]]], "expected": [1, 2, 3]},
        ],
    },
    {
        "task_id": "ds-track-1-task-1",
        "track_id": "ds-track-1",
        "domain": "data_science",
        "title": "Count Orders by Status",
        "prompt": "Given an orders table, write a SQL query that counts orders grouped by status.",
        "language": "sql",
        "entry_point": "SELECT",
        "starter_code": "SELECT status, COUNT(*) AS total\nFROM orders\n-- TODO: group by status\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 1,
        "estimate_minutes": 12,
        "skill_tag": "SQL Aggregations",
        "xp": 45,
        "hints": [
            "Use GROUP BY with the status column.",
        ],
        "tests": [],
    },
    {
        "task_id": "ds-track-1-task-2",
        "track_id": "ds-track-1",
        "domain": "data_science",
        "title": "Calculate Rolling 7-day Average",
        "prompt": "Write Python code that calculates a rolling 7-day average of daily signups.",
        "language": "python",
        "entry_point": "rolling_average",
        "starter_code": "import pandas as pd\n\ndef rolling_average(df: pd.DataFrame) -> pd.Series:\n    # df has columns ['date', 'signups']\n    # TODO: compute rolling mean over window=7 ordered by date\n    pass\n",
        "solution_code": "import pandas as pd\n\ndef rolling_average(df: pd.DataFrame) -> pd.Series:\n    ordered = df.sort_values('date')\n    return ordered['signups'].rolling(window=7, min_periods=1).mean()\n",
        "difficulty": "Intermediate",
        "category": "Intermediate",
        "order": 2,
        "estimate_minutes": 20,
        "skill_tag": "Pandas",
        "xp": 60,
        "hints": [
            "Sort by date before applying rolling.",
            "Use min_periods=1 to avoid NaNs at the start.",
        ],
        "tests": [],
    },
    {
        "task_id": "web-track-1-task-1",
        "track_id": "web-track-1",
        "domain": "web_development",
        "title": "Debounce Function",
        "prompt": "Implement a JavaScript debounce helper that delays function execution.",
        "language": "javascript",
        "entry_point": "debounce",
        "starter_code": "export function debounce(fn, delay) {\n  // TODO: return a debounced version of fn\n}\n",
        "solution_code": "export function debounce(fn, delay) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n}\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 1,
        "estimate_minutes": 15,
        "skill_tag": "JavaScript Utilities",
        "xp": 45,
        "hints": [
            "Store the timer id outside the returned function.",
        ],
        "tests": [],
    },
    {
        "task_id": "web-track-1-task-2",
        "track_id": "web-track-1",
        "domain": "web_development",
        "title": "Fetch JSON Helper",
        "prompt": "Write an async helper that fetches JSON with basic error handling.",
        "language": "javascript",
        "entry_point": "fetchJson",
        "starter_code": "export async function fetchJson(url) {\n  // TODO: use fetch and return parsed JSON\n}\n",
        "solution_code": "export async function fetchJson(url) {\n  const res = await fetch(url);\n  if (!res.ok) {\n    throw new Error(`Request failed with ${res.status}`);\n  }\n  return res.json();\n}\n",
        "difficulty": "Beginner",
        "category": "Beginner",
        "order": 2,
        "estimate_minutes": 10,
        "skill_tag": "APIs",
        "xp": 40,
        "hints": [
            "Throw an error when response.ok is false.",
        ],
        "tests": [],
    },
]


async def seed() -> None:
    db = get_mongo_db()

    for task in TASKS:
        await db.coding_tasks.update_one(
            {"task_id": task["task_id"]},
            {"$set": task},
            upsert=True,
        )

    print(f"Seeded {len(TASKS)} coding tasks.")


if __name__ == "__main__":
    asyncio.run(seed())
import asyncio
from typing import Any, Dict, List

from app.db import get_mongo_db


TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "ai-ml-track-1-task-1",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "order": 1,
        "category": "Beginner",
        "title": "Reverse a List",
        "difficulty": "Beginner",
        "estimate_minutes": 10,
        "skill_tag": "Python Basics",
        "language": "python",
        "entrypoint": "reverse_list",
        "starter_code": "def reverse_list(items: list[int]) -> list[int]:\n    # TODO: return a new list in reverse order\n    pass\n",
        "description": "Return a new list with the items of `items` reversed. Do not mutate the incoming list.",
        "tests": [
            {"name": "simple", "input": {"args": [[1, 2, 3]]}, "expected": [3, 2, 1]},
            {"name": "single", "input": {"args": [[42]]}, "expected": [42]},
            {"name": "empty", "input": {"args": [[]]}, "expected": []},
        ],
        "xp": 40,
        "hints": [
            "Slicing with a negative step can reverse a list quickly.",
            "Alternatively, build a new list by iterating from the end to the start.",
        ],
    },
    {
        "task_id": "ai-ml-track-1-task-2",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "order": 2,
        "category": "Beginner",
        "title": "Compute Mean",
        "difficulty": "Beginner",
        "estimate_minutes": 12,
        "skill_tag": "Math Utilities",
        "language": "python",
        "entrypoint": "mean",
        "starter_code": "def mean(values: list[float]) -> float:\n    # TODO: compute the mean of values\n    pass\n",
        "description": "Return the arithmetic mean of `values`. Raise ValueError when the list is empty.",
        "tests": [
            {"name": "integers", "input": {"args": [[2, 4, 6, 8]]}, "expected": 5.0},
            {"name": "floats", "input": {"args": [[1.5, 2.5, 3.5]]}, "expected": 2.5},
            {"name": "empty", "input": {"args": [[]]}, "expected": "ValueError"},
        ],
        "xp": 45,
        "hints": [
            "Use sum(values) and len(values).",
            "Defend against division by zero by checking the list length first.",
        ],
    },
    {
        "task_id": "ai-ml-track-1-task-3",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "order": 3,
        "category": "Beginner",
        "title": "Filter Positive Numbers",
        "difficulty": "Beginner",
        "estimate_minutes": 8,
        "skill_tag": "List Comprehension",
        "language": "python",
        "entrypoint": "filter_positive",
        "starter_code": "def filter_positive(values: list[int]) -> list[int]:\n    # TODO: return only positive values\n    pass\n",
        "description": "Return a list containing only values greater than zero.",
        "tests": [
            {"name": "mixed", "input": {"args": [[-3, -1, 0, 4, 7]]}, "expected": [4, 7]},
            {"name": "negatives", "input": {"args": [[-5, -2]]}, "expected": []},
        ],
        "xp": 35,
    },
    {
        "task_id": "ai-ml-track-1-task-4",
        "track_id": "ai-ml-track-1",
        "domain": "ai_ml",
        "order": 4,
        "category": "Beginner",
        "title": "Flatten Nested Lists",
        "difficulty": "Beginner",
        "estimate_minutes": 15,
        "skill_tag": "Python Lists",
        "language": "python",
        "entrypoint": "flatten",
        "starter_code": "def flatten(nested: list[list[int]]) -> list[int]:\n    # TODO: flatten nested lists into a single list\n    pass\n",
        "description": "Flatten a list of lists into a single list while preserving order.",
        "tests": [
            {"name": "basic", "input": {"args": [[[1, 2], [3], [4, 5]]]}, "expected": [1, 2, 3, 4, 5]},
            {"name": "empty-inner", "input": {"args": [[[1], [], [2]]]}, "expected": [1, 2]},
        ],
        "xp": 50,
    },
    {
        "task_id": "ai-ml-track-2-task-1",
        "track_id": "ai-ml-track-2",
        "domain": "ai_ml",
        "order": 1,
        "category": "Intermediate",
        "title": "Standardize Feature",
        "difficulty": "Intermediate",
        "estimate_minutes": 20,
        "skill_tag": "Data Cleaning",
        "language": "python",
        "entrypoint": "standardize",
        "starter_code": "def standardize(values: list[float]) -> list[float]:\n    # TODO: return standardized values\n    pass\n",
        "description": "Return z-score standardized values (mean 0, standard deviation 1). When variance is zero, return zeros.",
        "tests": [
            {"name": "variance", "input": {"args": [[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]]}, "expected": [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]},
            {"name": "constant", "input": {"args": [[3.0, 3.0, 3.0]]}, "expected": [0.0, 0.0, 0.0]},
        ],
        "xp": 70,
    },
    {
        "task_id": "ai-ml-track-2-task-2",
        "track_id": "ai-ml-track-2",
        "domain": "ai_ml",
        "order": 2,
        "category": "Intermediate",
        "title": "Fill Missing with Mean",
        "difficulty": "Intermediate",
        "estimate_minutes": 18,
        "skill_tag": "Data Cleaning",
        "language": "python",
        "entrypoint": "fill_missing",
        "starter_code": "def fill_missing(values: list[float | None]) -> list[float]:\n    # TODO: impute None values with the mean\n    pass\n",
        "description": "Replace None values with the mean of existing numbers. Round to two decimal places.",
        "tests": [
            {"name": "basic", "input": {"args": [[1.0, None, 3.0]]}, "expected": [1.0, 2.0, 3.0]},
            {"name": "no-missing", "input": {"args": [[2.0, 4.0]]}, "expected": [2.0, 4.0]},
        ],
        "xp": 65,
    },
    {
        "task_id": "ai-ml-track-2-task-3",
        "track_id": "ai-ml-track-2",
        "domain": "ai_ml",
        "order": 3,
        "category": "Intermediate",
        "title": "Select High Variance Features",
        "difficulty": "Intermediate",
        "estimate_minutes": 22,
        "skill_tag": "Feature Engineering",
        "language": "python",
        "entrypoint": "select_high_variance",
        "starter_code": "def select_high_variance(columns: dict[str, list[float]], threshold: float) -> list[str]:\n    # TODO: return feature names with variance above threshold\n    pass\n",
        "description": "Return feature names whose variance exceeds the provided threshold.",
        "tests": [
            {
                "name": "variance-filter",
                "input": {"args": [{"a": [1, 1, 1], "b": [1, 2, 3], "c": [5, 7, 9]}, 2.0]},
                "expected": ["b", "c"],
            }
        ],
        "xp": 75,
    },
    {
        "task_id": "ai-ml-track-2-task-4",
        "track_id": "ai-ml-track-2",
        "domain": "ai_ml",
        "order": 4,
        "category": "Intermediate",
        "title": "Summarize Features",
        "difficulty": "Intermediate",
        "estimate_minutes": 20,
        "skill_tag": "Data Profiling",
        "language": "python",
        "entrypoint": "summarize_features",
        "starter_code": "def summarize_features(columns: dict[str, list[float]]) -> dict[str, dict[str, float]]:\n    # TODO: compute min, max, and mean per feature\n    pass\n",
        "description": "Return a dictionary with summary statistics (min, max, mean) for each feature.",
        "tests": [
            {
                "name": "summary",
                "input": {"args": [{"price": [10, 12, 14], "units": [3, 3, 3]}]},
                "expected": {
                    "price": {"min": 10.0, "max": 14.0, "mean": 12.0},
                    "units": {"min": 3.0, "max": 3.0, "mean": 3.0},
                },
            }
        ],
        "xp": 75,
    },
]


async def seed() -> None:
    db = get_mongo_db()

    for task in TASKS:
        await db.coding_tasks.update_one(
            {"task_id": task["task_id"]},
            {"$set": task},
            upsert=True,
        )

    print(f"Seeded {len(TASKS)} coding tasks.")


if __name__ == "__main__":
    asyncio.run(seed())

