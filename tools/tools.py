"""
Tool definitions for the DEPI AI Mentor agents.
Tools are wrapped as LangChain-compatible callables and registered for function calling.
"""

import ast
import io
import json
import math
import sys
import traceback
from contextlib import redirect_stdout
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from langchain.tools import tool


# ---------------------------------------------------------------------------
# Code Execution Tool
# ---------------------------------------------------------------------------

@tool
def execute_python_code(code: str) -> str:
    """
    Execute a Python code snippet in a sandboxed environment and return stdout output.
    Use this to test student code submissions for correctness.

    Args:
        code: Valid Python source code as a string.

    Returns:
        Captured stdout output, or an error message if execution fails.
    """
    stdout_capture = io.StringIO()
    try:
        # Basic safety: block dangerous imports
        forbidden = ["os.system", "subprocess", "shutil.rmtree", "__import__('os')"]
        for term in forbidden:
            if term in code:
                return f"Security Error: use of '{term}' is not permitted."

        # Parse AST to detect syntax errors early
        ast.parse(code)

        namespace: Dict[str, Any] = {}
        with redirect_stdout(stdout_capture):
            exec(code, namespace)  # noqa: S102

        output = stdout_capture.getvalue()
        return output if output else "(No output produced)"
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    except Exception:
        return f"RuntimeError:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports standard arithmetic, math functions (sqrt, log, sin, cos, etc.),
    and statistical helpers (mean, variance).

    Args:
        expression: A Python-compatible mathematical expression string.

    Returns:
        The computed result as a string, or an error message.
    """
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
        "mean": lambda lst: sum(lst) / len(lst),
        "variance": lambda lst: sum((x - sum(lst) / len(lst)) ** 2 for x in lst) / len(lst),
    }
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Calculation Error: {exc}"


# ---------------------------------------------------------------------------
# Study Scheduler Tool
# ---------------------------------------------------------------------------

@tool
def generate_study_schedule(
    track: str,
    level: str,
    total_weeks: int,
    start_date: Optional[str] = None,
) -> str:
    """
    Generate a weekly study schedule with start and end dates for each week.

    Args:
        track: The learning track (e.g., 'Data Analysis', 'AI Engineering').
        level: Student level — 'Beginner', 'Intermediate', or 'Advanced'.
        total_weeks: Total number of weeks in the learning path.
        start_date: ISO date string (YYYY-MM-DD) for week 1 start. Defaults to today.

    Returns:
        A JSON string with a list of schedule entries.
    """
    today = date.today() if not start_date else date.fromisoformat(start_date)
    schedule = []
    for week in range(1, total_weeks + 1):
        week_start = today + timedelta(weeks=week - 1)
        week_end = week_start + timedelta(days=6)
        schedule.append(
            {
                "week": week,
                "start_date": week_start.isoformat(),
                "end_date": week_end.isoformat(),
                "study_days": 5,
                "hours_per_day": 2 if level == "Beginner" else 3,
            }
        )
    return json.dumps(schedule, indent=2)


# ---------------------------------------------------------------------------
# Web Search Tool (stub — replace with real search API if available)
# ---------------------------------------------------------------------------

@tool
def search_learning_resources(query: str, resource_type: str = "article") -> str:
    """
    Search for learning resources related to a topic.
    Returns a list of recommended resources (simulated in this demo).

    Args:
        query: The topic or concept to search for.
        resource_type: Type of resource — 'article', 'video', 'course', or 'documentation'.

    Returns:
        A JSON string containing a list of recommended resource titles and URLs.
    """
    # In production, integrate with a real search API (Tavily, SerpAPI, etc.)
    resources = {
        "python": [
            {"title": "Python Official Documentation", "url": "https://docs.python.org/3/"},
            {"title": "Real Python Tutorials", "url": "https://realpython.com/"},
        ],
        "pandas": [
            {"title": "Pandas User Guide", "url": "https://pandas.pydata.org/docs/user_guide/"},
            {"title": "Kaggle Pandas Course", "url": "https://www.kaggle.com/learn/pandas"},
        ],
        "sql": [
            {"title": "SQLZoo Interactive Tutorials", "url": "https://sqlzoo.net/"},
            {"title": "Mode SQL Tutorial", "url": "https://mode.com/sql-tutorial/"},
        ],
        "power bi": [
            {
                "title": "Microsoft Power BI Documentation",
                "url": "https://learn.microsoft.com/en-us/power-bi/",
            },
            {"title": "Power BI Guided Learning", "url": "https://learn.microsoft.com/en-us/power-bi/guided-learning/"},
        ],
        "machine learning": [
            {"title": "Scikit-learn Documentation", "url": "https://scikit-learn.org/stable/"},
            {"title": "Kaggle ML Course", "url": "https://www.kaggle.com/learn/intro-to-machine-learning"},
        ],
    }

    query_lower = query.lower()
    matched: List[Dict] = []
    for key, items in resources.items():
        if key in query_lower:
            matched.extend(items)

    if not matched:
        matched = [
            {
                "title": f"Google Search: {query}",
                "url": f"https://www.google.com/search?q={query.replace(' ', '+')}+tutorial",
            }
        ]

    return json.dumps(matched, indent=2)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ALL_TOOLS = [execute_python_code, calculate, generate_study_schedule, search_learning_resources]
