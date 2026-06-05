"""
Shared state schema for the LangGraph multi-agent workflow.
All agents read and write to this typed dictionary.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Student identity
    student_id: int
    student_name: str
    student_email: str
    track: str  # Data Analysis | AI Engineering | Data Science

    # Conversation history (short-term memory)
    messages: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]

    # Assessment
    assessment_scores: Dict[str, float]  # {"python": 0.8, "sql": 0.4, ...}
    student_level: str  # Beginner | Intermediate | Advanced

    # Learning path
    learning_path: List[Dict[str, Any]]  # list of week dicts
    current_week: int
    current_topic: str

    # Content retrieval
    retrieved_content: List[Dict[str, str]]  # [{"source": ..., "text": ...}]

    # Assignment
    current_assignment: Dict[str, Any]
    student_submission: str

    # Evaluation
    evaluation_result: Dict[str, Any]  # {score, feedback, correctness, quality, best_practices}

    # Interview
    interview_questions: List[Dict[str, str]]
    interview_answers: List[str]
    interview_result: Dict[str, Any]

    # Career advice
    career_advice: Dict[str, Any]

    # Routing / control
    next_agent: str  # which agent the supervisor routes to next
    action_required: str  # human_review | continue | extend_week | complete
    error: Optional[str]
