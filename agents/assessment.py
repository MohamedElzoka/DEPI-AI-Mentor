"""
Assessment Agent.

Conducts a multi-topic skill assessment for a new student and returns
structured scores and an overall level determination.
"""

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from state import AgentState


SYSTEM_PROMPT = """You are the Assessment Agent for the Digital Egypt Pioneers (DEPI) AI Mentor platform.
Your role is to evaluate a student's existing knowledge across the following topics:
- Python programming
- SQL databases
- Statistics and probability
- Excel
- Power BI

You ask targeted questions, evaluate the student's answers, and assign skill scores.

Rules:
- Ask one question at a time per topic.
- Questions should be concise and appropriate for the stated topic.
- After receiving an answer, evaluate it honestly.
- At the end, produce a structured JSON assessment report.
- Be encouraging but accurate in your evaluation.
- Level determination: Beginner (avg < 0.4), Intermediate (0.4-0.7), Advanced (> 0.7)
"""

# Pre-defined assessment questions per topic
ASSESSMENT_QUESTIONS: Dict[str, List[str]] = {
    "python": [
        "What is the difference between a list and a tuple in Python?",
        "Write a Python function that takes a list of numbers and returns the sum of even numbers.",
        "What does the `with` statement do in Python? Give a use case.",
    ],
    "sql": [
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "Write a SQL query to find the top 5 customers by total order amount.",
        "What is a database index and when would you use one?",
    ],
    "statistics": [
        "What is the difference between mean, median, and mode? When is each most useful?",
        "Explain what a p-value means in hypothesis testing.",
        "What is the Central Limit Theorem and why is it important?",
    ],
    "excel": [
        "What is a VLOOKUP function and what are its limitations?",
        "How would you create a pivot table? What is it used for?",
    ],
    "power_bi": [
        "What is the difference between a measure and a calculated column in Power BI?",
        "What does the CALCULATE function do in DAX?",
    ],
}


class AssessmentAgent(BaseAgent):
    """
    Evaluates a student's skill level through a structured Q&A session.
    Returns scores per topic and an overall level (Beginner/Intermediate/Advanced).
    """

    def __init__(self):
        super().__init__("AssessmentAgent")

    def get_next_question(self, state: AgentState) -> str:
        """
        Return the next assessment question based on progress tracked in messages.
        Returns an empty string when the assessment is complete.
        """
        answered_topics = self._get_answered_topics(state)
        for topic, questions in ASSESSMENT_QUESTIONS.items():
            if topic not in answered_topics:
                return f"[{topic.upper()}] {questions[0]}"
        return ""  # All topics assessed

    def _get_answered_topics(self, state: AgentState) -> List[str]:
        """Determine which topics have already been asked about."""
        answered = []
        messages = state.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                for topic in ASSESSMENT_QUESTIONS.keys():
                    if f"[{topic.upper()}]" in content and topic not in answered:
                        answered.append(topic)
        return answered

    def evaluate_responses(self, state: AgentState) -> Dict[str, Any]:
        """
        After collecting all answers, evaluate them and return structured scores.

        Returns:
            {
                "scores": {"python": 0.0-1.0, "sql": ..., ...},
                "overall_level": "Beginner"|"Intermediate"|"Advanced",
                "strengths": [...],
                "weaknesses": [...],
                "summary": "..."
            }
        """
        messages = state.get("messages", [])
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        track = state.get("track", "Data Analysis")

        prompt = f"""
Review the following assessment conversation and evaluate the student's knowledge.
Track: {track}

CONVERSATION:
{history_text}

Return a JSON object with exactly these keys:
{{
  "scores": {{
    "python": <float 0-1>,
    "sql": <float 0-1>,
    "statistics": <float 0-1>,
    "excel": <float 0-1>,
    "power_bi": <float 0-1>
  }},
  "overall_level": "<Beginner|Intermediate|Advanced>",
  "strengths": ["<topic>", ...],
  "weaknesses": ["<topic>", ...],
  "summary": "<2-3 sentences describing the student's profile>"
}}

Be strict and accurate. A student who cannot answer basic questions should score below 0.3.
"""
        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            json_mode=True,
        )
        return self._parse_json(response)

    def run(self, state: AgentState) -> AgentState:
        """
        Main agent entry point called by the supervisor graph.
        Either returns the next question or triggers evaluation when done.
        """
        messages = state.get("messages", [])
        answered_topics = self._get_answered_topics(state)
        all_topics = list(ASSESSMENT_QUESTIONS.keys())

        if len(answered_topics) >= len(all_topics):
            # All topics have been asked — evaluate now
            result = self.evaluate_responses(state)
            state["assessment_scores"] = result["scores"]
            state["student_level"] = result["overall_level"]
            state["next_agent"] = "learning_path"
            reply = (
                f"Assessment complete.\n"
                f"Overall Level: {result['overall_level']}\n"
                f"Strengths: {', '.join(result.get('strengths', []))}\n"
                f"Areas to Improve: {', '.join(result.get('weaknesses', []))}\n\n"
                f"{result.get('summary', '')}\n\n"
                "I will now generate your personalized learning path."
            )
        else:
            # Ask the next question
            next_q = self.get_next_question(state)
            reply = next_q
            state["next_agent"] = "assessment"  # Stay in assessment loop

        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        return state
