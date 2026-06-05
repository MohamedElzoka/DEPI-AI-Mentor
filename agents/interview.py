"""
Interview Agent.

Conducts mock technical and behavioral interviews aligned to the student's
track and learning path topics. Provides structured feedback afterward.
"""

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from state import AgentState


SYSTEM_PROMPT = """You are the Interview Agent for the DEPI AI Mentor platform.
You conduct realistic mock interviews to prepare students for job interviews.

Interview types:
- technical: Topic-specific technical questions
- behavioral: Situation-based questions using the STAR method
- mixed: A combination of both

Rules:
- Ask one question at a time and wait for the answer.
- Technical questions should test actual understanding, not memorization.
- Behavioral questions should relate to realistic scenarios.
- After all questions are answered, provide a comprehensive evaluation.
- Be professional but encouraging in your feedback.
"""

TECHNICAL_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "Data Analysis": [
        "Explain the difference between correlation and causation. Give a real-world example.",
        "How would you handle a dataset with 30% missing values? Walk me through your decision process.",
        "What is data normalization and when would you use it?",
        "Describe a situation where a simple average would be misleading.",
        "What is the difference between INNER JOIN and LEFT JOIN in SQL? When would you use each?",
    ],
    "AI Engineering": [
        "Explain the bias-variance tradeoff. How do you manage it in practice?",
        "What is overfitting and what techniques can you use to prevent it?",
        "Describe the attention mechanism in transformers in simple terms.",
        "What is RAG (Retrieval-Augmented Generation) and when would you use it over fine-tuning?",
        "How would you evaluate the quality of an LLM-generated response?",
    ],
    "Data Science": [
        "Walk me through how you would approach a new data science problem from scratch.",
        "What is the Central Limit Theorem and why does it matter for data science?",
        "Explain precision vs recall. When would you prioritize one over the other?",
        "What is feature engineering? Give an example from a real project.",
        "How do you select the right machine learning algorithm for a problem?",
    ],
}

BEHAVIORAL_QUESTIONS = [
    "Tell me about a time you had to work with messy or incomplete data. How did you handle it?",
    "Describe a project where you had to communicate technical findings to a non-technical audience.",
    "Tell me about a time you made a mistake in your analysis. How did you catch it and fix it?",
    "Describe a situation where you had to learn a new technology quickly. How did you approach it?",
    "Tell me about a challenging project. What was your role and what did you learn from it?",
]


class InterviewAgent(BaseAgent):
    """
    Conducts mock interviews and provides post-interview evaluation and feedback.
    """

    def __init__(self):
        super().__init__("InterviewAgent")

    def get_questions(self, track: str, session_type: str = "mixed") -> List[str]:
        """Select interview questions based on track and session type."""
        technical = TECHNICAL_QUESTION_TEMPLATES.get(
            track, TECHNICAL_QUESTION_TEMPLATES["Data Analysis"]
        )
        if session_type == "technical":
            return technical[:4]
        elif session_type == "behavioral":
            return BEHAVIORAL_QUESTIONS[:3]
        else:  # mixed
            return technical[:3] + BEHAVIORAL_QUESTIONS[:2]

    def evaluate_interview(
        self,
        questions: List[str],
        answers: List[str],
        track: str,
        level: str,
    ) -> Dict[str, Any]:
        """
        Evaluate all interview answers and return a structured report.

        Returns:
            {
                "overall_score": 0-100,
                "technical_score": 0-100,
                "communication_score": 0-100,
                "per_question_feedback": [...],
                "strengths": [...],
                "improvements": [...],
                "hiring_recommendation": "...",
                "summary": "..."
            }
        """
        qa_pairs = [
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(questions, answers))
        ]
        qa_text = "\n\n".join(qa_pairs)

        prompt = f"""
Track: {track}
Level: {level}

Interview Q&A:
{qa_text}

Evaluate this mock interview performance. Return a JSON object with exactly these keys:
{{
  "overall_score": <int 0-100>,
  "technical_score": <int 0-100>,
  "communication_score": <int 0-100>,
  "per_question_feedback": [
    {{"question": "<Q1>", "score": <0-10>, "feedback": "<brief feedback>"}},
    ...
  ],
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "hiring_recommendation": "<Strong Yes | Yes | Maybe | No>",
  "summary": "<3-4 sentences summarizing interview performance and readiness>"
}}

Be honest and constructive. {level} students should be evaluated relative to entry-level expectations.
"""
        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            json_mode=True,
        )
        return self._parse_json(response)

    def run(self, state: AgentState) -> AgentState:
        """Main agent entry point. Manages the interview flow."""
        track = state.get("track", "Data Analysis")
        level = state.get("student_level", "Beginner")
        messages = state.get("messages", [])
        interview_questions = state.get("interview_questions", [])
        interview_answers = state.get("interview_answers", [])

        if not interview_questions:
            # Start the interview: generate and present the first question
            questions = self.get_questions(track, session_type="mixed")
            state["interview_questions"] = questions
            state["interview_answers"] = []
            state["next_agent"] = "interview"

            reply = (
                "Welcome to your Mock Interview! This will consist of technical and behavioral questions "
                f"aligned to the {track} track.\n\n"
                "Take your time to think before answering. Ready? Let's begin.\n\n"
                f"Question 1: {questions[0]}"
            )

        elif len(interview_answers) < len(interview_questions):
            # Record the latest answer (passed as the last user message)
            last_user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            interview_answers.append(last_user_msg)
            state["interview_answers"] = interview_answers

            if len(interview_answers) < len(interview_questions):
                # Ask the next question
                next_q = interview_questions[len(interview_answers)]
                state["next_agent"] = "interview"
                reply = f"Thank you. Question {len(interview_answers) + 1}: {next_q}"
            else:
                # All questions answered — evaluate
                result = self.evaluate_interview(
                    interview_questions, interview_answers, track, level
                )
                state["interview_result"] = result
                state["next_agent"] = "career"

                per_q_text = "\n".join(
                    f"  Q{i+1}: {item.get('feedback', '')} (Score: {item.get('score', 0)}/10)"
                    for i, item in enumerate(result.get("per_question_feedback", []))
                )
                strengths_text = "\n".join(f"  + {s}" for s in result.get("strengths", []))
                improvements_text = "\n".join(f"  - {i}" for i in result.get("improvements", []))

                reply = (
                    f"Interview Complete!\n\n"
                    f"Overall Score: {result.get('overall_score', 0)}/100\n"
                    f"Technical: {result.get('technical_score', 0)}/100 | "
                    f"Communication: {result.get('communication_score', 0)}/100\n"
                    f"Recommendation: {result.get('hiring_recommendation', '')}\n\n"
                    f"Per-Question Feedback:\n{per_q_text}\n\n"
                    f"Your Strengths:\n{strengths_text}\n\n"
                    f"Areas to Improve:\n{improvements_text}\n\n"
                    f"{result.get('summary', '')}\n\n"
                    "I will now generate your Career Advice report."
                )
        else:
            # Interview already completed — route to career
            state["next_agent"] = "career"
            reply = "Your interview has already been completed. Moving to Career Advice."

        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        return state
