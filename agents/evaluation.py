"""
Evaluation Agent.

Receives student submissions (code or text answers), executes code when applicable,
and produces structured feedback with scores.
"""

import json
from typing import Any, Dict

from agents.base import BaseAgent
from memory.long_term import MemoryManager
from state import AgentState
from tools.tools import execute_python_code


SYSTEM_PROMPT = """You are the Evaluation Agent for the DEPI AI Mentor platform.
Your role is to fairly and constructively evaluate student submissions.

Evaluation dimensions:
1. Correctness (0-10): Does the code/answer produce the right result?
2. Code Quality (0-10): Is the code clean, readable, and well-structured?
3. Best Practices (0-10): Does the student follow language/domain best practices?

Guidelines:
- Be honest but encouraging in your feedback.
- Highlight what the student did well before pointing out mistakes.
- Provide specific, actionable improvement suggestions.
- For code, mention any edge cases not handled.
- Final score = (correctness * 0.5) + (quality * 0.25) + (best_practices * 0.25)
- Scale final score to 0-100.
"""

PASS_THRESHOLD = 60.0  # Minimum score to progress to the next week


class EvaluationAgent(BaseAgent):
    """
    Evaluates student submissions and determines if the student should
    advance to the next week or receive additional practice material.
    """

    def __init__(self):
        super().__init__("EvaluationAgent")

    def evaluate_submission(
        self,
        submission: str,
        assignment: Dict[str, Any],
        level: str,
        topic: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a student submission against the assignment requirements.

        Returns:
            {
                "correctness_score": 0-10,
                "quality_score": 0-10,
                "best_practices_score": 0-10,
                "final_score": 0-100,
                "passed": bool,
                "feedback": "...",
                "strengths": [...],
                "improvements": [...]
            }
        """
        # If the assignment is a code task, try to execute the code first
        execution_output = ""
        if assignment.get("type") == "code":
            execution_output = execute_python_code.invoke({"code": submission})

        prompt = f"""
Topic: {topic}
Level: {level}
Assignment Type: {assignment.get('type')}

Assignment Description:
{assignment.get('description', '')}

Evaluation Criteria:
{json.dumps(assignment.get('evaluation_criteria', {}), indent=2)}

Expected Output:
{assignment.get('expected_output', 'N/A')}

Student Submission:
```
{submission}
```

Code Execution Output (if applicable):
{execution_output if execution_output else 'N/A'}

Evaluate the submission and return a JSON object with exactly these keys:
{{
  "correctness_score": <int 0-10>,
  "quality_score": <int 0-10>,
  "best_practices_score": <int 0-10>,
  "final_score": <float 0-100>,
  "passed": <true if final_score >= {PASS_THRESHOLD}>,
  "feedback": "<2-4 sentences of overall feedback>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"]
}}

Be fair and constructive. For {level} students, adjust expectations accordingly.
"""
        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            json_mode=True,
        )
        return self._parse_json(response)

    def run(self, state: AgentState) -> AgentState:
        """Main agent entry point. Evaluates the submission stored in state."""
        submission = state.get("student_submission", "")
        assignment = state.get("current_assignment", {})
        level = state.get("student_level", "Beginner")
        topic = state.get("current_topic", "")
        student_id = state.get("student_id")

        if not submission:
            messages = state.get("messages", [])
            messages.append({
                "role": "assistant",
                "content": "Please submit your solution so I can evaluate it.",
            })
            state["messages"] = messages
            state["next_agent"] = "evaluation"
            return state

        result = self.evaluate_submission(submission, assignment, level, topic)
        state["evaluation_result"] = result

        # Update long-term memory based on results
        if student_id:
            memory = MemoryManager(student_id)
            score = result.get("final_score", 0)
            if score >= 80:
                memory.record_strength(topic, f"Scored {score:.0f}/100 on {topic} assignment")
            elif score < PASS_THRESHOLD:
                memory.record_weakness(
                    topic,
                    f"Scored {score:.0f}/100. Improvements: {'; '.join(result.get('improvements', [])[:2])}",
                    confidence=0.8,
                )

        # Build the feedback message
        strengths_text = "\n".join(f"  + {s}" for s in result.get("strengths", []))
        improvements_text = "\n".join(f"  - {i}" for i in result.get("improvements", []))
        passed = result.get("passed", False)

        score_label = f"{result.get('final_score', 0):.0f}/100"
        breakdown = (
            f"  Correctness: {result.get('correctness_score', 0)}/10\n"
            f"  Code Quality: {result.get('quality_score', 0)}/10\n"
            f"  Best Practices: {result.get('best_practices_score', 0)}/10"
        )

        if passed:
            next_action = "Great work! Moving on to the next topic."
            state["next_agent"] = "adaptive"
        else:
            next_action = (
                f"Your score is below the pass threshold ({PASS_THRESHOLD:.0f}/100). "
                "I'll provide additional practice material before we move on."
            )
            state["next_agent"] = "adaptive"

        reply = (
            f"Evaluation Results — {topic}\n"
            f"Score: {score_label}\n{breakdown}\n\n"
            f"Feedback:\n{result.get('feedback', '')}\n\n"
            f"What you did well:\n{strengths_text}\n\n"
            f"Areas to improve:\n{improvements_text}\n\n"
            f"{next_action}"
        )

        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        return state
