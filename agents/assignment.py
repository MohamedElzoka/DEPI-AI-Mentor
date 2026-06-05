"""
Assignment Generator Agent.

Creates personalized, topic-appropriate assignments for each student
based on their current week, level, and identified weaknesses.
"""

import json
from typing import Any, Dict

from agents.base import BaseAgent
from memory.long_term import MemoryManager
from state import AgentState


SYSTEM_PROMPT = """You are the Assignment Generator Agent for the DEPI AI Mentor platform.
Your role is to create meaningful, educational assignments for students.

Assignment types:
- code: Python coding exercise requiring a working script
- quiz: Multiple-choice or short-answer conceptual questions
- project: A mini-project combining multiple skills

Guidelines:
- Assignments must be appropriate for the student's level.
- Instructions must be clear and unambiguous.
- Code assignments must specify expected input/output.
- Projects should have a realistic real-world context.
- Always include evaluation criteria so the student knows what is expected.
- For code assignments, include a starter template when helpful.
"""


class AssignmentAgent(BaseAgent):
    """
    Generates a customized assignment for the student's current week topic.
    """

    def __init__(self):
        super().__init__("AssignmentAgent")

    def generate_assignment(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate a structured assignment for the current week.

        Returns:
            {
                "title": "...",
                "type": "code|quiz|project",
                "difficulty": "Beginner|Intermediate|Advanced",
                "description": "...",
                "tasks": [...],
                "evaluation_criteria": {...},
                "starter_code": "..."  # optional
            }
        """
        topic = state.get("current_topic", "Python Basics")
        level = state.get("student_level", "Beginner")
        track = state.get("track", "Data Analysis")
        week = state.get("current_week", 1)
        student_id = state.get("student_id")

        # Load weaknesses from long-term memory
        weaknesses = []
        if student_id:
            memory = MemoryManager(student_id)
            weaknesses = memory.get_weaknesses()

        weakness_note = (
            f"Note: The student has shown weakness in: {', '.join(weaknesses)}. "
            "Consider incorporating these areas if relevant."
            if weaknesses
            else ""
        )

        # Determine assignment type based on week progression
        assignment_type = "code" if week % 3 != 0 else "project"
        if topic.lower() in ["statistics", "statistics fundamentals", "descriptive statistics"]:
            assignment_type = "quiz"

        prompt = f"""
Track: {track}
Level: {level}
Week: {week}
Topic: {topic}
Assignment Type: {assignment_type}
{weakness_note}

Create a detailed assignment. Return a JSON object with exactly these keys:
{{
  "title": "<assignment title>",
  "type": "{assignment_type}",
  "difficulty": "{level}",
  "description": "<clear, detailed description of what the student must do>",
  "tasks": [
    "<task 1 description>",
    "<task 2 description>",
    "<task 3 description>"
  ],
  "evaluation_criteria": {{
    "correctness": "<what counts as correct>",
    "code_quality": "<style and readability expectations>",
    "best_practices": "<specific best practices to follow>"
  }},
  "starter_code": "<Python starter template if type is code, else empty string>",
  "expected_output": "<describe expected output/result>"
}}

Make the assignment practical and relevant to real-world {track} work.
"""
        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            json_mode=True,
        )
        return self._parse_json(response)

    def run(self, state: AgentState) -> AgentState:
        """Main agent entry point called by the supervisor."""
        assignment = self.generate_assignment(state)
        state["current_assignment"] = assignment

        # Format the assignment for display
        tasks_text = "\n".join(
            f"  {i+1}. {task}" for i, task in enumerate(assignment.get("tasks", []))
        )
        criteria = assignment.get("evaluation_criteria", {})
        criteria_text = "\n".join(
            f"  - {k.replace('_', ' ').title()}: {v}"
            for k, v in criteria.items()
        )
        starter = assignment.get("starter_code", "")
        starter_section = f"\nStarter Code:\n```python\n{starter}\n```\n" if starter else ""

        reply = (
            f"Assignment: {assignment.get('title', 'Week Assignment')}\n"
            f"Type: {assignment.get('type', '').upper()} | "
            f"Difficulty: {assignment.get('difficulty', '')}\n\n"
            f"{assignment.get('description', '')}\n\n"
            f"Tasks:\n{tasks_text}\n"
            f"{starter_section}\n"
            f"Evaluation Criteria:\n{criteria_text}\n\n"
            "Submit your solution when you're ready and I'll evaluate it."
        )

        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        state["next_agent"] = "evaluation"
        return state
