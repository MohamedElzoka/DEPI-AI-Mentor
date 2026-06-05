"""
Adaptive Learning Agent.

Analyzes student performance and decides whether to advance to the next week,
extend the current week with additional practice, or flag for human review.
"""

from typing import Any, Dict, List

from agents.base import BaseAgent
from memory.long_term import MemoryManager
from state import AgentState


SYSTEM_PROMPT = """You are the Adaptive Learning Agent for the DEPI AI Mentor platform.
Your role is to analyze student performance and adjust the learning path accordingly.

Decision rules:
- Score >= 80: Strong performance — advance normally.
- Score 60-79: Adequate performance — advance with a reinforcement tip.
- Score < 60: Weak performance — add supplementary material and one practice exercise before advancing.
- Three consecutive weak scores on the same topic: Flag for human review.

Always be supportive and frame setbacks as growth opportunities.
"""

PASS_THRESHOLD = 60.0


class AdaptiveLearningAgent(BaseAgent):
    """
    Reviews evaluation results and adapts the learning plan:
    - Advances the week counter if the student passed.
    - Extends with extra practice if the student failed.
    - Routes to Interview Agent when all weeks are completed.
    """

    def __init__(self):
        super().__init__("AdaptiveLearningAgent")

    def _generate_extra_practice(
        self, topic: str, level: str, improvements: List[str]
    ) -> str:
        """Generate a targeted mini-exercise based on identified weaknesses."""
        improvements_text = "\n".join(f"- {i}" for i in improvements)
        prompt = f"""
Topic: {topic}
Level: {level}
Student needs to improve:
{improvements_text}

Create one short, focused practice exercise (not a full assignment) that specifically
addresses these weaknesses. The exercise should take 20-30 minutes.
Format it clearly with instructions and what success looks like.
"""
        return self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=prompt)

    def _generate_reinforcement_tip(self, topic: str, improvements: List[str]) -> str:
        """Generate a concise tip to reinforce a concept the student partially understood."""
        improvements_text = ", ".join(improvements[:2])
        prompt = f"""
Topic: {topic}
Student's areas to improve: {improvements_text}

Write a concise (3-5 sentence) pro tip that reinforces the key concept the student
almost got right. Make it practical and memorable.
"""
        return self._call_llm(system_prompt=SYSTEM_PROMPT, user_message=prompt)

    def run(self, state: AgentState) -> AgentState:
        """Analyze evaluation result and update learning plan state."""
        evaluation = state.get("evaluation_result", {})
        learning_path = state.get("learning_path", [])
        current_week = state.get("current_week", 1)
        topic = state.get("current_topic", "")
        level = state.get("student_level", "Beginner")
        improvements = evaluation.get("improvements", [])
        final_score = evaluation.get("final_score", 0)
        messages = state.get("messages", [])

        total_weeks = len(learning_path)

        if final_score >= PASS_THRESHOLD:
            # Advance to the next week
            next_week = current_week + 1

            if next_week > total_weeks:
                # Learning path complete — route to interview
                state["next_agent"] = "interview"
                reply = (
                    "Congratulations! You have completed all weeks of your learning path.\n"
                    "You are now ready for your Mock Interview. Let's test your readiness!"
                )
            else:
                state["current_week"] = next_week
                next_topic = next(
                    (w["topic"] for w in learning_path if w["week"] == next_week), "Next Topic"
                )
                state["current_topic"] = next_topic
                state["next_agent"] = "content"

                if final_score >= 80:
                    tip = ""
                else:
                    tip = "\n\nPro Tip: " + self._generate_reinforcement_tip(topic, improvements)

                reply = (
                    f"Well done! You passed Week {current_week}. Moving to Week {next_week}: {next_topic}.{tip}\n\n"
                    "Loading your learning materials..."
                )
        else:
            # Stay on current week — provide extra practice
            extra_practice = self._generate_extra_practice(topic, level, improvements)
            state["next_agent"] = "assignment"  # Will generate a new assignment after practice

            reply = (
                f"You'll need a bit more practice on {topic} before moving on.\n\n"
                f"Here is a focused practice exercise:\n\n{extra_practice}\n\n"
                "Complete this exercise and then try the main assignment again."
            )

            # Record in memory
            student_id = state.get("student_id")
            if student_id:
                memory = MemoryManager(student_id)
                memory.record_weakness(
                    topic,
                    f"Required extra practice in Week {current_week}. Score: {final_score:.0f}/100.",
                )

        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        return state
