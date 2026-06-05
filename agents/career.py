"""
Career Advisor Agent.

Provides personalized career guidance after the student completes their
learning path and mock interview, including job recommendations, skills gap
analysis, and CV/LinkedIn improvement tips.
"""

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from memory.long_term import MemoryManager
from state import AgentState


SYSTEM_PROMPT = """You are the Career Advisor Agent for the DEPI AI Mentor platform.
Your role is to provide personalized, actionable career guidance to students who have
completed their DEPI learning track.

Your advice covers:
1. Job role recommendations based on skills and track
2. Skills gap analysis (what to learn next)
3. CV improvement tips
4. LinkedIn profile optimization
5. Job search strategy

Guidelines:
- Be specific, not generic. Tailor all advice to the student's track and performance.
- Mention Egyptian and regional job market considerations where relevant.
- Include international opportunities when appropriate.
- Be encouraging while being honest about areas that need improvement.
"""

JOB_ROLES_BY_TRACK: Dict[str, List[Dict[str, str]]] = {
    "Data Analysis": [
        {"title": "Junior Data Analyst", "level": "Entry", "skills_needed": "Excel, SQL, Power BI, Python basics"},
        {"title": "Business Intelligence Analyst", "level": "Entry-Mid", "skills_needed": "SQL, Power BI/Tableau, data modeling"},
        {"title": "Reporting Analyst", "level": "Entry", "skills_needed": "Excel, Power BI, SQL, communication"},
        {"title": "Data Analyst", "level": "Mid", "skills_needed": "Python, SQL, statistics, visualization"},
    ],
    "AI Engineering": [
        {"title": "Junior ML Engineer", "level": "Entry", "skills_needed": "Python, scikit-learn, SQL, Git"},
        {"title": "AI Application Developer", "level": "Entry-Mid", "skills_needed": "Python, LLM APIs, FastAPI, prompt engineering"},
        {"title": "NLP Engineer", "level": "Mid", "skills_needed": "Python, transformers, LangChain, NLP pipelines"},
        {"title": "Generative AI Engineer", "level": "Mid", "skills_needed": "LLMs, RAG, LangGraph, vector databases"},
    ],
    "Data Science": [
        {"title": "Junior Data Scientist", "level": "Entry", "skills_needed": "Python, ML, statistics, visualization"},
        {"title": "Machine Learning Engineer", "level": "Mid", "skills_needed": "Python, scikit-learn, deep learning, MLOps"},
        {"title": "Research Scientist", "level": "Mid-Senior", "skills_needed": "Advanced ML, publications, Python, statistics"},
        {"title": "Applied Scientist", "level": "Mid", "skills_needed": "ML, experimentation, Python, business understanding"},
    ],
}

SKILLS_GAP_BY_TRACK: Dict[str, List[str]] = {
    "Data Analysis": [
        "Advanced SQL (window functions, CTEs)",
        "Python automation for reports",
        "Statistical hypothesis testing",
        "Data storytelling and presentation",
        "Google Analytics or Adobe Analytics",
    ],
    "AI Engineering": [
        "MLOps (MLflow, Docker, Kubernetes)",
        "Cloud AI services (AWS SageMaker, Azure ML)",
        "Advanced prompt engineering and evaluation",
        "System design for AI applications",
        "Open-source contribution",
    ],
    "Data Science": [
        "Advanced feature engineering",
        "Experiment design and A/B testing",
        "Time series forecasting",
        "Deep learning (TensorFlow/PyTorch)",
        "Academic paper reading and implementation",
    ],
}


class CareerAdvisorAgent(BaseAgent):
    """
    Generates a comprehensive career development report for the student.
    """

    def __init__(self):
        super().__init__("CareerAdvisorAgent")

    def generate_career_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Produce a personalized career development report.

        Returns:
            {
                "recommended_roles": [...],
                "skills_gap": [...],
                "cv_tips": [...],
                "linkedin_tips": [...],
                "next_steps": [...],
                "summary": "..."
            }
        """
        track = state.get("track", "Data Analysis")
        level = state.get("student_level", "Beginner")
        name = state.get("student_name", "Student")
        student_id = state.get("student_id")

        # Load memory for personalized advice
        strengths = []
        weaknesses = []
        if student_id:
            memory = MemoryManager(student_id)
            strengths = memory.get_strengths()
            weaknesses = memory.get_weaknesses()

        interview_result = state.get("interview_result", {})
        interview_score = interview_result.get("overall_score", 0)

        recommended_roles = JOB_ROLES_BY_TRACK.get(track, JOB_ROLES_BY_TRACK["Data Analysis"])
        skills_gap = SKILLS_GAP_BY_TRACK.get(track, [])

        prompt = f"""
Student: {name}
Track: {track}
Level: {level}
Strengths (from memory): {', '.join(strengths) if strengths else 'Not yet recorded'}
Weaknesses (from memory): {', '.join(weaknesses) if weaknesses else 'None identified'}
Mock Interview Score: {interview_score}/100

Recommended job roles for this track:
{json.dumps(recommended_roles, indent=2)}

Potential skills gap topics:
{json.dumps(skills_gap, indent=2)}

Generate a personalized career development report. Return a JSON object with these keys:
{{
  "recommended_roles": [
    {{"title": "...", "fit_score": "<High|Medium|Low>", "reason": "<why this role fits>"}}
  ],
  "skills_gap": ["<skill 1>", "<skill 2>", "<skill 3>"],
  "cv_tips": [
    "<specific CV improvement tip 1>",
    "<specific CV improvement tip 2>",
    "<specific CV improvement tip 3>"
  ],
  "linkedin_tips": [
    "<LinkedIn optimization tip 1>",
    "<LinkedIn optimization tip 2>",
    "<LinkedIn optimization tip 3>"
  ],
  "next_steps": [
    "<actionable next step 1>",
    "<actionable next step 2>",
    "<actionable next step 3>"
  ],
  "certifications": [
    {{"name": "<cert name>", "provider": "<provider>", "priority": "<High|Medium>"}}
  ],
  "summary": "<3-4 sentences of personalized career summary>"
}}

Make the advice specific, actionable, and relevant to the Egyptian/MENA job market.
"""
        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            json_mode=True,
        )
        return self._parse_json(response)

    def run(self, state: AgentState) -> AgentState:
        """Main agent entry point. Generates and presents the career report."""
        report = self.generate_career_report(state)
        state["career_advice"] = report

        # Format the career report for display
        name = state.get("student_name", "Student")
        track = state.get("track", "Data Analysis")

        roles_text = "\n".join(
            f"  - {r['title']} ({r['fit_score']} Fit): {r.get('reason', '')}"
            for r in report.get("recommended_roles", [])
        )
        gap_text = "\n".join(f"  - {g}" for g in report.get("skills_gap", []))
        cv_text = "\n".join(f"  - {t}" for t in report.get("cv_tips", []))
        linkedin_text = "\n".join(f"  - {t}" for t in report.get("linkedin_tips", []))
        steps_text = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(report.get("next_steps", []))
        )
        certs_text = "\n".join(
            f"  - {c['name']} by {c['provider']} (Priority: {c['priority']})"
            for c in report.get("certifications", [])
        )

        reply = (
            f"Career Development Report — {name}\n"
            f"Track: {track}\n\n"
            f"Recommended Job Roles:\n{roles_text}\n\n"
            f"Skills Gap (What to Learn Next):\n{gap_text}\n\n"
            f"CV Improvement Tips:\n{cv_text}\n\n"
            f"LinkedIn Optimization:\n{linkedin_text}\n\n"
            f"Recommended Certifications:\n{certs_text}\n\n"
            f"Your Next Steps:\n{steps_text}\n\n"
            f"Summary:\n{report.get('summary', '')}\n\n"
            "Congratulations on completing your DEPI learning journey! "
            "You are now equipped to pursue exciting opportunities in the Egyptian digital economy. "
            "Best of luck!"
        )

        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": reply})
        state["messages"] = messages
        state["next_agent"] = "complete"
        state["action_required"] = "complete"
        return state
