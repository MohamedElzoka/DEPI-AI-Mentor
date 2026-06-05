"""
Learning Path Generator Agent.

Builds a personalized, week-by-week learning plan based on the student's
track, skill level, and assessment results.
"""

import json
from typing import Any, Dict, List

from agents.base import BaseAgent
from state import AgentState


SYSTEM_PROMPT = """You are the Learning Path Generator Agent for the DEPI AI Mentor platform.
Your role is to create structured, personalized week-by-week learning plans for students.

Guidelines:
- Build plans appropriate for the student's level (Beginner/Intermediate/Advanced).
- Sequence topics logically — fundamentals before advanced concepts.
- Include concrete subtopics for each week.
- Adjust scope and depth based on weaknesses identified in assessment.
- Plans should be 6-10 weeks depending on level and track complexity.
- Each week must have a clear main topic and 3-5 subtopics.
- Always return valid JSON exactly matching the requested format.
"""

# Track-specific topic sequences
TRACK_CURRICULA: Dict[str, Dict[str, List[Dict]]] = {
    "Data Analysis": {
        "Beginner": [
            {"week": 1, "topic": "Python Fundamentals", "subtopics": ["Variables and data types", "Control flow", "Functions", "Lists and dictionaries"]},
            {"week": 2, "topic": "Python for Data", "subtopics": ["File I/O", "Working with CSV", "List comprehensions", "Error handling"]},
            {"week": 3, "topic": "Pandas Basics", "subtopics": ["Series and DataFrames", "Reading data", "Selecting and filtering", "Basic statistics"]},
            {"week": 4, "topic": "Data Cleaning", "subtopics": ["Handling missing values", "Removing duplicates", "Data type conversion", "String cleaning"]},
            {"week": 5, "topic": "Data Visualization", "subtopics": ["Matplotlib basics", "Seaborn charts", "Bar, line, scatter plots", "Formatting and labels"]},
            {"week": 6, "topic": "SQL Fundamentals", "subtopics": ["SELECT queries", "WHERE and ORDER BY", "Aggregate functions", "GROUP BY"]},
            {"week": 7, "topic": "SQL Joins and Subqueries", "subtopics": ["INNER and LEFT JOIN", "Subqueries", "CTEs", "Practice exercises"]},
            {"week": 8, "topic": "Power BI", "subtopics": ["Connecting data sources", "Building reports", "DAX basics", "Dashboard design"]},
        ],
        "Intermediate": [
            {"week": 1, "topic": "Advanced Pandas", "subtopics": ["Merge and join", "Pivot tables", "GroupBy advanced", "Time series"]},
            {"week": 2, "topic": "Data Cleaning Best Practices", "subtopics": ["Outlier detection", "Imputation strategies", "Feature engineering", "Pipelines"]},
            {"week": 3, "topic": "Advanced Visualization", "subtopics": ["Plotly interactive charts", "Seaborn advanced", "Subplots", "Storytelling with data"]},
            {"week": 4, "topic": "SQL Advanced", "subtopics": ["Window functions", "Query optimization", "Indexes", "Stored procedures"]},
            {"week": 5, "topic": "Power BI Advanced", "subtopics": ["Advanced DAX", "Row-level security", "Custom visuals", "Performance tuning"]},
            {"week": 6, "topic": "Statistics for Data Analysis", "subtopics": ["Hypothesis testing", "Correlation and regression", "A/B testing", "Statistical inference"]},
        ],
        "Advanced": [
            {"week": 1, "topic": "Analytical Thinking and Problem Framing", "subtopics": ["Defining KPIs", "Business metrics", "Analytics strategy", "Stakeholder communication"]},
            {"week": 2, "topic": "Advanced Statistical Analysis", "subtopics": ["Multivariate analysis", "Time series forecasting", "Cohort analysis", "Funnel analysis"]},
            {"week": 3, "topic": "Machine Learning for Analysts", "subtopics": ["Linear regression", "Classification basics", "Clustering", "Model evaluation"]},
            {"week": 4, "topic": "Data Engineering Basics", "subtopics": ["ETL pipelines", "Database design", "Data warehouse concepts", "dbt basics"]},
            {"week": 5, "topic": "Capstone Project", "subtopics": ["End-to-end analysis", "Dashboard creation", "Stakeholder presentation", "Documentation"]},
        ],
    },
    "AI Engineering": {
        "Beginner": [
            {"week": 1, "topic": "Python Fundamentals", "subtopics": ["Data types", "Functions", "OOP basics", "Modules"]},
            {"week": 2, "topic": "Python for AI", "subtopics": ["NumPy", "Pandas", "Matplotlib", "Jupyter notebooks"]},
            {"week": 3, "topic": "Machine Learning Fundamentals", "subtopics": ["ML workflow", "Scikit-learn", "Train/test split", "Model evaluation"]},
            {"week": 4, "topic": "Supervised Learning", "subtopics": ["Linear regression", "Logistic regression", "Decision trees", "Random forests"]},
            {"week": 5, "topic": "Unsupervised Learning", "subtopics": ["K-Means clustering", "PCA", "Dimensionality reduction", "Anomaly detection"]},
            {"week": 6, "topic": "Deep Learning Basics", "subtopics": ["Neural network architecture", "Keras/TensorFlow", "Training loops", "Activation functions"]},
            {"week": 7, "topic": "NLP Fundamentals", "subtopics": ["Text preprocessing", "TF-IDF", "Word embeddings", "Sentiment analysis"]},
            {"week": 8, "topic": "Model Deployment", "subtopics": ["FastAPI", "Docker basics", "REST APIs", "Model serving"]},
        ],
        "Intermediate": [
            {"week": 1, "topic": "Advanced ML Techniques", "subtopics": ["Ensemble methods", "Gradient boosting", "XGBoost", "Hyperparameter tuning"]},
            {"week": 2, "topic": "Deep Learning Advanced", "subtopics": ["CNNs", "Transfer learning", "Regularization", "Batch normalization"]},
            {"week": 3, "topic": "Transformers and LLMs", "subtopics": ["Attention mechanism", "BERT and GPT", "Hugging Face", "Fine-tuning"]},
            {"week": 4, "topic": "LLM Application Development", "subtopics": ["OpenAI API", "Prompt engineering", "LangChain basics", "RAG systems"]},
            {"week": 5, "topic": "Agentic AI", "subtopics": ["Agent architectures", "Tool calling", "LangGraph", "Multi-agent systems"]},
            {"week": 6, "topic": "MLOps", "subtopics": ["Experiment tracking (MLflow)", "CI/CD for ML", "Model monitoring", "Cloud deployment"]},
        ],
        "Advanced": [
            {"week": 1, "topic": "Advanced LLM Engineering", "subtopics": ["Context window management", "Structured outputs", "Function calling", "Evaluation frameworks"]},
            {"week": 2, "topic": "Advanced RAG Systems", "subtopics": ["Chunking strategies", "Hybrid search", "Re-ranking", "Evaluation metrics"]},
            {"week": 3, "topic": "Production AI Systems", "subtopics": ["Scalable architectures", "Caching strategies", "Latency optimization", "Cost management"]},
            {"week": 4, "topic": "AI Safety and Ethics", "subtopics": ["Alignment basics", "Bias detection", "Responsible AI", "Governance frameworks"]},
            {"week": 5, "topic": "Capstone Project", "subtopics": ["System design", "Implementation", "Testing and evaluation", "Documentation and demo"]},
        ],
    },
    "Data Science": {
        "Beginner": [
            {"week": 1, "topic": "Python for Data Science", "subtopics": ["Python basics", "NumPy", "Pandas", "Matplotlib"]},
            {"week": 2, "topic": "Statistics Fundamentals", "subtopics": ["Descriptive statistics", "Probability basics", "Distributions", "Sampling"]},
            {"week": 3, "topic": "Exploratory Data Analysis", "subtopics": ["EDA workflow", "Univariate analysis", "Bivariate analysis", "Correlation"]},
            {"week": 4, "topic": "Data Cleaning and Preparation", "subtopics": ["Missing data", "Outliers", "Feature engineering", "Encoding"]},
            {"week": 5, "topic": "SQL for Data Science", "subtopics": ["SQL basics", "Joins", "Aggregations", "Window functions"]},
            {"week": 6, "topic": "Machine Learning Introduction", "subtopics": ["Scikit-learn", "Regression", "Classification", "Model evaluation"]},
            {"week": 7, "topic": "Visualization and Storytelling", "subtopics": ["Seaborn", "Plotly", "Dashboard design", "Communicating insights"]},
            {"week": 8, "topic": "First Capstone Project", "subtopics": ["Dataset selection", "EDA", "Modeling", "Presentation"]},
        ],
        "Intermediate": [
            {"week": 1, "topic": "Advanced Machine Learning", "subtopics": ["Ensemble methods", "Feature selection", "Cross-validation", "Pipelines"]},
            {"week": 2, "topic": "Statistical Inference", "subtopics": ["Hypothesis testing", "Confidence intervals", "A/B testing", "Bayesian basics"]},
            {"week": 3, "topic": "Deep Learning", "subtopics": ["Neural networks", "CNNs", "RNNs", "Transfer learning"]},
            {"week": 4, "topic": "Natural Language Processing", "subtopics": ["Text processing", "Embeddings", "Transformers", "Text classification"]},
            {"week": 5, "topic": "Time Series Analysis", "subtopics": ["Stationarity", "ARIMA", "Prophet", "LSTM forecasting"]},
            {"week": 6, "topic": "Model Deployment and MLOps", "subtopics": ["FastAPI", "Docker", "MLflow", "Monitoring"]},
        ],
    },
}


class LearningPathAgent(BaseAgent):
    """
    Generates a personalized weekly learning plan for a student.
    Uses track-specific curricula adjusted by level and weakness data.
    """

    def __init__(self):
        super().__init__("LearningPathAgent")

    def generate_path(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Build the learning path, optionally using the LLM to customize it
        based on specific weaknesses identified in assessment.

        Returns a list of week dicts.
        """
        track = state.get("track", "Data Analysis")
        level = state.get("student_level", "Beginner")
        scores = state.get("assessment_scores", {})
        name = state.get("student_name", "Student")

        # Get base curriculum
        track_map = TRACK_CURRICULA.get(track, TRACK_CURRICULA["Data Analysis"])
        base_path = track_map.get(level, track_map["Beginner"])

        # Identify topics that need extra attention
        weak_topics = [topic for topic, score in scores.items() if score < 0.5]

        if not weak_topics:
            return base_path

        # Use LLM to potentially adjust the path for weaknesses
        path_json = json.dumps(base_path, indent=2)
        prompt = f"""
Student: {name}
Track: {track}
Level: {level}
Weak topics (score < 0.5): {', '.join(weak_topics)}

Base learning path:
{path_json}

Review this learning path. If the student has weaknesses in topics that appear in the path,
add more subtopics to those weeks to reinforce them. You may also add a review week if needed.
Keep the total path between 6 and 10 weeks.

Return the complete learning path as a JSON array with exactly this structure:
[
  {{
    "week": 1,
    "topic": "Topic Name",
    "subtopics": ["subtopic 1", "subtopic 2", "subtopic 3"]
  }},
  ...
]
"""
        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_message=prompt,
            json_mode=True,
        )
        parsed = self._parse_json(response)
        # LLM may return {"weeks": [...]} or directly a list
        if isinstance(parsed, list):
            return parsed
        return parsed.get("weeks", base_path)

    def run(self, state: AgentState) -> AgentState:
        """Main agent entry point called by the supervisor."""
        learning_path = self.generate_path(state)
        state["learning_path"] = learning_path
        state["current_week"] = 1
        state["current_topic"] = learning_path[0]["topic"] if learning_path else ""
        state["next_agent"] = "content"

        # Build a readable summary message
        lines = [
            f"Your personalized {len(learning_path)}-week learning plan has been created!\n"
        ]
        for week in learning_path:
            subtopics = ", ".join(week.get("subtopics", []))
            lines.append(f"Week {week['week']}: {week['topic']}")
            lines.append(f"  Subtopics: {subtopics}")
        lines.append("\nLet's start with Week 1. I'll now find the best learning resources for you.")

        messages = state.get("messages", [])
        messages.append({"role": "assistant", "content": "\n".join(lines)})
        state["messages"] = messages
        return state
