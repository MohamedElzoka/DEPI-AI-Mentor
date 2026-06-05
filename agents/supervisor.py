"""
Supervisor Agent and LangGraph workflow definition.

The Supervisor orchestrates all specialized agents by reading the `next_agent`
field in the shared state and routing execution accordingly.
The full multi-agent graph is defined and compiled here.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from agents.adaptive import AdaptiveLearningAgent
from agents.assessment import AssessmentAgent
from agents.assignment import AssignmentAgent
from agents.career import CareerAdvisorAgent
from agents.content import ContentAgent
from agents.evaluation import EvaluationAgent
from agents.interview import InterviewAgent
from agents.learning_path import LearningPathAgent
from state import AgentState


# ---------------------------------------------------------------------------
# Agent singletons
# ---------------------------------------------------------------------------

assessment_agent = AssessmentAgent()
learning_path_agent = LearningPathAgent()
content_agent = ContentAgent()
assignment_agent = AssignmentAgent()
evaluation_agent = EvaluationAgent()
adaptive_agent = AdaptiveLearningAgent()
interview_agent = InterviewAgent()
career_agent = CareerAdvisorAgent()


# ---------------------------------------------------------------------------
# Node wrappers — each node calls its agent's run() method
# ---------------------------------------------------------------------------

def run_assessment(state: AgentState) -> AgentState:
    return assessment_agent.run(state)


def run_learning_path(state: AgentState) -> AgentState:
    return learning_path_agent.run(state)


def run_content(state: AgentState) -> AgentState:
    return content_agent.run(state)


def run_assignment(state: AgentState) -> AgentState:
    return assignment_agent.run(state)


def run_evaluation(state: AgentState) -> AgentState:
    return evaluation_agent.run(state)


def run_adaptive(state: AgentState) -> AgentState:
    return adaptive_agent.run(state)


def run_interview(state: AgentState) -> AgentState:
    return interview_agent.run(state)


def run_career(state: AgentState) -> AgentState:
    return career_agent.run(state)


# ---------------------------------------------------------------------------
# Router function — reads next_agent from state to determine next node
# ---------------------------------------------------------------------------

def route(state: AgentState) -> str:
    """
    Routing function used as the conditional edge after each node.
    Maps the `next_agent` string to the corresponding graph node name.
    """
    next_agent = state.get("next_agent", "assessment")
    route_map = {
        "assessment": "assessment",
        "learning_path": "learning_path",
        "content": "content",
        "assignment": "assignment",
        "evaluation": "evaluation",
        "adaptive": "adaptive",
        "interview": "interview",
        "career": "career",
        "complete": END,
    }
    return route_map.get(next_agent, END)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and return the compiled LangGraph StateGraph for the full
    DEPI AI Mentor multi-agent workflow.
    """
    graph = StateGraph(AgentState)

    # Register all agent nodes
    graph.add_node("assessment", run_assessment)
    graph.add_node("learning_path", run_learning_path)
    graph.add_node("content", run_content)
    graph.add_node("assignment", run_assignment)
    graph.add_node("evaluation", run_evaluation)
    graph.add_node("adaptive", run_adaptive)
    graph.add_node("interview", run_interview)
    graph.add_node("career", run_career)

    # Set the entry point
    graph.set_entry_point("assessment")

    # Add conditional edges from each node using the router
    for node in [
        "assessment",
        "learning_path",
        "content",
        "assignment",
        "evaluation",
        "adaptive",
        "interview",
        "career",
    ]:
        graph.add_conditional_edges(node, route)

    return graph.compile()


# Compile the graph once at module import time
mentor_graph = build_graph()
