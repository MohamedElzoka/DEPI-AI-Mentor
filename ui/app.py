"""
Streamlit UI for the DEPI AI Mentor platform.

Provides a clean chat-based interface that communicates with the FastAPI backend.
Manages session state for student onboarding and the full learning journey.
"""

import json
import time
from typing import Any, Dict, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DEPI AI Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "student_id": None,
        "student_name": None,
        "track": None,
        "agent_state": None,
        "chat_history": [],
        "current_agent": "assessment",
        "onboarded": False,
        "assignment_mode": False,
        "current_assignment": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# API helper functions
# ---------------------------------------------------------------------------

def register_student(name: str, email: str, university: str, year: str, major: str, track: str) -> Optional[Dict]:
    """Register a student via the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/students/register",
            json={
                "name": name,
                "email": email,
                "university": university,
                "study_year": year,
                "major": major,
                "track": track,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Please ensure it is running.")
        return None


def send_message(student_id: int, message: str, agent_state: Dict) -> Optional[Dict]:
    """Send a chat message to the agent via the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "student_id": student_id,
                "message": message,
                "current_state": agent_state,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Chat error: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server.")
        return None


def submit_assignment_api(student_id: int, submission: str, agent_state: Dict) -> Optional[Dict]:
    """Submit an assignment for evaluation via the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/submit-assignment",
            json={
                "student_id": student_id,
                "submission": submission,
                "current_state": agent_state,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Submission error: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server.")
        return None


def get_progress(student_id: int) -> Optional[Dict]:
    """Fetch student progress from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/students/{student_id}/progress", timeout=30)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("DEPI AI Mentor")
        st.caption("Digital Egypt Pioneers Initiative")
        st.divider()

        if st.session_state.onboarded:
            st.success(f"Student: {st.session_state.student_name}")
            st.info(f"Track: {st.session_state.track}")

            current_agent = st.session_state.current_agent
            agent_labels = {
                "assessment": "Step 1: Assessment",
                "learning_path": "Step 2: Learning Path",
                "content": "Step 3: Content",
                "assignment": "Step 4: Assignment",
                "evaluation": "Step 5: Evaluation",
                "adaptive": "Step 6: Adaptive Learning",
                "interview": "Step 7: Mock Interview",
                "career": "Step 8: Career Advice",
                "complete": "Completed",
            }
            st.write("Current Stage:")
            st.write(f"**{agent_labels.get(current_agent, current_agent)}**")
            st.divider()

            # Show progress if available
            if st.button("Refresh Progress", use_container_width=True):
                progress = get_progress(st.session_state.student_id)
                if progress:
                    st.metric("Assignments Completed", progress.get("total_assignments", 0))
                    st.metric("Average Score", f"{progress.get('average_score', 0):.1f}/100")
                    strengths = progress.get("strengths", [])
                    weaknesses = progress.get("weaknesses", [])
                    if strengths:
                        st.write("Strengths:", ", ".join(strengths))
                    if weaknesses:
                        st.write("Weaknesses:", ", ".join(weaknesses))

            st.divider()
            if st.button("Reset Session", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.info("Please complete registration to begin your learning journey.")


render_sidebar()


# ---------------------------------------------------------------------------
# Onboarding form
# ---------------------------------------------------------------------------

def render_onboarding():
    st.title("Welcome to DEPI AI Mentor")
    st.markdown(
        "The Digital Egypt Pioneers AI Mentor is your personal guide through "
        "your tech learning journey. Let's get started!"
    )
    st.divider()

    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name", placeholder="Mohamed Ahmed")
            email = st.text_input("Email Address", placeholder="student@university.edu.eg")
            university = st.text_input("University", placeholder="Cairo University")

        with col2:
            year = st.selectbox("Study Year", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate"])
            major = st.text_input("Major / Specialization", placeholder="Computer Science")
            track = st.selectbox(
                "Learning Track",
                ["Data Analysis", "AI Engineering", "Data Science"],
                help="Choose the career path you want to pursue",
            )

        st.divider()
        submitted = st.form_submit_button("Start My Learning Journey", use_container_width=True, type="primary")

        if submitted:
            if not name or not email:
                st.error("Please fill in your name and email.")
            else:
                with st.spinner("Setting up your account..."):
                    result = register_student(name, email, university, year, major, track)
                    if result:
                        st.session_state.student_id = result["student_id"]
                        st.session_state.student_name = name
                        st.session_state.track = track
                        st.session_state.agent_state = result["initial_state"]
                        st.session_state.onboarded = True
                        st.session_state.current_agent = "assessment"
                        st.success("Registration successful! Starting your assessment...")
                        time.sleep(1)
                        st.rerun()


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def render_chat():
    st.title(f"DEPI AI Mentor — {st.session_state.track}")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Assignment submission mode
    if st.session_state.assignment_mode and st.session_state.current_agent in ["evaluation", "assignment"]:
        st.divider()
        st.subheader("Submit Your Assignment")

        assignment = st.session_state.current_assignment
        if assignment:
            st.info(f"Assignment: {assignment.get('title', 'Current Assignment')}")
            assignment_type = assignment.get("type", "code")

            if assignment_type == "code":
                submission = st.text_area(
                    "Paste your Python code here:",
                    height=300,
                    placeholder="# Write your solution here\n",
                )
            else:
                submission = st.text_area(
                    "Your answer:",
                    height=200,
                    placeholder="Type your answer here...",
                )

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Submit", type="primary", use_container_width=True):
                    if submission.strip():
                        with st.spinner("Evaluating your submission..."):
                            result = submit_assignment_api(
                                st.session_state.student_id,
                                submission,
                                st.session_state.agent_state,
                            )
                            if result:
                                st.session_state.chat_history.append(
                                    {"role": "user", "content": f"[Submission]\n```\n{submission[:200]}...\n```"}
                                )
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": result["reply"]}
                                )
                                st.session_state.agent_state = result["updated_state"]
                                st.session_state.current_agent = result.get("next_agent", "adaptive")
                                st.session_state.assignment_mode = False
                                st.rerun()
                    else:
                        st.warning("Please enter your submission before clicking Submit.")
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.assignment_mode = False
                    st.rerun()
        return

    # Regular chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            result = send_message(
                st.session_state.student_id,
                user_input,
                st.session_state.agent_state,
            )

        if result:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["reply"]}
            )
            st.session_state.agent_state = result["updated_state"]
            next_agent = result.get("next_agent", "assessment")
            st.session_state.current_agent = next_agent

            # Check if an assignment was generated — enable submission mode
            updated_state = result["updated_state"]
            if next_agent == "evaluation" and updated_state.get("current_assignment"):
                st.session_state.assignment_mode = True
                st.session_state.current_assignment = updated_state.get("current_assignment")

        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if not st.session_state.onboarded:
    render_onboarding()
else:
    render_chat()
