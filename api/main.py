"""
FastAPI application for the DEPI AI Mentor platform.

Exposes REST endpoints for:
- Student registration and onboarding
- Chat interactions routed through the agent graph
- Assignment submission
- Progress tracking
- Memory inspection
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from agents.supervisor import mentor_graph
from config import settings
from database import (
    Assignment,
    AssessmentResult,
    LearningPath,
    SessionLocal,
    Student,
    WeekPlan,
    create_tables,
    get_db,
)
from memory.long_term import MemoryManager
from state import AgentState


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DEPI AI Mentor API",
    description="Digital Egypt Pioneers — Personalized AI Learning Mentor Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
def startup_event():
    create_tables()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class StudentCreate(BaseModel):
    name: str
    email: str
    university: Optional[str] = None
    study_year: Optional[str] = None
    major: Optional[str] = None
    track: str  # Data Analysis | AI Engineering | Data Science


class ChatMessage(BaseModel):
    student_id: int
    message: str
    current_state: Optional[Dict[str, Any]] = None


class SubmitAssignment(BaseModel):
    student_id: int
    assignment_id: Optional[int] = None
    submission: str
    current_state: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    next_agent: str
    updated_state: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "DEPI AI Mentor API",
        "description": "Digital Egypt Pioneers — Personalized AI Learning Mentor",
        "version": "1.0.0",
    }


@app.post("/students/register", response_model=Dict[str, Any])
def register_student(student_data: StudentCreate, db: Session = Depends(get_db)):
    """Register a new student and return their ID and initial state."""
    # Check if email already exists
    existing = db.query(Student).filter_by(email=student_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    student = Student(
        name=student_data.name,
        email=student_data.email,
        university=student_data.university,
        study_year=student_data.study_year,
        major=student_data.major,
        track=student_data.track,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    # Build initial state
    initial_state: AgentState = {
        "student_id": student.id,
        "student_name": student.name,
        "student_email": student.email,
        "track": student.track,
        "messages": [],
        "current_week": 1,
        "next_agent": "assessment",
    }

    return {
        "student_id": student.id,
        "name": student.name,
        "track": student.track,
        "message": "Registration successful. Ready to start assessment.",
        "initial_state": initial_state,
    }


@app.get("/students/{student_id}", response_model=Dict[str, Any])
def get_student(student_id: int, db: Session = Depends(get_db)):
    """Retrieve student profile and current progress."""
    student = db.query(Student).filter_by(id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    memory = MemoryManager(student_id)
    memory_summary = memory.get_summary(db)

    return {
        "id": student.id,
        "name": student.name,
        "email": student.email,
        "university": student.university,
        "track": student.track,
        "level": student.level,
        "created_at": student.created_at.isoformat(),
        "memory_summary": memory_summary,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatMessage, db: Session = Depends(get_db)):
    """
    Main chat endpoint. Receives a student message, appends it to the state,
    runs the appropriate agent via the supervisor graph, and returns the reply.
    """
    student = db.query(Student).filter_by(id=payload.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    # Restore or initialize state
    state: AgentState = payload.current_state or {
        "student_id": student.id,
        "student_name": student.name,
        "student_email": student.email,
        "track": student.track,
        "messages": [],
        "current_week": 1,
        "next_agent": "assessment",
    }

    # Append the new user message to history
    messages = state.get("messages", [])
    messages.append({"role": "user", "content": payload.message})
    state["messages"] = messages

    # Run the graph (single step — one agent handles the current turn)
    updated_state = mentor_graph.invoke(state)

    # Extract the latest assistant reply
    updated_messages = updated_state.get("messages", [])
    latest_reply = next(
        (m["content"] for m in reversed(updated_messages) if m["role"] == "assistant"),
        "I didn't understand that. Could you please rephrase?",
    )

    # Update student level in DB if it changed
    new_level = updated_state.get("student_level")
    if new_level and new_level != student.level:
        student.level = new_level
        db.commit()

    return ChatResponse(
        reply=latest_reply,
        next_agent=updated_state.get("next_agent", "assessment"),
        updated_state=dict(updated_state),
    )


@app.post("/submit-assignment", response_model=Dict[str, Any])
def submit_assignment(payload: SubmitAssignment, db: Session = Depends(get_db)):
    """
    Submit an assignment for evaluation.
    Stores the submission and triggers the evaluation agent.
    """
    student = db.query(Student).filter_by(id=payload.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    state: AgentState = payload.current_state or {}
    state["student_id"] = student.id
    state["student_name"] = student.name
    state["track"] = student.track
    state["student_submission"] = payload.submission
    state["next_agent"] = "evaluation"

    messages = state.get("messages", [])
    messages.append({"role": "user", "content": f"[SUBMISSION]\n{payload.submission}"})
    state["messages"] = messages

    updated_state = mentor_graph.invoke(state)

    eval_result = updated_state.get("evaluation_result", {})
    latest_reply = next(
        (m["content"] for m in reversed(updated_state.get("messages", [])) if m["role"] == "assistant"),
        "",
    )

    # Persist assignment record to database
    if eval_result:
        current_week = state.get("current_week", 1)
        assignment_record = Assignment(
            student_id=student.id,
            title=state.get("current_assignment", {}).get("title", "Assignment"),
            description=state.get("current_assignment", {}).get("description", ""),
            assignment_type=state.get("current_assignment", {}).get("type", "code"),
            difficulty=student.level,
            submission=payload.submission,
            score=eval_result.get("final_score"),
            feedback=eval_result.get("feedback"),
            correctness_score=eval_result.get("correctness_score"),
            quality_score=eval_result.get("quality_score"),
            best_practices_score=eval_result.get("best_practices_score"),
            submitted_at=datetime.utcnow(),
        )
        db.add(assignment_record)
        db.commit()

    return {
        "reply": latest_reply,
        "evaluation": eval_result,
        "next_agent": updated_state.get("next_agent", "adaptive"),
        "updated_state": dict(updated_state),
    }


@app.get("/students/{student_id}/progress", response_model=Dict[str, Any])
def get_progress(student_id: int, db: Session = Depends(get_db)):
    """Return a student's learning progress summary."""
    student = db.query(Student).filter_by(id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    assignments = db.query(Assignment).filter_by(student_id=student_id).all()
    avg_score = (
        sum(a.score for a in assignments if a.score is not None) / len(assignments)
        if assignments
        else 0
    )

    memory = MemoryManager(student_id)

    return {
        "student_id": student_id,
        "name": student.name,
        "track": student.track,
        "level": student.level,
        "total_assignments": len(assignments),
        "average_score": round(avg_score, 1),
        "strengths": memory.get_strengths(db),
        "weaknesses": memory.get_weaknesses(db),
        "memory_summary": memory.get_summary(db),
    }


@app.get("/students/{student_id}/memory", response_model=Dict[str, Any])
def get_memory(student_id: int, db: Session = Depends(get_db)):
    """Return all long-term memory entries for a student."""
    student = db.query(Student).filter_by(id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    memory = MemoryManager(student_id)
    entries = memory.get_all(db)

    return {
        "student_id": student_id,
        "entries": [
            {
                "type": e.memory_type,
                "topic": e.topic,
                "detail": e.detail,
                "confidence": e.confidence,
                "created_at": e.created_at.isoformat(),
            }
            for e in entries
        ],
    }
