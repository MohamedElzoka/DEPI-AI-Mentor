"""
SQLAlchemy database models for the DEPI AI Mentor platform.
Stores student profiles, learning paths, assignments, grades, and long-term memory.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from config import settings


# ---------------------------------------------------------------------------
# Base and engine
# ---------------------------------------------------------------------------

engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Student(Base):
    """Stores student onboarding information and current status."""

    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    university = Column(String(200))
    study_year = Column(String(50))
    major = Column(String(100))
    track = Column(String(100))  # Data Analysis | AI Engineering | Data Science
    level = Column(String(20), default="Beginner")  # Beginner | Intermediate | Advanced
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    learning_path = relationship("LearningPath", back_populates="student", uselist=False)
    assignments = relationship("Assignment", back_populates="student")
    memory_entries = relationship("LongTermMemory", back_populates="student")
    interview_sessions = relationship("InterviewSession", back_populates="student")
    assessment_result = relationship("AssessmentResult", back_populates="student", uselist=False)


class AssessmentResult(Base):
    """Stores the initial skill assessment scores per student."""

    __tablename__ = "assessment_results"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), unique=True)
    python_score = Column(Float, default=0.0)
    sql_score = Column(Float, default=0.0)
    statistics_score = Column(Float, default=0.0)
    excel_score = Column(Float, default=0.0)
    power_bi_score = Column(Float, default=0.0)
    overall_level = Column(String(20))  # Beginner | Intermediate | Advanced
    assessed_at = Column(DateTime, default=datetime.utcnow)
    raw_responses = Column(Text)  # JSON string of Q&A pairs

    student = relationship("Student", back_populates="assessment_result")


class LearningPath(Base):
    """Stores the generated weekly learning plan for a student."""

    __tablename__ = "learning_paths"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), unique=True)
    total_weeks = Column(Integer, default=8)
    current_week = Column(Integer, default=1)
    is_completed = Column(Boolean, default=False)
    path_json = Column(Text)  # Full JSON of all weeks and topics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("Student", back_populates="learning_path")
    weeks = relationship("WeekPlan", back_populates="learning_path")


class WeekPlan(Base):
    """Represents a single week inside a learning path."""

    __tablename__ = "week_plans"

    id = Column(Integer, primary_key=True, index=True)
    learning_path_id = Column(Integer, ForeignKey("learning_paths.id"))
    week_number = Column(Integer)
    topic = Column(String(200))
    subtopics = Column(Text)  # JSON list
    status = Column(String(20), default="pending")  # pending | in_progress | completed | extended
    score = Column(Float, nullable=True)

    learning_path = relationship("LearningPath", back_populates="weeks")
    assignments = relationship("Assignment", back_populates="week")


class Assignment(Base):
    """Stores generated assignments and student submissions."""

    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    week_id = Column(Integer, ForeignKey("week_plans.id"), nullable=True)
    title = Column(String(200))
    description = Column(Text)
    assignment_type = Column(String(50))  # code | quiz | project
    difficulty = Column(String(20))
    submission = Column(Text, nullable=True)  # Student's submitted answer or code
    score = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)
    correctness_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    best_practices_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=True)

    student = relationship("Student", back_populates="assignments")
    week = relationship("WeekPlan", back_populates="assignments")


class LongTermMemory(Base):
    """Persistent memory entries for each student — weaknesses, strengths, milestones."""

    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    memory_type = Column(String(50))  # strength | weakness | milestone | preference
    topic = Column(String(200))
    detail = Column(Text)
    confidence = Column(Float, default=1.0)  # 0-1 scale
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("Student", back_populates="memory_entries")


class InterviewSession(Base):
    """Stores mock interview sessions and results."""

    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    session_type = Column(String(50))  # technical | behavioral | mixed
    questions_json = Column(Text)  # JSON list of {question, answer, score, feedback}
    overall_score = Column(Float, nullable=True)
    strengths = Column(Text, nullable=True)
    improvements = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="interview_sessions")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency-injection helper for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
