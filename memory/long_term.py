"""
Long-term memory manager.
Reads and writes student-specific memory entries from PostgreSQL,
and provides a formatted summary for agent prompts.
"""

import json
from typing import List, Optional

from sqlalchemy.orm import Session

from database import LongTermMemory, SessionLocal


class MemoryManager:
    """Persists and retrieves long-term memory for a given student."""

    def __init__(self, student_id: int):
        self.student_id = student_id

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def add_entry(
        self,
        memory_type: str,
        topic: str,
        detail: str,
        confidence: float = 1.0,
        db: Optional[Session] = None,
    ) -> None:
        """Insert or update a memory entry."""
        close = db is None
        if db is None:
            db = SessionLocal()
        try:
            existing = (
                db.query(LongTermMemory)
                .filter_by(student_id=self.student_id, memory_type=memory_type, topic=topic)
                .first()
            )
            if existing:
                existing.detail = detail
                existing.confidence = confidence
            else:
                entry = LongTermMemory(
                    student_id=self.student_id,
                    memory_type=memory_type,
                    topic=topic,
                    detail=detail,
                    confidence=confidence,
                )
                db.add(entry)
            db.commit()
        finally:
            if close:
                db.close()

    def record_weakness(self, topic: str, detail: str, confidence: float = 0.8) -> None:
        self.add_entry("weakness", topic, detail, confidence)

    def record_strength(self, topic: str, detail: str, confidence: float = 0.9) -> None:
        self.add_entry("strength", topic, detail, confidence)

    def record_milestone(self, topic: str, detail: str) -> None:
        self.add_entry("milestone", topic, detail, 1.0)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_all(self, db: Optional[Session] = None) -> List[LongTermMemory]:
        """Return all memory entries for this student."""
        close = db is None
        if db is None:
            db = SessionLocal()
        try:
            return (
                db.query(LongTermMemory)
                .filter_by(student_id=self.student_id)
                .order_by(LongTermMemory.memory_type)
                .all()
            )
        finally:
            if close:
                db.close()

    def get_summary(self, db: Optional[Session] = None) -> str:
        """Return a concise text summary suitable for injection into an agent prompt."""
        entries = self.get_all(db)
        if not entries:
            return "No long-term memory entries yet."

        sections: dict = {"strength": [], "weakness": [], "milestone": [], "preference": []}
        for e in entries:
            sections.setdefault(e.memory_type, []).append(f"  - {e.topic}: {e.detail}")

        lines = []
        for section, items in sections.items():
            if items:
                lines.append(f"{section.upper()}S:")
                lines.extend(items)
        return "\n".join(lines)

    def get_weaknesses(self, db: Optional[Session] = None) -> List[str]:
        """Return list of weakness topics."""
        entries = self.get_all(db)
        return [e.topic for e in entries if e.memory_type == "weakness"]

    def get_strengths(self, db: Optional[Session] = None) -> List[str]:
        """Return list of strength topics."""
        entries = self.get_all(db)
        return [e.topic for e in entries if e.memory_type == "strength"]
