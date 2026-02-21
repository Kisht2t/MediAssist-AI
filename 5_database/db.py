"""
Step 6 — SQLite Database Layer
================================
Manages persistent storage for the MediAssist application using SQLAlchemy + SQLite.

Tables:
  sessions        — one row per chat session (user identity, start time)
  queries         — every patient message with its triage classification
  ner_entities    — extracted medical entities per query
  retrieved_docs  — which document chunks were used for each query
  responses       — the final generated response per query

Usage:
  from 5_database.db import Database

  db = Database()
  session_id = db.create_session()
  query_id   = db.save_query(session_id, "I have chest pain", "HIGH", is_emergency=True)
  db.save_response(query_id, "Please call 911 immediately...")
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = Path(__file__).parent / "mediassist.db"
DB_URL       = f"sqlite:///{DB_PATH}"
# ─────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "sessions"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_label = Column(String(100), nullable=True)   # optional name/label

    queries = relationship("Query", back_populates="session", cascade="all, delete-orphan")


class Query(Base):
    __tablename__ = "queries"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    patient_text = Column(Text, nullable=False)
    triage_level = Column(String(10), nullable=False)   # LOW | MEDIUM | HIGH
    is_emergency = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

    session         = relationship("ChatSession", back_populates="queries")
    ner_entities    = relationship("NEREntity",     back_populates="query", cascade="all, delete-orphan")
    retrieved_docs  = relationship("RetrievedDoc",  back_populates="query", cascade="all, delete-orphan")
    response        = relationship("Response",       back_populates="query", uselist=False, cascade="all, delete-orphan")


class NEREntity(Base):
    __tablename__ = "ner_entities"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    query_id    = Column(Integer, ForeignKey("queries.id"), nullable=False)
    entity_type = Column(String(50))     # symptoms | body_parts | durations | measurements
    entity_text = Column(String(255))

    query = relationship("Query", back_populates="ner_entities")


class RetrievedDoc(Base):
    __tablename__ = "retrieved_docs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    query_id     = Column(Integer, ForeignKey("queries.id"), nullable=False)
    rank         = Column(Integer)        # position after reranking (1 = top)
    source       = Column(String(255))    # metadata["source"] from ChromaDB
    content_snip = Column(Text)           # first 500 chars of the chunk

    query = relationship("Query", back_populates="retrieved_docs")


class Response(Base):
    __tablename__ = "responses"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    query_id      = Column(Integer, ForeignKey("queries.id"), nullable=False, unique=True)
    answer_text   = Column(Text, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    query = relationship("Query", back_populates="response")


# ── Database Helper Class ─────────────────────────────────────────────────────

class Database:
    """
    High-level interface for all database operations.

    Example:
        db = Database()
        session_id = db.create_session(user_label="Demo User")
        query_id   = db.save_query(session_id, "Headache for 3 days", "MEDIUM")
        db.save_entities(query_id, {"symptoms": ["headache"], "durations": ["3 days"]})
        db.save_retrieved_docs(query_id, [{"source": "common_symptoms", "content": "..."}])
        db.save_response(query_id, "Based on your symptoms...")
    """

    def __init__(self, db_url: str = DB_URL):
        self._engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self._engine)
        print(f"Database ready: {db_url}")

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self, user_label: str = None) -> int:
        with Session(self._engine) as sess:
            chat_session = ChatSession(user_label=user_label)
            sess.add(chat_session)
            sess.commit()
            sess.refresh(chat_session)
            return chat_session.id

    def get_session_history(self, session_id: int) -> list[dict]:
        """Returns all queries + responses for a session (for chat replay)."""
        with Session(self._engine) as sess:
            session = sess.get(ChatSession, session_id)
            if not session:
                return []

            history = []
            for q in sorted(session.queries, key=lambda x: x.created_at):
                entry = {
                    "patient_text": q.patient_text,
                    "triage_level": q.triage_level,
                    "is_emergency": q.is_emergency,
                    "answer": q.response.answer_text if q.response else None,
                    "timestamp": q.created_at.isoformat(),
                }
                history.append(entry)
            return history

    # ── Queries ───────────────────────────────────────────────────────────────

    def save_query(
        self,
        session_id: int,
        patient_text: str,
        triage_level: str,
        is_emergency: bool = False,
    ) -> int:
        with Session(self._engine) as sess:
            query = Query(
                session_id=session_id,
                patient_text=patient_text,
                triage_level=triage_level,
                is_emergency=is_emergency,
            )
            sess.add(query)
            sess.commit()
            sess.refresh(query)
            return query.id

    # ── NER Entities ──────────────────────────────────────────────────────────

    def save_entities(self, query_id: int, entities: dict) -> None:
        """
        entities: {"symptoms": [...], "body_parts": [...], ...}
        """
        with Session(self._engine) as sess:
            for entity_type, values in entities.items():
                for text in values:
                    if text:
                        sess.add(NEREntity(
                            query_id=query_id,
                            entity_type=entity_type,
                            entity_text=str(text)[:255],
                        ))
            sess.commit()

    # ── Retrieved Docs ────────────────────────────────────────────────────────

    def save_retrieved_docs(self, query_id: int, docs: list[dict]) -> None:
        """
        docs: list of {"source": str, "content": str}
        """
        with Session(self._engine) as sess:
            for rank, doc in enumerate(docs, start=1):
                sess.add(RetrievedDoc(
                    query_id=query_id,
                    rank=rank,
                    source=doc.get("source", "unknown")[:255],
                    content_snip=doc.get("content", "")[:500],
                ))
            sess.commit()

    # ── Responses ─────────────────────────────────────────────────────────────

    def save_response(self, query_id: int, answer_text: str) -> None:
        with Session(self._engine) as sess:
            sess.add(Response(query_id=query_id, answer_text=answer_text))
            sess.commit()

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_triage_stats(self) -> dict:
        """Returns count of queries per triage level."""
        from sqlalchemy import func, select

        with Session(self._engine) as sess:
            rows = sess.execute(
                select(Query.triage_level, func.count(Query.id))
                .group_by(Query.triage_level)
            ).all()
            return {level: count for level, count in rows}

    def get_recent_queries(self, limit: int = 20) -> list[dict]:
        """Returns most recent queries across all sessions."""
        from sqlalchemy import select, desc

        with Session(self._engine) as sess:
            rows = sess.execute(
                select(Query).order_by(desc(Query.created_at)).limit(limit)
            ).scalars().all()

            return [
                {
                    "id": q.id,
                    "patient_text": q.patient_text[:100],
                    "triage_level": q.triage_level,
                    "is_emergency": q.is_emergency,
                    "timestamp": q.created_at.isoformat(),
                }
                for q in rows
            ]


# ── Init check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = Database()
    print("Tables created successfully.")

    # Demo round-trip
    sid = db.create_session(user_label="Test User")
    qid = db.save_query(sid, "I have a headache and fever for 3 days.", "MEDIUM")
    db.save_entities(qid, {"symptoms": ["headache", "fever"], "durations": ["3 days"]})
    db.save_retrieved_docs(qid, [
        {"source": "common_symptoms", "content": "FEVER: Moderate (100.4–103°F)..."},
        {"source": "triage_levels",   "content": "MEDIUM: Fever above 101°F lasting..."},
    ])
    db.save_response(qid, "Based on your description, you have a medium-urgency condition...")

    print(f"\nSession history for session {sid}:")
    for entry in db.get_session_history(sid):
        print(f"  [{entry['triage_level']}] {entry['patient_text'][:60]}")

    print(f"\nTriage stats: {db.get_triage_stats()}")
