"""Storage interface — SQLite+Chroma for local dev, swappable to Postgres+pgvector."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime

import chromadb
from sqlalchemy import Boolean, Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.schemas import Job


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./jobs.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")


# ---------------------------------------------------------------------------
# SQLAlchemy model (relational store — dedup, expiry, seen-state)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    dedup_key = Column(String(64), primary_key=True)
    source = Column(String(32), nullable=False)
    source_id = Column(String(256), nullable=False)
    company = Column(String(256), nullable=False)
    title = Column(String(256), nullable=False)
    location = Column(String(256), default="")
    remote = Column(Boolean, default=False)
    url = Column(String(1024), nullable=False)
    description = Column(Text, default="")
    department = Column(String(256), default="")
    employment_type = Column(String(64), default="unknown")
    match_score = Column(Float, nullable=True)
    posted_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    seen = Column(Boolean, default=False)
    applied = Column(Boolean, default=False)
    raw_json = Column(Text, default="{}")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Abstract storage interface
# ---------------------------------------------------------------------------


class JobStore(ABC):
    @abstractmethod
    def upsert(self, job: Job) -> bool:
        """Store job. Returns True if it was new (not a dedup)."""

    @abstractmethod
    def get(self, dedup_key: str) -> Job | None: ...

    @abstractmethod
    def mark_seen(self, dedup_key: str) -> None: ...

    @abstractmethod
    def unseen_above_threshold(self, threshold: float) -> list[Job]: ...

    @abstractmethod
    def expire_stale(self) -> int:
        """Remove jobs whose expires_at is in the past. Returns count removed."""


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SQLiteJobStore(JobStore):
    def __init__(self) -> None:
        init_db()
        self._Session = SessionLocal

    def _row_to_job(self, row: JobRow) -> Job:
        raw = json.loads(row.raw_json or "{}")
        return Job(
            dedup_key=row.dedup_key,
            source=row.source,  # type: ignore[arg-type]
            source_id=row.source_id,
            company=row.company,
            title=row.title,
            location=row.location or "",
            remote=row.remote or False,
            url=row.url,
            description=row.description or "",
            department=row.department or "",
            employment_type=row.employment_type,  # type: ignore[arg-type]
            match_score=row.match_score,
            posted_at=row.posted_at,
            expires_at=row.expires_at,
            ingested_at=row.ingested_at,
            raw=raw,
        )

    def upsert(self, job: Job) -> bool:
        with self._Session() as session:
            existing = session.get(JobRow, job.dedup_key)
            if existing:
                return False
            row = JobRow(
                dedup_key=job.dedup_key,
                source=job.source.value,
                source_id=job.source_id,
                company=job.company,
                title=job.title,
                location=job.location,
                remote=job.remote,
                url=job.url,
                description=job.description,
                department=job.department,
                employment_type=job.employment_type.value,
                match_score=job.match_score,
                posted_at=job.posted_at,
                expires_at=job.expires_at,
                ingested_at=job.ingested_at,
                raw_json=json.dumps(job.raw),
            )
            session.add(row)
            session.commit()
            return True

    def get(self, dedup_key: str) -> Job | None:
        with self._Session() as session:
            row = session.get(JobRow, dedup_key)
            return self._row_to_job(row) if row else None

    def mark_seen(self, dedup_key: str) -> None:
        with self._Session() as session:
            row = session.get(JobRow, dedup_key)
            if row:
                row.seen = True
                session.commit()

    def unseen_above_threshold(self, threshold: float) -> list[Job]:
        with self._Session() as session:
            rows = (
                session.query(JobRow)
                .filter(JobRow.seen == False, JobRow.match_score >= threshold)  # noqa: E712
                .order_by(JobRow.match_score.desc())
                .all()
            )
            return [self._row_to_job(r) for r in rows]

    def expire_stale(self) -> int:
        with self._Session() as session:
            now = datetime.utcnow()
            result = (
                session.query(JobRow)
                .filter(JobRow.expires_at != None, JobRow.expires_at < now)  # noqa: E711
                .delete()
            )
            session.commit()
            return result


# ---------------------------------------------------------------------------
# Chroma vector store (used by match agent)
# ---------------------------------------------------------------------------


class JobVectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._col = self._client.get_or_create_collection("jobs")

    def upsert(self, job: Job, embedding: list[float]) -> None:
        self._col.upsert(
            ids=[job.dedup_key],
            embeddings=[embedding],
            metadatas=[{"company": job.company, "title": job.title, "url": job.url}],
            documents=[job.description],
        )

    def query(self, embedding: list[float], n_results: int = 20) -> list[dict]:
        results = self._col.query(query_embeddings=[embedding], n_results=n_results)
        return [
            {"dedup_key": id_, "distance": dist, **meta}
            for id_, dist, meta in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]
