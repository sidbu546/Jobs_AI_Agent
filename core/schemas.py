"""Canonical Pydantic schemas for jobs and profile.

All ingestion adapters map their raw payloads into these types.
All agents read from these types. Nothing else touches the raw API responses.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Job schema
# ---------------------------------------------------------------------------


class JobSource(StrEnum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    USAJOBS = "usajobs"
    UNKNOWN = "unknown"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    UNKNOWN = "unknown"


class SalaryRange(BaseModel):
    min_usd: int | None = None
    max_usd: int | None = None
    currency: str = "USD"


class Job(BaseModel):
    # Stable dedup key — built from company + normalized_title + location
    dedup_key: str = Field(default="", description="SHA256 of company+normalized_title+location")

    source: JobSource
    source_id: str = Field(description="The job ID as given by the source ATS")
    company: str
    title: str
    normalized_title: str = Field(default="", description="Lowercased, stripped title for dedup")

    location: str = ""
    remote: bool = False
    url: str

    description: str = ""
    department: str = ""
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    salary: SalaryRange | None = None

    posted_at: datetime | None = None
    expires_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    # Opaque raw payload — never read by agents, only for debugging
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # Set by the match agent
    match_score: float | None = None

    @field_validator("title", "company", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def build_derived_fields(self) -> Job:
        self.normalized_title = re.sub(r"\s+", " ", self.title.lower().strip())
        key_parts = f"{self.company.lower()}|{self.normalized_title}|{self.location.lower()}"
        self.dedup_key = hashlib.sha256(key_parts.encode()).hexdigest()
        return self


# ---------------------------------------------------------------------------
# Profile schema
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    start: str  # "2022-06" or "2022"
    end: str | None = None  # None → present


class Experience(BaseModel):
    company: str
    title: str
    location: str = ""
    dates: DateRange
    bullets: list[str] = Field(description="Accomplishment bullets — the tailoring agent reorders these")
    technologies: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: str
    bullets: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    url: str = ""
    dates: DateRange | None = None


class Education(BaseModel):
    institution: str
    degree: str
    field: str
    gpa: str = ""
    dates: DateRange
    highlights: list[str] = Field(default_factory=list)


class WorkAuth(BaseModel):
    """LOCKED — these strings are echoed verbatim, never generated.

    The system only ever reads and repeats these values. It must never
    generate, infer, or modify work-auth answers — flag any question
    that doesn't map to a key here for human review.
    """

    authorized_to_work_us: bool = False
    will_require_sponsorship: bool = False
    visa_type: str = ""

    # Verbatim canonical answers keyed by the exact ATS question text patterns.
    # Matched case-insensitively by the fill agent. Always populated explicitly
    # from profile_data/work_auth.yaml at load time — this default is only a
    # schema fallback, never real data.
    canonical_answers: dict[str, str] = Field(default_factory=dict)

    def answer_for(self, question: str) -> str | None:
        """Return the verbatim answer for a work-auth question, or None if not mapped."""
        q = question.lower().strip()
        for pattern, answer in self.canonical_answers.items():
            if pattern in q:
                return answer
        return None


class Profile(BaseModel):
    name: str
    email: str
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""
    summary: str = ""

    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    # Separate locked block — agents must not read this as free text to generate from
    work_auth: WorkAuth = Field(default_factory=WorkAuth)
