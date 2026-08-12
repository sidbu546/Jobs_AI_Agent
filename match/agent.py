"""Match agent — scores job descriptions against the profile.

Two-stage:
  1. Cheap deterministic pre-filter (title keywords, location, seniority).
  2. Semantic cosine similarity via sentence-transformers (local, no API key needed).

Model: all-MiniLM-L6-v2 (~90 MB, downloads once, runs on CPU in <1s per job).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from core.schemas import Job, Profile
from candidate.kb import profile_as_text


def _focused_profile(profile: Profile) -> str:
    lines = []
    for exp in profile.experiences:
        lines.append(f"{exp.title} at {exp.company}")
    for proj in profile.projects:
        tech = ", ".join(proj.technologies[:6])
        lines.append(f"{proj.name}: {tech}")
    lines.append("Skills: " + ", ".join(profile.skills))
    for edu in profile.education:
        lines.append(f"{edu.degree} {edu.field} {edu.institution}")
    return "\n".join(lines)


logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Seniority ──────────────────────────────────────────────────────────────
# Titles with these words require 4+ years — block them for an entry-level search.
_EXCLUDE_SENIORITY = re.compile(
    r"\b(senior|sr\.?|staff|lead|principal|architect|founding|distinguished|"
    r"director|vp|vice president|manager|head of|executive|chief|c-level|"
    r"president|founder|partner|experienced)\b",
    re.IGNORECASE,
)

# These override the seniority block — explicitly entry-level / intern titles.
_ENTRY_LEVEL_OVERRIDE = re.compile(
    r"\b(intern|internship|new grad|new-grad|junior|associate|entry.?level|"
    r"early career|university grad|campus|student|graduate)\b",
    re.IGNORECASE,
)

# ── Domain ─────────────────────────────────────────────────────────────────
# At least one of these must appear in the title.
_INCLUDE_TITLE_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp", "software engineer", "backend engineer", "research engineer",
    "applied scientist", "python developer", "llm", "generative ai",
    "research scientist", "ml ops", "mlops", "data engineer", "ai researcher",
    "computer vision", "deep learning", "applied ml", "ai scientist",
    "applied research", "engineer", "scientist", "researcher", "developer",
    "intern",
}

# ── Location ───────────────────────────────────────────────────────────────
_EXCLUDE_LOCATIONS = {
    "india", "london", "berlin", "toronto", "tokyo", "sydney", "singapore",
    "remote (uk)", "remote (eu)", "europe only", "canada only",
    "australia", "germany", "france", "netherlands", "uk only",
    "paris", "seoul", "warsaw", "munich", "amsterdam", "budapest",
    "casablanca", "madrid", "stockholm", "luxembourg", "abu dhabi",
    "montreal", "belgrade", "zurich", "melbourne",
}

_US_LOCATION_HINTS = {
    "san francisco", "new york", "seattle", "austin", "boston",
    "los angeles", "chicago", "denver", "remote, us", "remote us",
    "united states", "usa", "palo alto", "mountain view", "menlo park",
    "new york city", "nyc", "cambridge", "silicon valley", "bay area",
    " ca,", " ny,", " wa,", " tx,", " ma,", " co,", " ca ", " ny ",
}

# ── Experience requirement extraction ──────────────────────────────────────
# Detect if the JD explicitly requires 3+ years of professional experience.
_EXP_REQUIRED = re.compile(
    r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s+(?:of\s+)?(?:professional\s+|industry\s+|work\s+)?experience",
    re.IGNORECASE,
)


def _requires_too_much_experience(description: str) -> bool:
    """Return True if the JD states 3+ years of professional experience as a requirement."""
    if not description:
        return False
    for match in _EXP_REQUIRED.finditer(description[:2000]):
        years = int(match.group(1))
        if years >= 3:
            return True
    return False


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer  # lazy — avoids PyTorch load on app startup
    logger.info("Loading sentence-transformers model '%s'…", _EMBEDDING_MODEL)
    return SentenceTransformer(_EMBEDDING_MODEL)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# MiniLM JD-vs-profile cosine lands ~0.15 (noise) to ~0.45 (strong match).
_SCORE_FLOOR = 0.15
_SCORE_CEIL  = 0.45

def normalize_score(raw: float) -> float:
    return round(max(0.0, min(1.0, (raw - _SCORE_FLOOR) / (_SCORE_CEIL - _SCORE_FLOOR))), 3)


def _passes_pre_filter(job: Job) -> bool:
    title_lower = job.title.lower()
    location_lower = job.location.lower()

    is_entry_override = bool(_ENTRY_LEVEL_OVERRIDE.search(title_lower))

    # Block senior/staff/lead roles unless the title explicitly says entry/intern
    if not is_entry_override and _EXCLUDE_SENIORITY.search(title_lower):
        return False

    # Must be a relevant AI/ML/engineering domain
    if not any(kw in title_lower for kw in _INCLUDE_TITLE_KEYWORDS):
        return False

    # Location: block known non-US cities/regions
    has_exclude = any(loc in location_lower for loc in _EXCLUDE_LOCATIONS)
    if has_exclude and not job.remote:
        return False

    # Description-based experience guard (skip if requires 3+ years)
    if not is_entry_override and _requires_too_much_experience(job.description):
        return False

    return True


class MatchAgent:
    def __init__(self, profile: Profile, threshold: float = 0.40) -> None:
        self._threshold = threshold
        self._model = _load_model()
        profile_text = _focused_profile(profile)
        self._profile_embedding: np.ndarray = self._model.encode(profile_text, convert_to_numpy=True)

    def score(self, job: Job) -> Job:
        """Score a single job. Prefer score_batch for bulk operations."""
        if not _passes_pre_filter(job):
            job.match_score = 0.0
            return job
        jd_text = f"{job.title}\n{job.title}\n{job.description[:600]}"
        jd_emb: np.ndarray = self._model.encode(jd_text, convert_to_numpy=True)
        job.match_score = normalize_score(_cosine(self._profile_embedding, jd_emb))
        return job

    def score_batch(self, jobs: list[Job]) -> list[Job]:
        """Score jobs with one vectorised encode call — ~5x faster than calling score() in a loop."""
        to_score: list[Job] = []
        for job in jobs:
            if _passes_pre_filter(job):
                to_score.append(job)
            else:
                job.match_score = 0.0

        if to_score:
            texts = [f"{j.title}\n{j.title}\n{j.description[:600]}" for j in to_score]
            embeddings: np.ndarray = self._model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            for job, emb in zip(to_score, embeddings):
                job.match_score = normalize_score(_cosine(self._profile_embedding, emb))

        return jobs

    def is_worth_applying(self, job: Job) -> bool:
        return (job.match_score or 0.0) >= self._threshold
