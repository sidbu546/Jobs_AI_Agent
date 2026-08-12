"""Jobicy remote jobs adapter — free API, no auth, keyword tag search.

Endpoint: https://jobicy.com/api/v2/remote-jobs?count=N&tag=<keyword>
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.schemas import EmploymentType, Job, JobSource
from ingestion.base import IngestionAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://jobicy.com/api/v2/remote-jobs"
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_NL = re.compile(r"\n{3,}")

_TYPE_MAP = {
    "full-time": EmploymentType.FULL_TIME,
    "part-time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "internship": EmploymentType.INTERNSHIP,
    "freelance": EmploymentType.CONTRACT,
}


def _strip_html(html: str) -> str:
    text = _HTML_TAG.sub("\n", html or "")
    return _MULTI_NL.sub("\n\n", text).strip()


def _map_job(raw: dict) -> Job:
    types = [t.lower() for t in (raw.get("jobType") or [])]
    emp_type = EmploymentType.UNKNOWN
    for t in types:
        if t in _TYPE_MAP:
            emp_type = _TYPE_MAP[t]
            break

    posted_at = None
    pub = raw.get("pubDate", "")
    if pub:
        try:
            posted_at = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except ValueError:
            pass

    return Job(
        source=JobSource.UNKNOWN,
        source_id=str(raw.get("id", "")),
        company=raw.get("companyName", ""),
        title=raw.get("jobTitle", ""),
        location=raw.get("jobGeo", "Worldwide"),
        remote=True,
        url=raw.get("url", ""),
        description=_strip_html(raw.get("jobDescription", "")),
        employment_type=emp_type,
        posted_at=posted_at,
        raw=raw,
    )


class JobicyAdapter(IngestionAdapter):
    """Keyword/tag-based adapter for Jobicy remote jobs."""

    source_name = "jobicy"

    def __init__(self, query: str, limit: int = 50) -> None:
        self._query = query
        self._limit = min(limit, 50)  # API max is 50

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _fetch(self, client: httpx.AsyncClient) -> list[dict]:
        params = {"count": self._limit, "tag": self._query}
        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("jobs", [])
        logger.info("Jobicy '%s' → %d listings", self._query, len(jobs))
        return jobs

    async def fetch_jobs(self, **_kwargs) -> AsyncIterator[Job]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                raw_jobs = await self._fetch(client)
                for raw in raw_jobs:
                    try:
                        yield _map_job(raw)
                    except Exception as exc:
                        logger.warning("Skipping malformed Jobicy job: %s", exc)
            except httpx.HTTPError as exc:
                logger.error("Jobicy error: %s", exc)

    async def health_check(self) -> bool:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(_BASE_URL, params={"count": 1, "tag": "engineer"})
                return resp.status_code == 200
            except httpx.HTTPError:
                return False
