"""Remotive job search adapter — free API, no auth, keyword-driven.

Endpoint: https://remotive.com/api/remote-jobs?search={query}&limit=N
All results are remote. Returns company + full description.
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

_BASE_URL = "https://remotive.com/api/remote-jobs"

_TYPE_MAP = {
    "full_time": EmploymentType.FULL_TIME,
    "part_time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "internship": EmploymentType.INTERNSHIP,
    "freelance": EmploymentType.CONTRACT,
}

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_NL = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    text = _HTML_TAG.sub("\n", html or "")
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _map_job(raw: dict) -> Job:
    return Job(
        source=JobSource.UNKNOWN,
        source_id=str(raw.get("id", "")),
        company=raw.get("company_name", ""),
        title=raw.get("title", ""),
        location=raw.get("candidate_required_location", "Worldwide"),
        remote=True,
        url=raw.get("url", ""),
        description=_strip_html(raw.get("description", "")),
        department=raw.get("category", ""),
        employment_type=_TYPE_MAP.get(raw.get("job_type", ""), EmploymentType.UNKNOWN),
        posted_at=datetime.fromisoformat(raw["publication_date"].replace("Z", "+00:00"))
        if raw.get("publication_date")
        else None,
        raw=raw,
    )


_US_LOCATION_TERMS = {"usa", "united states", "us", "america", "worldwide", "anywhere", "global"}


def _is_us_eligible(location: str) -> bool:
    """Return True if the job's required location includes the US."""
    loc = location.lower()
    return any(term in loc for term in _US_LOCATION_TERMS)


class RemotiveAdapter(IngestionAdapter):
    """Search-first adapter: keywords → matching jobs across all companies."""

    source_name = "remotive"

    def __init__(
        self,
        query: str,
        limit: int = 100,
        timeout: float = 15.0,
        job_type: str | None = None,
        us_only: bool = True,
    ) -> None:
        self._query = query
        self._limit = limit
        self._timeout = timeout
        self._job_type = job_type   # "full_time" | "internship" | None (all)
        self._us_only = us_only

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _fetch(self, client: httpx.AsyncClient) -> list[dict]:
        params: dict = {"search": self._query, "limit": self._limit}
        if self._job_type:
            params["job_type"] = self._job_type
        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("jobs", [])
        logger.info("Remotive '%s' → %d listings", self._query, len(jobs))
        return jobs

    async def fetch_jobs(self, **_kwargs) -> AsyncIterator[Job]:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                raw_jobs = await self._fetch(client)
                for raw in raw_jobs:
                    try:
                        job = _map_job(raw)
                        if self._us_only and not _is_us_eligible(job.location):
                            continue
                        yield job
                    except Exception as exc:
                        logger.warning("Skipping malformed Remotive job: %s", exc)
            except httpx.HTTPStatusError as exc:
                logger.error("Remotive HTTP %s for query '%s'", exc.response.status_code, self._query)
            except httpx.HTTPError as exc:
                logger.error("Remotive network error: %s", exc)

    async def health_check(self) -> bool:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(_BASE_URL, params={"search": "engineer", "limit": 1})
                return resp.status_code == 200
            except httpx.HTTPError:
                return False
