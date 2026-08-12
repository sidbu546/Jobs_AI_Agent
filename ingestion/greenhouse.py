"""Greenhouse public board API adapter.

Endpoint: https://boards-api.greenhouse.io/v1/boards/{company}/jobs
No auth required — public board listings only.

Usage:
    adapter = GreenhouseAdapter(companies=["stripe", "airbnb"])
    async for job in adapter.fetch_jobs():
        store.upsert(job)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.schemas import EmploymentType, Job, JobSource
from ingestion.base import IngestionAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

# Greenhouse content[] key → our EmploymentType
_EMPLOYMENT_MAP: dict[str, EmploymentType] = {
    "full time": EmploymentType.FULL_TIME,
    "full-time": EmploymentType.FULL_TIME,
    "part time": EmploymentType.PART_TIME,
    "part-time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "intern": EmploymentType.INTERNSHIP,
    "internship": EmploymentType.INTERNSHIP,
}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _detect_remote(title: str, location: str, content: str) -> bool:
    haystack = f"{title} {location} {content}".lower()
    return any(kw in haystack for kw in ("remote", "anywhere", "distributed", "work from home"))


def _map_employment_type(raw_type: str | None) -> EmploymentType:
    if not raw_type:
        return EmploymentType.UNKNOWN
    return _EMPLOYMENT_MAP.get(raw_type.lower().strip(), EmploymentType.UNKNOWN)


def _map_job(company: str, raw: dict) -> Job:
    """Map a single Greenhouse job object to our canonical schema."""
    url = raw.get("absolute_url") or ""
    title = raw.get("title") or ""
    if not url or not title:
        raise ValueError(f"Missing required fields (title={title!r}, url={url!r})")

    location = raw.get("location", {}) or {}
    location_str = location.get("name", "") if isinstance(location, dict) else str(location)

    content = raw.get("content", "") or ""
    job_id = str(raw.get("id", ""))

    # Greenhouse nests department under departments[]
    departments = raw.get("departments", []) or []
    dept_name = departments[0].get("name", "") if departments else ""

    metadata = raw.get("metadata", []) or []
    employment_type_raw = next(
        (m.get("value") for m in metadata if m.get("name", "").lower() in ("employment type", "type")),
        None,
    )

    return Job(
        source=JobSource.GREENHOUSE,
        source_id=job_id,
        company=company,
        title=title,
        location=location_str,
        remote=_detect_remote(title, location_str, content),
        url=url,
        description=content,
        department=dept_name,
        employment_type=_map_employment_type(employment_type_raw),
        posted_at=_parse_dt(raw.get("updated_at")),
        raw=raw,
    )


class GreenhouseAdapter(IngestionAdapter):
    source_name = "greenhouse"

    def __init__(self, companies: list[str], timeout: float = 15.0) -> None:
        self._companies = companies
        self._timeout = timeout

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _fetch_board(self, client: httpx.AsyncClient, company: str) -> list[dict]:
        url = f"{_BASE_URL}/{company}/jobs"
        resp = await client.get(url, params={"content": "true"})
        resp.raise_for_status()
        data = resp.json()
        return data.get("jobs", [])

    async def _safe_fetch(self, client: httpx.AsyncClient, company: str) -> tuple[str, list[dict]]:
        """Fetch one company board, returning (company, jobs) — never raises."""
        try:
            jobs = await self._fetch_board(client, company)
            logger.info("Greenhouse %s → %d listings", company, len(jobs))
            return company, jobs
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Greenhouse board not found: %s (check slug)", company)
            else:
                logger.error("Greenhouse %s HTTP %s", company, exc.response.status_code)
            return company, []
        except httpx.HTTPError as exc:
            logger.error("Greenhouse %s network error: %s", company, exc)
            return company, []

    async def fetch_jobs(self, **_kwargs) -> AsyncIterator[Job]:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Fetch all company boards in parallel — was sequential (18 × RTT)
            results = await asyncio.gather(
                *[self._safe_fetch(client, c) for c in self._companies]
            )
        for company, raw_jobs in results:
            for raw in raw_jobs:
                try:
                    yield _map_job(company, raw)
                except Exception as exc:
                    logger.warning("Skipping malformed job from %s: %s", company, exc)

    async def health_check(self) -> bool:
        if not self._companies:
            return False
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{_BASE_URL}/{self._companies[0]}/jobs")
                return resp.status_code == 200
            except httpx.HTTPError:
                return False
