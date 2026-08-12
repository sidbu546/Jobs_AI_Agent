"""Lever public postings API adapter — stub, mirrors Greenhouse pattern.

Endpoint: https://api.lever.co/v0/postings/{company}?mode=json
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.schemas import EmploymentType, Job, JobSource
from ingestion.base import IngestionAdapter

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_NL = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    text = _HTML_TAG.sub("\n", html or "")
    return _MULTI_NL.sub("\n\n", text).strip()


logger = logging.getLogger(__name__)

_BASE_URL = "https://api.lever.co/v0/postings"

_COMMITMENT_MAP: dict[str, EmploymentType] = {
    "full-time": EmploymentType.FULL_TIME,
    "part-time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "intern": EmploymentType.INTERNSHIP,
    "internship": EmploymentType.INTERNSHIP,
}


def _map_job(company: str, raw: dict) -> Job:
    categories = raw.get("categories", {}) or {}
    location = categories.get("location", "") or raw.get("workplaceType", "")
    commitment = categories.get("commitment", "")
    team = categories.get("team", "")

    posted_ts = raw.get("createdAt")
    posted_at = datetime.fromtimestamp(posted_ts / 1000, tz=UTC) if posted_ts else None

    # descriptionBody is either a nested dict (old format) or an HTML string (new format)
    desc_body = raw.get("descriptionBody")
    if isinstance(desc_body, dict):
        description_parts = [
            section.get("content", "")
            for section in (desc_body.get("descriptionBodySections") or [])
        ]
        description = _strip_html("\n".join(description_parts))
    elif isinstance(desc_body, str) and desc_body.strip():
        description = _strip_html(desc_body)
    else:
        description = raw.get("descriptionPlain", "") or _strip_html(raw.get("description", ""))

    return Job(
        source=JobSource.LEVER,
        source_id=raw.get("id", ""),
        company=company,
        title=raw.get("text", ""),
        location=location,
        remote="remote" in location.lower() or raw.get("workplaceType", "").lower() == "remote",
        url=raw.get("hostedUrl", ""),
        description=description,
        department=team,
        employment_type=_COMMITMENT_MAP.get(commitment.lower(), EmploymentType.UNKNOWN),
        posted_at=posted_at,
        raw=raw,
    )


class LeverAdapter(IngestionAdapter):
    source_name = "lever"

    def __init__(self, companies: list[str], timeout: float = 15.0) -> None:
        self._companies = companies
        self._timeout = timeout

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _fetch_board(self, client: httpx.AsyncClient, company: str) -> list[dict]:
        resp = await client.get(f"{_BASE_URL}/{company}", params={"mode": "json", "limit": 250})
        resp.raise_for_status()
        return resp.json()

    async def _safe_fetch(self, client: httpx.AsyncClient, company: str) -> tuple[str, list[dict]]:
        try:
            jobs = await self._fetch_board(client, company)
            logger.info("Lever %s → %d listings", company, len(jobs))
            return company, jobs
        except httpx.HTTPStatusError as exc:
            logger.error("Lever %s HTTP %s", company, exc.response.status_code)
            return company, []
        except httpx.HTTPError as exc:
            logger.error("Lever %s network error: %s", company, exc)
            return company, []

    async def fetch_jobs(self, **_kwargs) -> AsyncIterator[Job]:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            results = await asyncio.gather(
                *[self._safe_fetch(client, c) for c in self._companies]
            )
        for company, raw_jobs in results:
            for raw in raw_jobs:
                try:
                    yield _map_job(company, raw)
                except Exception as exc:
                    logger.warning("Skipping malformed Lever job from %s: %s", company, exc)

    async def health_check(self) -> bool:
        if not self._companies:
            return False
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{_BASE_URL}/{self._companies[0]}", params={"mode": "json"})
                return resp.status_code == 200
            except httpx.HTTPError:
                return False
