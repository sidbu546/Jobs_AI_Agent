"""Tests for the Greenhouse adapter — mock HTTP, no real API calls."""

import pytest
from pytest_httpx import HTTPXMock

from core.schemas import JobSource
from ingestion.greenhouse import GreenhouseAdapter, _map_job

SAMPLE_RAW = {
    "id": 12345,
    "title": "Machine Learning Engineer",
    "absolute_url": "https://boards.greenhouse.io/stripe/jobs/12345",
    "location": {"name": "San Francisco, CA or Remote"},
    "content": "We are looking for an ML engineer with Python and PyTorch experience.",
    "updated_at": "2026-06-01T12:00:00Z",
    "departments": [{"name": "Engineering"}],
    "metadata": [{"name": "Employment Type", "value": "Full Time"}],
}


def test_map_job_canonical():
    job = _map_job("stripe", SAMPLE_RAW)
    assert job.source == JobSource.GREENHOUSE
    assert job.company == "stripe"
    assert job.title == "Machine Learning Engineer"
    assert job.source_id == "12345"
    assert job.department == "Engineering"
    assert job.remote is True  # "Remote" in location string
    assert job.dedup_key  # non-empty


def test_map_job_dedup_key_stable():
    job_a = _map_job("stripe", SAMPLE_RAW)
    job_b = _map_job("stripe", {**SAMPLE_RAW, "id": 99999})  # different id, same title/location
    assert job_a.dedup_key == job_b.dedup_key


@pytest.mark.asyncio
async def test_fetch_jobs_happy_path(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://boards-api.greenhouse.io/v1/boards/stripe/jobs?content=true",
        json={"jobs": [SAMPLE_RAW]},
    )
    adapter = GreenhouseAdapter(companies=["stripe"])
    jobs = [job async for job in adapter.fetch_jobs()]
    assert len(jobs) == 1
    assert jobs[0].title == "Machine Learning Engineer"


@pytest.mark.asyncio
async def test_fetch_jobs_404_is_skipped(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://boards-api.greenhouse.io/v1/boards/notacompany/jobs?content=true",
        status_code=404,
    )
    adapter = GreenhouseAdapter(companies=["notacompany"])
    jobs = [job async for job in adapter.fetch_jobs()]
    assert jobs == []


@pytest.mark.asyncio
async def test_fetch_jobs_skips_malformed(httpx_mock: HTTPXMock):
    bad = {"id": None, "title": None, "absolute_url": None}  # will fail Job validation
    good = SAMPLE_RAW
    httpx_mock.add_response(
        url="https://boards-api.greenhouse.io/v1/boards/stripe/jobs?content=true",
        json={"jobs": [bad, good]},
    )
    adapter = GreenhouseAdapter(companies=["stripe"])
    jobs = [job async for job in adapter.fetch_jobs()]
    # The good one should survive; the bad one is logged and skipped
    assert len(jobs) == 1
