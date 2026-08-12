"""Tests for the match agent — local embeddings, no API calls."""

import pytest
from core.schemas import Job, JobSource
from match.agent import MatchAgent, _passes_pre_filter
from candidate.kb import load_profile


def _make_job(title: str, location: str = "Remote", description: str = "") -> Job:
    return Job(
        source=JobSource.GREENHOUSE,
        source_id="test",
        company="Acme",
        title=title,
        location=location,
        url="https://example.com",
        description=description,
    )


# --- Pre-filter tests (no model needed) ---

def test_prefilter_passes_ml_engineer():
    assert _passes_pre_filter(_make_job("Machine Learning Engineer"))


def test_prefilter_passes_senior_ml():
    # Senior ML/AI titles are allowed — many entry-level AI roles carry "Senior" at startups
    assert _passes_pre_filter(_make_job("Senior Machine Learning Engineer"))

def test_prefilter_blocks_director():
    assert not _passes_pre_filter(_make_job("Director of AI"))


def test_prefilter_blocks_director():
    assert not _passes_pre_filter(_make_job("Director of AI"))


def test_prefilter_blocks_unrelated_title():
    assert not _passes_pre_filter(_make_job("Sales Account Executive"))


def test_prefilter_blocks_non_us_non_remote():
    assert not _passes_pre_filter(_make_job("ML Engineer", location="London"))


def test_prefilter_passes_remote_any_location():
    job = _make_job("ML Engineer", location="London")
    job.remote = True
    assert _passes_pre_filter(job)


# --- Semantic scoring tests (loads model once) ---

@pytest.fixture(scope="module")
def match_agent():
    return MatchAgent(profile=load_profile(), threshold=0.45)


def test_ml_job_scores_higher_than_sales(match_agent):
    ml_job = _make_job(
        "Machine Learning Engineer",
        description="Build NLP pipelines, fine-tune LLMs, deploy with PyTorch and Python.",
    )
    sales_job = _make_job(
        "Sales Development Representative",
        description="Cold calling, pipeline management, CRM experience required.",
    )
    ml_score = match_agent.score(ml_job).match_score
    sales_score = match_agent.score(sales_job).match_score
    assert ml_score is not None
    # Sales job should be pre-filtered to 0.0
    assert sales_score == 0.0
    assert ml_score > 0.0


def test_relevant_job_passes_threshold(match_agent):
    job = _make_job(
        "AI Research Engineer",
        description=(
            "Research engineer to build RAG pipelines, evaluate LLMs, "
            "Python, PyTorch, HuggingFace, ChromaDB. NLP experience preferred."
        ),
    )
    result = match_agent.score(job)
    assert result.match_score is not None
    assert result.match_score > 0.3  # Should have meaningful similarity to Siddhanth's profile


def test_score_is_between_0_and_1(match_agent):
    job = _make_job("Data Scientist", description="Python, scikit-learn, pandas, SQL.")
    result = match_agent.score(job)
    assert 0.0 <= (result.match_score or 0.0) <= 1.0
