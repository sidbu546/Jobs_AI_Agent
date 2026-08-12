"""Tests for the eval harness — this is the CI gate."""

from core.schemas import Job, JobSource  # noqa: F401 (JobSource used in test body)
from eval.harness import check_faithfulness, check_tailoring_relevance, check_work_auth_integrity

REAL_BULLETS = {
    "Engineered a custom UI module for high-dimensional data visualization, increasing operational productivity by 30% for teams monitoring real-time data streams",
    "Conducted comparative research on RAG vs. long-context architectures using Llama 3 and Qwen 2.5 to optimize model reasoning",
}

FAKE_JOB = Job(
    source=JobSource.GREENHOUSE,
    source_id="eval-1",
    company="Acme AI",
    title="Machine Learning Engineer",
    url="https://example.com",
    description="We need Python machine learning experience with neural networks and deep learning.",
)


def test_faithfulness_passes_with_real_bullets():
    tailored = {"experiences": [{"company": "Test", "selected_bullets": [list(REAL_BULLETS)[0]]}], "projects": []}
    result = check_faithfulness(tailored, REAL_BULLETS)
    assert result.passed


def test_faithfulness_fails_with_fabricated_bullet():
    tailored = {
        "experiences": [{"company": "Test", "selected_bullets": ["I led a team of 50 engineers at Google"]}],
        "projects": [],
    }
    result = check_faithfulness(tailored, REAL_BULLETS)
    assert not result.passed
    assert any("Fabricated" in f for f in result.failures)


def test_work_auth_integrity_passes_canonical():
    result = check_work_auth_integrity("", {"require sponsorship": "Yes"})
    assert result.passed


def test_work_auth_integrity_fails_generated():
    result = check_work_auth_integrity("", {"require sponsorship": "No, I am a US citizen"})
    assert not result.passed


def test_tailoring_relevance_passes():
    bullet = "Conducted comparative research on RAG vs. long-context architectures using Llama 3 and Qwen 2.5 to optimize model reasoning"
    jd_with_overlap = Job(
        source=JobSource.GREENHOUSE,
        source_id="eval-overlap",
        company="Acme AI",
        title="Research Engineer",
        url="https://example.com",
        description="Seeking research experience with RAG architectures and long-context reasoning for enterprise AI.",
    )
    tailored = {"experiences": [{"selected_bullets": [bullet]}], "projects": []}
    result = check_tailoring_relevance(tailored, jd_with_overlap, min_keyword_hits=2)
    assert result.passed


def test_tailoring_relevance_fails_empty():
    tailored = {"experiences": [], "projects": []}
    result = check_tailoring_relevance(tailored, FAKE_JOB, min_keyword_hits=2)
    assert not result.passed
