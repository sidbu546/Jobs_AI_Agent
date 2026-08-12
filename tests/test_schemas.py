"""Unit tests for canonical schemas — no network, no LLM."""

import pytest
from core.schemas import Job, JobSource, WorkAuth


def test_job_dedup_key_is_deterministic():
    j1 = Job(source=JobSource.GREENHOUSE, source_id="1", company="Stripe", title="ML Engineer", url="https://x.com")
    j2 = Job(source=JobSource.GREENHOUSE, source_id="999", company="Stripe", title="ML Engineer", url="https://y.com")
    assert j1.dedup_key == j2.dedup_key


def test_job_dedup_key_differs_on_title():
    j1 = Job(source=JobSource.GREENHOUSE, source_id="1", company="Stripe", title="ML Engineer", url="https://x.com")
    j2 = Job(source=JobSource.GREENHOUSE, source_id="1", company="Stripe", title="Data Scientist", url="https://x.com")
    assert j1.dedup_key != j2.dedup_key


def test_normalized_title_lowercased():
    j = Job(source=JobSource.GREENHOUSE, source_id="1", company="Acme", title="  Senior ML Engineer  ", url="https://x.com")
    assert j.normalized_title == "senior ml engineer"


def _make_work_auth() -> WorkAuth:
    return WorkAuth(
        authorized_to_work_us=True,
        will_require_sponsorship=True,
        visa_type="Test Visa",
        canonical_answers={"require sponsorship": "Yes"},
    )


def test_work_auth_verbatim_echo():
    wa = _make_work_auth()
    assert wa.answer_for("require sponsorship") == "Yes"
    assert wa.answer_for("Are you a US citizen?") is None


def test_work_auth_case_insensitive():
    wa = _make_work_auth()
    assert wa.answer_for("REQUIRE SPONSORSHIP NOW OR IN THE FUTURE") == "Yes"


def test_work_auth_unmapped_returns_none():
    wa = _make_work_auth()
    assert wa.answer_for("What is your favorite color?") is None
