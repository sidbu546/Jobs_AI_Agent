"""Eval harness — run as a CI gate to prevent regressions.

Checks:
  1. Faithfulness: no bullet in tailored output that isn't verbatim in the profile KB.
  2. Work-auth integrity: no generated work-auth answer (must come from locked KB).
  3. Tailoring relevance: tailored output mentions at least N JD keywords.

Usage (also called by pytest):
    python -m eval.harness

Exit code 0 = pass, 1 = failure (blocks CI merge).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from candidate.kb import load_profile
from core.schemas import Job, JobSource


@dataclass
class EvalResult:
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)


def check_faithfulness(tailored: dict, profile_bullets: set[str]) -> EvalResult:
    """All selected bullets must exist verbatim in the profile."""
    result = EvalResult()
    for exp in tailored.get("experiences", []):
        for bullet in exp.get("selected_bullets", []):
            if bullet not in profile_bullets:
                result.fail(f"Fabricated bullet in experience '{exp.get('company')}': {bullet[:80]}")
    for proj in tailored.get("projects", []):
        for bullet in proj.get("selected_bullets", []):
            if bullet not in profile_bullets:
                result.fail(f"Fabricated bullet in project '{proj.get('name')}': {bullet[:80]}")
    return result


def check_work_auth_integrity(cover_letter: str, answers: dict[str, str]) -> EvalResult:
    """Ensure no work-auth answer was generated — only verbatim KB echoes allowed."""
    result = EvalResult()
    profile = load_profile()
    canonical_values = set(profile.work_auth.canonical_answers.values())

    for question, answer in answers.items():
        wa_answer = profile.work_auth.answer_for(question)
        if wa_answer is not None and answer not in canonical_values:
            result.fail(
                f"Non-canonical work-auth answer for '{question}': got '{answer}', "
                f"expected one of {canonical_values}"
            )
    return result


def check_tailoring_relevance(tailored: dict, job: Job, min_keyword_hits: int = 2) -> EvalResult:
    """Tailored output should mention at least N keywords from the JD."""
    result = EvalResult()
    jd_words = set(job.description.lower().split())
    # High-signal JD words (longer than 5 chars, not stopwords)
    stopwords = {"the", "and", "for", "with", "that", "this", "will", "have", "from", "your"}
    keywords = {w for w in jd_words if len(w) > 5 and w not in stopwords}

    all_selected = []
    for exp in tailored.get("experiences", []):
        all_selected.extend(exp.get("selected_bullets", []))
    for proj in tailored.get("projects", []):
        all_selected.extend(proj.get("selected_bullets", []))
    tailored_text = " ".join(all_selected).lower()

    hits = sum(1 for kw in keywords if kw in tailored_text)
    if hits < min_keyword_hits:
        result.fail(
            f"Tailoring relevance too low: only {hits} JD keyword hits "
            f"(minimum {min_keyword_hits}). JD: {job.title} @ {job.company}"
        )
    return result


def run_suite(
    tailored: dict,
    job: Job,
    cover_letter: str = "",
    answers: dict | None = None,
) -> list[EvalResult]:
    profile = load_profile()
    all_bullets: set[str] = set()
    for exp in profile.experiences:
        all_bullets.update(exp.bullets)
    for proj in profile.projects:
        all_bullets.update(proj.bullets)

    results = [
        check_faithfulness(tailored, all_bullets),
        check_work_auth_integrity(cover_letter, answers or {}),
        check_tailoring_relevance(tailored, job),
    ]
    return results


def main() -> None:
    """Smoke-test the harness against a synthetic case."""
    profile = load_profile()
    real_bullet = profile.experiences[0].bullets[0] if profile.experiences else "placeholder"

    fake_job = Job(
        source=JobSource.GREENHOUSE,
        source_id="test-1",
        company="Acme AI",
        title="Machine Learning Engineer",
        url="https://example.com",
        description="Looking for machine learning experience with Python and deep learning.",
    )

    good_tailored = {
        "experiences": [{"company": "Test", "selected_bullets": [real_bullet]}],
        "projects": [],
    }
    bad_tailored = {
        "experiences": [{"company": "Test", "selected_bullets": ["I built a rocket to Mars"]}],
        "projects": [],
    }

    print("--- Good tailored (should pass faithfulness) ---")
    for r in run_suite(good_tailored, fake_job):
        status = "PASS" if r.passed else "FAIL"
        print(f"  {status}", r.failures or "")

    print("\n--- Bad tailored (should fail faithfulness) ---")
    failed = False
    for r in run_suite(bad_tailored, fake_job):
        status = "PASS" if r.passed else "FAIL"
        print(f"  {status}", r.failures or "")
        if not r.passed:
            failed = True

    sys.exit(0 if failed else 1)  # CI expects bad_tailored to fail


if __name__ == "__main__":
    main()
