"""Profile knowledge base.

Loads profile YAML files and exposes a retrieval interface for agents.
All agent reads go through here — never direct YAML access.

The work_auth block is separate and locked: it is echoed verbatim,
never used as a generation source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from core.schemas import (
    DateRange,
    Education,
    Experience,
    Profile,
    Project,
    WorkAuth,
)

_PROFILE_DATA_DIR = Path(__file__).parent.parent / "profile_data"


def _load_yaml(filename: str) -> dict | list:
    path = _PROFILE_DATA_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def _build_experience(raw: list[dict]) -> list[Experience]:
    return [
        Experience(
            company=e["company"],
            title=e["title"],
            location=e.get("location", ""),
            dates=DateRange(start=e["dates"]["start"], end=e["dates"].get("end")),
            bullets=e.get("bullets", []),
            technologies=e.get("technologies", []),
        )
        for e in raw
    ]


def _build_projects(raw: list[dict]) -> list[Project]:
    return [
        Project(
            name=p["name"],
            description=p["description"],
            bullets=p.get("bullets", []),
            technologies=p.get("technologies", []),
            url=p.get("url", ""),
            dates=DateRange(start=p["dates"]["start"], end=p["dates"].get("end")) if "dates" in p else None,
        )
        for p in raw
    ]


def _build_education(raw: list[dict]) -> list[Education]:
    return [
        Education(
            institution=e["institution"],
            degree=e["degree"],
            field=e["field"],
            gpa=str(e.get("gpa", "")),
            dates=DateRange(start=e["dates"]["start"], end=e["dates"].get("end")),
            highlights=e.get("highlights", []),
        )
        for e in raw
    ]


@lru_cache(maxsize=1)
def load_profile() -> Profile:
    meta = _load_yaml("meta.yaml")
    assert isinstance(meta, dict)

    experiences_raw = _load_yaml("experience.yaml")
    assert isinstance(experiences_raw, list)

    projects_raw = _load_yaml("projects.yaml")
    assert isinstance(projects_raw, list)

    skills_raw = _load_yaml("skills.yaml")
    assert isinstance(skills_raw, list)

    education_raw = _load_yaml("education.yaml")
    assert isinstance(education_raw, list)

    work_auth_raw = _load_yaml("work_auth.yaml")
    assert isinstance(work_auth_raw, dict)

    work_auth = WorkAuth(
        authorized_to_work_us=work_auth_raw["authorized_to_work_us"],
        will_require_sponsorship=work_auth_raw["will_require_sponsorship"],
        visa_type=work_auth_raw["visa_type"],
        canonical_answers=work_auth_raw.get("canonical_answers", {}),
    )

    return Profile(
        name=meta["name"],
        email=meta["email"],
        phone=meta.get("phone", ""),
        location=meta.get("location", ""),
        linkedin=meta.get("linkedin", ""),
        github=meta.get("github", ""),
        portfolio=meta.get("portfolio", ""),
        summary=meta.get("summary", ""),
        experiences=_build_experience(experiences_raw),
        projects=_build_projects(projects_raw),
        skills=skills_raw,
        education=_build_education(education_raw),
        work_auth=work_auth,
    )


def profile_as_text(profile: Profile | None = None) -> str:
    """Flat text representation for embedding / RAG context."""
    p = profile or load_profile()
    lines: list[str] = [f"Name: {p.name}", f"Email: {p.email}", f"Location: {p.location}", ""]

    lines.append("EXPERIENCE")
    for exp in p.experiences:
        end = exp.dates.end or "Present"
        lines.append(f"  {exp.title} @ {exp.company} ({exp.dates.start}–{end})")
        for b in exp.bullets:
            lines.append(f"    • {b}")
        if exp.technologies:
            lines.append(f"    Tech: {', '.join(exp.technologies)}")
    lines.append("")

    lines.append("PROJECTS")
    for proj in p.projects:
        lines.append(f"  {proj.name}: {proj.description}")
        for b in proj.bullets:
            lines.append(f"    • {b}")
        if proj.technologies:
            lines.append(f"    Tech: {', '.join(proj.technologies)}")
    lines.append("")

    lines.append(f"SKILLS: {', '.join(p.skills)}")
    lines.append("")

    lines.append("EDUCATION")
    for edu in p.education:
        end = edu.dates.end or "Present"
        lines.append(f"  {edu.degree} in {edu.field} @ {edu.institution} ({edu.dates.start}–{end})")
        if edu.gpa:
            lines.append(f"    GPA: {edu.gpa}")
    return "\n".join(lines)
