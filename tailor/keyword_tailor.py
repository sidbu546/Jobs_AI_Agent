"""Keyword-based resume tailoring — works with zero API keys.

Extracts signal words from the JD, scores each profile bullet by overlap,
and selects the top bullets per role. Verbatim from the KB — no generation.
"""

from __future__ import annotations

import re
from collections import Counter

from core.schemas import Job, Profile

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "will", "have", "from",
    "your", "you", "our", "are", "was", "not", "can", "but", "all", "has",
    "been", "they", "their", "who", "what", "how", "its", "also", "we",
    "an", "a", "in", "to", "of", "is", "at", "by", "be", "as", "or",
    "on", "it", "if", "up", "do", "so", "no", "he", "she", "us", "my",
    "we", "than", "then", "more", "any", "may", "use",
}


def _extract_keywords(text: str, top_n: int = 30) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\+\#\.]*", text.lower())
    filtered = [w for w in words if len(w) > 3 and w not in _STOPWORDS]
    counts = Counter(filtered)
    return {w for w, _ in counts.most_common(top_n)}


def _bullet_score(bullet: str, keywords: set[str]) -> int:
    words = set(re.findall(r"[a-z][a-z0-9\+\#\.]*", bullet.lower()))
    return len(words & keywords)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9\+\#\.]*", text.lower()))


def _matched_keywords(keywords: set[str], profile: Profile) -> list[str]:
    """JD keywords that also appear in the candidate's own skills/technologies.

    Safe to echo in generated text — these are real, existing terms from the
    profile KB, not inserted from the job description.
    """
    candidate_terms = profile.skills[:]
    for exp in profile.experiences:
        candidate_terms.extend(exp.technologies)
    for proj in profile.projects:
        candidate_terms.extend(proj.technologies)

    display_by_token: dict[str, str] = {}
    for term in candidate_terms:
        for token in _tokenize(term):
            display_by_token.setdefault(token, term)

    matched, seen = [], set()
    for token in sorted(keywords & display_by_token.keys()):
        term = display_by_token[token]
        if term not in seen:
            seen.add(term)
            matched.append(term)
    return matched


def keyword_tailor(job: Job, profile: Profile, max_bullets_per_role: int = 3, max_projects: int = 3) -> dict:
    """
    Select and rank bullets from the profile KB that best match the JD keywords.
    Returns the same shape as TailoringAgent.tailor() so the UI can use either.
    """
    keywords = _extract_keywords(f"{job.title} {job.description}", top_n=40)

    experiences = []
    for exp in profile.experiences:
        scored = sorted(
            [(b, _bullet_score(b, keywords)) for b in exp.bullets],
            key=lambda x: x[1],
            reverse=True,
        )
        selected = [b for b, score in scored[:max_bullets_per_role] if score > 0]
        if not selected:
            selected = exp.bullets[:max_bullets_per_role]
        experiences.append({
            "company": exp.company,
            "title": exp.title,
            "dates": f"{exp.dates.start}–{exp.dates.end or 'Present'}",
            "selected_bullets": selected,
            "technologies": exp.technologies,
        })

    proj_scored = []
    for proj in profile.projects:
        all_text = f"{proj.name} {proj.description} {' '.join(proj.bullets)}"
        score = _bullet_score(all_text, keywords)
        proj_scored.append((proj, score))

    proj_scored.sort(key=lambda x: x[1], reverse=True)
    projects = []
    for proj, _ in proj_scored[:max_projects]:
        bullet_scored = sorted(
            [(b, _bullet_score(b, keywords)) for b in proj.bullets],
            key=lambda x: x[1], reverse=True,
        )
        selected = [b for b, s in bullet_scored[:2] if s > 0] or proj.bullets[:2]
        projects.append({
            "name": proj.name,
            "description": proj.description,
            "selected_bullets": selected,
            "technologies": proj.technologies,
        })

    jd_keywords_found = sorted(keywords)[:15]
    return {
        "experiences": experiences,
        "projects": projects,
        "matched_keywords": _matched_keywords(keywords, profile),
        "tailoring_notes": f"Keyword-matched against: {', '.join(jd_keywords_found)}",
        "method": "keyword",
    }


def build_cover_letter(job: Job, profile: Profile, tailored: dict) -> str:
    """Template-based cover letter — grounded in profile, no LLM needed."""
    top_exp = tailored["experiences"][0] if tailored["experiences"] else None
    top_proj = tailored["projects"][0] if tailored["projects"] else None

    top_bullet = ""
    if top_exp and top_exp["selected_bullets"]:
        top_bullet = top_exp["selected_bullets"][0]

    # Prioritize skills/technologies that genuinely match this JD's own language,
    # then fill remaining slots with the rest of the profile's skills.
    matched = tailored.get("matched_keywords") or []
    remaining = [s for s in profile.skills if s not in matched]
    top_tech = ", ".join((matched + remaining)[:6])

    edu = profile.education[0] if profile.education else None
    edu_line = (
        f"{edu.degree} in {edu.field} from {edu.institution}"
        if edu else "graduate studies in AI"
    )

    proj_line = ""
    if top_proj:
        proj_line = (
            f"Most recently, I built {top_proj['name']} — "
            f"{top_proj['description'].rstrip('.')}."
        )

    # Pulled from profile.work_auth (never hardcoded) — the locked KB is the
    # only source for this claim, same rule as everywhere else in the app.
    wa = profile.work_auth
    if wa.authorized_to_work_us:
        visa_note = f" on {wa.visa_type}" if wa.visa_type else ""
        work_auth_line = f"I am authorized to work in the US{visa_note}"
    else:
        work_auth_line = "I would need work authorization to take this role"
    work_auth_line += (
        f" and am eager to bring my skills in AI, machine learning, and software "
        f"engineering to {job.company}. I would welcome the opportunity to discuss "
        f"how my background aligns with your team's needs."
    )

    cover = f"""Dear Hiring Team at {job.company},

I am writing to express my strong interest in the {job.title} role. As a {edu_line} candidate with hands-on experience in {top_tech}, I am excited by the opportunity to contribute to your team.

{f"In my previous role as {top_exp['title']} at {top_exp['company']}, I {top_bullet[0].lower() + top_bullet[1:] if top_bullet else 'delivered impactful results in an engineering context'}." if top_exp else ""}

{proj_line}

{work_auth_line}

Thank you for your consideration.

Sincerely,
{profile.name}
{profile.email} | {profile.linkedin or profile.github}

---
[NOTE: Review all sections marked with brackets before sending. Verify every claim matches your experience.]
"""
    return cover.strip()
