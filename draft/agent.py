"""Cover letter and answer drafting agent.

Hard rule: every claim must be grounded in the profile KB.
Anything the model tries to generate that isn't grounded gets flagged
for human review — it is never silently included.
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic

from core.schemas import Job
from candidate import mcp_client


logger = logging.getLogger(__name__)

_GROUNDING_FLAG = "[NEEDS HUMAN REVIEW — not grounded in profile KB]"


class DraftAgent:
    def __init__(self) -> None:
        self._profile_text = mcp_client.get_profile_summary()
        self._client = Anthropic()

    def draft_cover_letter(self, job: Job, tailored: dict | None = None) -> str:
        """Draft a cover letter grounded in the profile KB.

        Reads the job description like a candidate doing real research: pulls
        out what the company actually builds/cares about (its overview/mission
        section, if present), then draws explicit, concrete connections
        between that and the candidate's real work — not generic enthusiasm.
        """
        system = (
            "You are writing a cover letter for the candidate below, for a specific "
            "job posting. Read the FULL job description carefully, including any "
            "company overview, mission, or 'About [Company]' section — treat this "
            "like real research into what the company actually builds and cares "
            "about, not boilerplate. Write a letter that is genuinely specific to "
            "THIS company and THIS role:\n"
            "1. Open with a specific, researched observation about the company "
            "(what it builds, its mission, or its technical focus, drawn from the "
            "job posting) tied to authentic enthusiasm — never generic phrases "
            "like 'I am excited about the opportunity' or 'I would be a great fit'.\n"
            "2. Draw an explicit, concrete connection between the candidate's most "
            "relevant real project or experience and the company's specific work "
            "or the role's stated responsibilities — name the actual project/role "
            "and explain HOW it connects, not just THAT it's relevant.\n"
            "3. A second concrete connection or relevant strength, same standard.\n"
            "4. A brief, confident close.\n\n"
            "You may ONLY reference facts, projects, skills, and accomplishments that "
            "appear verbatim in the PROFILE section — never invent experience. If you "
            f"cannot ground a claim in the profile, write '{_GROUNDING_FLAG}' in its place. "
            "Tone: genuine and specific, not salesy. 4 short paragraphs total."
        )

        user = f"""PROFILE:
{self._profile_text}

JOB: {job.title} at {job.company}
Full job description (look for a company overview/mission section and use it):
{job.description[:4000]}

Write the cover letter now. Do NOT invent experience. Do NOT use generic filler —
every sentence should be specific to this company and this role."""

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1100,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text.strip()
        flagged_count = text.count(_GROUNDING_FLAG)
        if flagged_count:
            logger.warning("%d ungrounded claim(s) flagged in cover letter for %s", flagged_count, job.title)
        return text

    def draft_outreach_email(self, job: Job, recruiter_name: str = "") -> dict:
        """Draft a short cold-outreach email to a recruiter/hiring manager.

        Grounded ONLY in facts from the profile KB — never invents experience.
        Always for human review before sending; this method never sends anything.
        Returns {"subject": str, "body": str}.
        """
        greeting_name = recruiter_name.strip() or "Hiring Team"
        system = (
            "You write short, genuine cold-outreach emails to recruiters/hiring "
            "managers, grounded ONLY in facts from the PROFILE section below — "
            "never invent experience. Read the job description (including any "
            "company overview) and write a concise, specific email:\n"
            "1. A greeting to the named recipient.\n"
            "2. One or two sentences on why THIS role/company specifically "
            "interests you, grounded in the job description's own language — "
            "not generic enthusiasm.\n"
            "3. One or two sentences naming a specific real project or skill "
            "from the PROFILE and how it's relevant to this role.\n"
            "4. A brief, low-pressure ask (e.g., a quick chat, or to be "
            "considered for the role) and a sign-off.\n"
            f"If you cannot ground a claim, write '{_GROUNDING_FLAG}' instead. "
            "Keep the whole email under 150 words — recruiters skim. "
            'Return ONLY valid JSON: {"subject": "...", "body": "..."} — no markdown fences.'
        )
        user = f"""PROFILE:
{self._profile_text}

RECIPIENT NAME: {greeting_name}

JOB: {job.title} at {job.company}
Full job description (look for a company overview/mission section and use it):
{job.description[:4000]}

Write the outreach email now."""

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Outreach email JSON parse failed: %s — %s", exc, raw[:200])
            result = {"subject": f"Interest in {job.title} at {job.company}", "body": raw}

        if result.get("body", "").count(_GROUNDING_FLAG):
            logger.warning("Ungrounded claim(s) flagged in outreach email for %s", job.title)
        return result

    def draft_answer(self, question: str, job: Job) -> str:
        """Draft an answer to a free-form application question, grounded in the profile."""
        # First check if it's a work-auth question — answer verbatim
        verbatim = mcp_client.get_work_auth_answer(question)
        if verbatim:
            return verbatim

        system = (
            "You are answering an application question on behalf of the candidate. "
            "Ground every claim in the PROFILE. "
            f"Mark anything you cannot ground as '{_GROUNDING_FLAG}'. "
            "Be concise — 2–4 sentences unless the question demands more."
        )

        user = f"""PROFILE:
{self._profile_text}

JOB CONTEXT: {job.title} at {job.company}

QUESTION: {question}

Answer:"""

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
