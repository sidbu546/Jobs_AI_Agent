"""Bullet rewriting agent — rephrases already-selected, verified-true bullets
to echo a job description's own terminology, for ATS keyword alignment.

Hard rule: rewriting may change WORDING ONLY. It must never change the scope,
technologies, numbers, or claims of the original bullet. If a rewrite
introduces a number that wasn't in the original, it is discarded and the
original bullet is used instead — this is enforced in code, not just prompted.
"""

from __future__ import annotations

import json
import logging
import re

from anthropic import Anthropic

from core.schemas import Job


logger = logging.getLogger(__name__)


def _numbers(text: str) -> set[str]:
    """Numeric tokens (30%, 100+, 45%, 12, ...) — used as a fabrication guard."""
    return set(re.findall(r"\d+[%\+]?", text))


class RewriteAgent:
    def __init__(self) -> None:
        self._client = Anthropic()

    def rewrite_bullets(self, bullets: list[str], job: Job) -> list[str]:
        """Rewrite each bullet's phrasing to echo the JD's own terminology.

        Falls back to the original bullet, unchanged, per-bullet, if the
        rewrite introduces a number that wasn't in the original, or if the
        call fails or returns an unusable shape.
        """
        if not bullets:
            return []

        system = (
            "You rewrite resume bullets so their PHRASING echoes a target job "
            "description's own terminology — this helps the resume pass "
            "applicant-tracking-system (ATS) keyword scans.\n\n"
            "Rules:\n"
            "1. You may freely change word choice, synonyms, sentence structure, "
            "and emphasis — this is expected for most bullets, not a rare exception.\n"
            "2. You must NEVER change, add, or remove any number, percentage, "
            "technology name, or the underlying scope/claim of a bullet. Every "
            "fact in the rewrite must be identical to the original — only the "
            "words describing it may change.\n"
            "3. Only leave a bullet completely unchanged if the job description "
            "shares no relevant terminology with that bullet's subject matter at "
            "all. Do not default to 'unchanged' out of caution when a safe, "
            "fact-preserving rewording is available — rewrite whenever you can.\n\n"
            "Example: bullet = 'Optimized MySQL database performance through "
            "schema restructuring, indexing, and query tuning, reducing database "
            "load by 45%.' If the JD emphasizes 'system performance' and "
            "'scalability', a good rewrite is: 'Improved system performance and "
            "scalability by restructuring MySQL schemas, adding indexes, and "
            "tuning queries, cutting database load by 45%.' — identical facts "
            "(MySQL, schema restructuring, indexing, query tuning, 45%), "
            "different phrasing.\n\n"
            "Return ONLY a JSON array of strings, same length and order as the input — no markdown fences."
        )
        user = f"""JOB DESCRIPTION:
{job.title} at {job.company}
{job.description[:4000]}

BULLETS TO REWRITE (JSON array, {len(bullets)} items):
{json.dumps(bullets, indent=2)}

Return ONLY a JSON array of {len(bullets)} rewritten bullets, same order."""

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1536,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            rewritten = json.loads(raw)
        except Exception as exc:
            logger.warning("Bullet rewrite failed, using originals: %s", exc)
            return bullets

        if not isinstance(rewritten, list) or len(rewritten) != len(bullets):
            logger.warning("Bullet rewrite returned unexpected shape, using originals")
            return bullets

        result = []
        for original, candidate in zip(bullets, rewritten):
            if not isinstance(candidate, str) or not candidate.strip():
                result.append(original)
                continue
            if _numbers(candidate) != _numbers(original):
                logger.warning("Rewrite altered numbers for a bullet — reverting to original")
                result.append(original)
            else:
                result.append(candidate)
        return result
