"""Tailoring agent — reorders/selects resume bullets to match a JD.

Constraints baked in (non-negotiable):
- Never fabricates a bullet. Only reorders or selects from profile KB.
- Never removes a bullet that is stronger than any JD keyword match.
- Output is a selection + ordering of real bullets, not new text.
- One-page constraint respected (max_bullets_per_role enforced).
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic

from candidate import mcp_client
from core.schemas import Job

logger = logging.getLogger(__name__)


class TailoringAgent:
    def __init__(self) -> None:
        self._experiences, self._projects = mcp_client.get_tailoring_inventory()
        self._client = Anthropic()

    def tailor(self, job: Job, max_bullets_per_role: int = 3) -> dict:
        """
        Returns a dict of:
          experiences: [{company, title, selected_bullets: [str]}, ...]
          projects: [{name, selected_bullets: [str]}, ...]

        All bullets are verbatim from the profile — no generation.
        """
        # Build the bullet inventory for the prompt
        exp_inventory = [
            {
                "company": e["company"],
                "title": e["title"],
                "bullets": e["bullets"],
                "technologies": e["technologies"],
            }
            for e in self._experiences
        ]
        proj_inventory = [
            {
                "name": p["name"],
                "bullets": p["bullets"],
                "technologies": p["technologies"],
            }
            for p in self._projects
        ]

        system = (
            "You are a resume tailoring assistant. Read the full job description "
            "carefully — including any company overview/mission section — to "
            "understand what this specific company and role actually prioritize, "
            "then select and order the bullets that best demonstrate fit for THAT. "
            "You ONLY select and reorder bullets from the provided inventory — "
            "you must NEVER write new bullets, modify wording, or invent accomplishments. "
            "Return a JSON object matching the schema specified. "
            "If no bullets are relevant, return an empty selected_bullets list."
        )

        user = f"""Job: {job.title} at {job.company}
Full job description (may include a company overview/mission section — use it to judge fit):
{job.description[:4000]}

EXPERIENCE INVENTORY (select and reorder bullets, max {max_bullets_per_role} per role):
{json.dumps(exp_inventory, indent=2)}

PROJECTS INVENTORY (select most relevant, up to 3 projects):
{json.dumps(proj_inventory, indent=2)}

Return JSON only:
{{
  "experiences": [
    {{"company": "...", "title": "...", "selected_bullets": ["exact bullet text", ...]}}
  ],
  "projects": [
    {{"name": "...", "selected_bullets": ["exact bullet text", ...]}}
  ],
  "tailoring_notes": "brief reasoning (not shown to recruiter)"
}}"""

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            result = json.loads(raw)
            self._validate_no_fabrication(result)
            self._enrich_with_metadata(result)
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Tailor agent returned unparseable output: %s — %s", exc, raw[:200])
            return self._fallback_result(max_bullets_per_role)

    def _enrich_with_metadata(self, result: dict) -> None:
        """Merge dates/technologies/description back in from the MCP inventory —
        the LLM's selection JSON only carries company/title/name + bullets."""
        exp_by_key = {(e["company"], e["title"]): e for e in self._experiences}
        for exp in result.get("experiences", []):
            src = exp_by_key.get((exp.get("company"), exp.get("title")))
            if src:
                exp["dates"] = f"{src['dates']['start']}–{src['dates'].get('end') or 'Present'}"
                exp["technologies"] = src.get("technologies", [])

        proj_by_name = {p["name"]: p for p in self._projects}
        for proj in result.get("projects", []):
            src = proj_by_name.get(proj.get("name"))
            if src:
                proj["description"] = src.get("description", "")
                proj["technologies"] = src.get("technologies", [])

    def _fallback_result(self, max_bullets_per_role: int) -> dict:
        experiences = [
            {
                "company": e["company"],
                "title": e["title"],
                "dates": f"{e['dates']['start']}–{e['dates'].get('end') or 'Present'}",
                "selected_bullets": e["bullets"][:max_bullets_per_role],
                "technologies": e.get("technologies", []),
            }
            for e in self._experiences
        ]
        projects = [
            {
                "name": p["name"],
                "description": p.get("description", ""),
                "selected_bullets": p["bullets"][:2],
                "technologies": p.get("technologies", []),
            }
            for p in self._projects
        ]
        return {"experiences": experiences, "projects": projects, "tailoring_notes": "fallback (unparseable LLM output)"}

    def _validate_no_fabrication(self, result: dict) -> None:
        """Raise if any selected bullet doesn't exist verbatim in the profile."""
        all_real_bullets = set()
        for exp in self._experiences:
            all_real_bullets.update(exp["bullets"])
        for proj in self._projects:
            all_real_bullets.update(proj["bullets"])

        for exp in result.get("experiences", []):
            for bullet in exp.get("selected_bullets", []):
                if bullet not in all_real_bullets:
                    raise ValueError(f"Fabricated bullet detected: '{bullet[:80]}...'")

        for proj in result.get("projects", []):
            for bullet in proj.get("selected_bullets", []):
                if bullet not in all_real_bullets:
                    raise ValueError(f"Fabricated bullet detected: '{bullet[:80]}...'")
