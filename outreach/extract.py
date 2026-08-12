"""Extract contact emails that are literally present in job posting text.

This never invents or looks up an email — it only surfaces addresses the
company itself already published in the posting.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Job boards' own noreply/system addresses — not useful outreach targets.
_PLATFORM_DOMAINS = {
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "myworkday.com",
    "example.com",
}


def find_emails_in_text(text: str) -> list[str]:
    if not text:
        return []
    found = {m.lower() for m in _EMAIL_RE.findall(text)}
    filtered = {e for e in found if e.split("@")[-1] not in _PLATFORM_DOMAINS}
    return sorted(filtered)
