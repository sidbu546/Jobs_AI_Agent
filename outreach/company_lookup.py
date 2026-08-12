"""Best-effort, ToS-respecting recruiter discovery helpers.

Does NOT scrape LinkedIn or any protected platform, and does NOT guess or
fabricate email addresses. Two things only:
  1. Build search URLs for LinkedIn/Google so the user can manually find and
     verify a real recruiter — this automates nothing on the platform side,
     it just opens the search the user would type themselves.
  2. Fetch the company's OWN public website (one page, on explicit request)
     and look for a contact/recruiting email it has already published there.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus, urlparse

import httpx

from outreach.extract import find_emails_in_text

_ATS_DOMAIN_SUFFIXES = ("greenhouse.io", "lever.co", "ashbyhq.com", "myworkday.com")


def _is_ats_host(host: str) -> bool:
    return any(host == suffix or host.endswith("." + suffix) for suffix in _ATS_DOMAIN_SUFFIXES)


def linkedin_search_url(company: str) -> str:
    query = f"{company} recruiter OR \"talent acquisition\""
    return f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(query)}"


def resolve_company_domain(company: str, job_url: str) -> tuple[str | None, bool]:
    """Returns (domain, is_guess). Prefers the job URL's own domain when it's
    not an ATS-hosted board; otherwise guesses <slug>.com from the company name."""
    try:
        host = urlparse(job_url).netloc.lower()
    except Exception:
        host = ""
    if host and not _is_ats_host(host):
        return host, False

    slug = re.sub(r"[^a-z0-9]", "", company.lower())
    return (f"{slug}.com", True) if slug else (None, True)


def find_published_contact_email(company_domain: str, timeout: float = 5.0) -> list[str]:
    """Fetch the company's own /careers page (falling back to its homepage) and
    look for a published contact email. Best-effort — returns [] on any
    failure (unreachable domain, timeout, wrong guess), never raises."""
    if not company_domain:
        return []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; JobsAIAgent/1.0; +personal job search tool)"}
    for path in ("/careers", "/"):
        try:
            resp = httpx.get(
                f"https://{company_domain}{path}",
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            )
            if resp.status_code == 200:
                emails = find_emails_in_text(resp.text)
                if emails:
                    return emails
        except Exception:
            continue
    return []
