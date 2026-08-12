"""Streamlit dashboard — keyword search → matched jobs → tailor → cover letter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # must run before the imports below — core/storage.py reads env vars at import time

from candidate.kb import load_profile  # noqa: E402
from core.config import settings  # noqa: E402
from core.storage import SQLiteJobStore  # noqa: E402
from ingestion.greenhouse import GreenhouseAdapter  # noqa: E402
from ingestion.jobicy import JobicyAdapter  # noqa: E402
from ingestion.lever import LeverAdapter  # noqa: E402
from ingestion.remotive import RemotiveAdapter  # noqa: E402
from match.agent import MatchAgent  # noqa: E402
from tailor.keyword_tailor import build_cover_letter, keyword_tailor  # noqa: E402

st.set_page_config(page_title="Jobs AI Agent", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Jobs AI Agent")
st.sidebar.markdown("---")

search_query = st.sidebar.text_input(
    "Job title / keywords",
    value="AI engineer machine learning NLP",
    help="Try: 'ML engineer Python LLM', 'NLP research engineer', 'data scientist deep learning'",
)

job_mode = st.sidebar.radio(
    "Experience level",
    ["Full-time (entry/mid)", "Internship"],
    index=0,
)

threshold = st.sidebar.slider(
    "Match threshold",
    min_value=0.10, max_value=0.90,
    value=settings.match_score_threshold,
    step=0.05,
    help="0.65+ = Strong  |  0.40–0.65 = Medium  |  < 0.40 = Low",
)

max_results = st.sidebar.slider("Max jobs to fetch", 20, 150, 100, step=10)

_has_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
if not _has_llm:
    st.sidebar.caption("Local mode — keyword tailoring active. Add ANTHROPIC_API_KEY for LLM drafting.")
else:
    st.sidebar.caption("AI mode — Claude rewrites resume bullets + cover letter for this job's language.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = SQLiteJobStore()
if "profile" not in st.session_state:
    st.session_state.profile = load_profile()
if "active_job" not in st.session_state:
    st.session_state.active_job = None

store: SQLiteJobStore = st.session_state.store
profile = st.session_state.profile

# ---------------------------------------------------------------------------
# Helper — MUST be defined before any tab references it
# ---------------------------------------------------------------------------

def _score_label(score: float) -> tuple[str, str]:
    if score >= 0.65:
        return "green", f"Strong · {int(score*100)}%"
    elif score >= 0.40:
        return "orange", f"Medium · {int(score*100)}%"
    else:
        return "red", f"Low · {int(score*100)}%"


def _build_resume_text(profile, tailored: dict) -> str:
    """Plain-text tailored resume, editable in the UI and safe to download as-is."""
    lines: list[str] = [profile.name]
    contact = " | ".join(filter(None, [profile.email, profile.phone, profile.location]))
    if contact:
        lines.append(contact)
    links = " | ".join(filter(None, [profile.linkedin, profile.github, profile.portfolio]))
    if links:
        lines.append(links)
    lines.append("")

    matched = tailored.get("matched_keywords")
    if matched:
        lines.append(f"KEY MATCHES FOR THIS ROLE: {', '.join(matched)}")
        lines.append("")

    lines.append("EXPERIENCE")
    for exp in tailored["experiences"]:
        if not exp["selected_bullets"]:
            continue
        lines.append(f"{exp['title']} — {exp['company']} ({exp['dates']})")
        for b in exp["selected_bullets"]:
            lines.append(f"  • {b}")
        if exp["technologies"]:
            lines.append(f"  Tech: {', '.join(exp['technologies'])}")
        lines.append("")

    if tailored["projects"]:
        lines.append("PROJECTS")
        for proj in tailored["projects"]:
            lines.append(f"{proj['name']}: {proj['description']}")
            for b in proj["selected_bullets"]:
                lines.append(f"  • {b}")
            if proj["technologies"]:
                lines.append(f"  Tech: {', '.join(proj['technologies'])}")
            lines.append("")

    if profile.skills:
        lines.append("SKILLS")
        lines.append(", ".join(profile.skills))
        lines.append("")

    if profile.education:
        lines.append("EDUCATION")
        for edu in profile.education:
            end = edu.dates.end or "Present"
            lines.append(f"{edu.degree} in {edu.field} — {edu.institution} ({edu.dates.start}–{end})")

    return "\n".join(lines).strip() + "\n"


def _render_job_card(job, store, profile):
    color, label = _score_label(job.match_score or 0)
    header = f":{color}[●] **{job.title}** — {job.company} | {label}"

    with st.expander(header, expanded=False):
        c1, c2 = st.columns([3, 1])

        with c1:
            meta = []
            if job.location:
                meta.append(f"📍 {job.location}")
            if job.posted_at:
                meta.append(f"📅 {job.posted_at.strftime('%b %d, %Y')}")
            if job.employment_type.value != "unknown":
                meta.append(f"💼 {job.employment_type.value.replace('_', ' ').title()}")
            if meta:
                st.caption("  ·  ".join(meta))

        with c2:
            st.markdown(f"[Open Posting ↗]({job.url})")
            if st.button("✓ Mark Seen", key=f"seen_{job.dedup_key}", use_container_width=True):
                store.mark_seen(job.dedup_key)
                st.rerun()
            if st.button("Tailor & Draft →", key=f"tailor_{job.dedup_key}",
                         type="primary", use_container_width=True):
                st.session_state["active_job"] = job
                st.success("Job selected! Go to **Tailor & Draft** tab.")

        st.markdown("---")
        st.markdown("**Job Description**")
        desc = job.description or "_No description available._"
        if len(desc) > 1500:
            st.markdown(desc[:1500] + "…")
            with st.expander("Show full description"):
                st.markdown(desc)
        else:
            st.markdown(desc)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_discover, tab_review, tab_tailor, tab_profile = st.tabs(
    ["Discover", "Review Queue", "Tailor & Draft", "My Profile"]
)

# ── Discover ─────────────────────────────────────────────────────────────────

# Verified Greenhouse company slugs (AI/tech companies with public boards)
_GREENHOUSE_COMPANIES = [
    "anthropic", "databricks", "stripe", "airbnb", "waymo", "reddit",
    "deepmind", "scaleai", "togetherai", "figma", "vercel", "pinterest",
    "lyft", "coinbase", "robinhood", "brex", "xai", "speechmatics",
]

# Verified Lever company slugs
_LEVER_COMPANIES = [
    "mistral", "anyscale", "imbue",
]

with tab_discover:
    st.header("Job Discovery")

    is_internship = job_mode == "Internship"
    job_type_param = "internship" if is_internship else None
    mode_label = "internship (0–1 yr)" if is_internship else "full-time entry/mid"

    st.caption(
        f"Searching **Remotive + Jobicy + Greenhouse ({len(_GREENHOUSE_COMPANIES)} companies) + Lever** · "
        f"keywords: **{search_query}** · mode: **{mode_label}** · threshold: **{threshold}**"
    )

    run_btn = st.button("Search & Match All Sources", type="primary")

    if run_btn:
        if not search_query.strip():
            st.warning("Enter job title or keywords in the sidebar.")
        else:
            async def _collect_all() -> list:
                """Fetch from all sources concurrently, return raw job list."""
                all_jobs: list = []

                async def _from_remotive():
                    queries = [search_query, "machine learning engineer Python", "NLP LLM engineer"]
                    if is_internship:
                        queries = [search_query + " intern", search_query + " internship"]
                    seen: set[str] = set()

                    async def _one_remotive(q: str) -> None:
                        adapter = RemotiveAdapter(
                            query=q, limit=max(30, max_results // 3),
                            job_type=job_type_param, us_only=True,
                        )
                        async for j in adapter.fetch_jobs():
                            if j.dedup_key not in seen:
                                seen.add(j.dedup_key)
                                all_jobs.append(j)

                    import asyncio as _aio2
                    await _aio2.gather(*[_one_remotive(q) for q in queries])

                async def _from_jobicy():
                    # Jobicy accepts only single-word tags — multi-word queries return 404.
                    # These tags are pre-validated against the Jobicy API.
                    tags = ["llm", "python", "data", "research"]
                    if is_internship:
                        tags = ["python", "engineer", "developer"]

                    async def _one_jobicy(tag: str) -> None:
                        adapter = JobicyAdapter(query=tag, limit=50)
                        async for j in adapter.fetch_jobs():
                            all_jobs.append(j)

                    import asyncio as _aio3
                    await _aio3.gather(*[_one_jobicy(t) for t in tags])

                async def _from_greenhouse():
                    adapter = GreenhouseAdapter(companies=_GREENHOUSE_COMPANIES, timeout=30.0)
                    async for j in adapter.fetch_jobs():
                        all_jobs.append(j)

                async def _from_lever():
                    adapter = LeverAdapter(companies=_LEVER_COMPANIES, timeout=20.0)
                    async for j in adapter.fetch_jobs():
                        all_jobs.append(j)

                import asyncio as _aio
                await _aio.gather(
                    _from_remotive(),
                    _from_jobicy(),
                    _from_greenhouse(),
                    _from_lever(),
                )
                return all_jobs

            status_box = st.empty()
            status_box.info("Fetching from Remotive, Jobicy, Greenhouse, and Lever in parallel…")

            all_raw = asyncio.run(_collect_all())
            status_box.empty()

            # Deduplicate across all sources
            seen_keys: set[str] = set()
            unique_jobs = []
            for j in all_raw:
                if j.dedup_key not in seen_keys:
                    seen_keys.add(j.dedup_key)
                    unique_jobs.append(j)

            if not unique_jobs:
                st.warning("No results returned. Check internet connection or try different keywords.")
            else:
                st.info(f"Fetched **{len(all_raw)}** listings → **{len(unique_jobs)}** unique. Scoring…")

                agent = MatchAgent(profile, threshold=threshold)
                with st.spinner(f"Scoring {len(unique_jobs)} jobs against your profile…"):
                    scored_jobs = agent.score_batch(unique_jobs)

                new_count = matched_count = 0
                matched_jobs = []
                for scored in scored_jobs:
                    if store.upsert(scored):
                        new_count += 1
                    if agent.is_worth_applying(scored):
                        matched_count += 1
                        matched_jobs.append(scored)

                st.success(
                    f"Scored **{len(unique_jobs)}** jobs → **{new_count}** new in DB → "
                    f"**{matched_count}** above threshold **{threshold}**"
                )

                if matched_jobs:
                    st.info(f"Go to **Review Queue** tab to see all {matched_count} matched jobs.")
                    st.markdown("**Top matches this run:**")
                    top = sorted(matched_jobs, key=lambda x: x.match_score or 0, reverse=True)[:10]
                    for j in top:
                        color, label = _score_label(j.match_score or 0)
                        st.markdown(
                            f"- :{color}[**{j.title}**] @ {j.company} — {label}  ·  "
                            f"📍{j.location}  ·  [Apply ↗]({j.url})"
                        )
                else:
                    st.warning(
                        f"No jobs hit threshold **{threshold}** this run. "
                        "Try lowering it in the sidebar, or use broader keywords."
                    )


# ── Review Queue ─────────────────────────────────────────────────────────────
with tab_review:
    st.header("Review Queue")

    col_ctrl1, col_ctrl2 = st.columns([4, 1])
    with col_ctrl2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    unseen = store.unseen_above_threshold(threshold)

    if not unseen:
        st.info(
            "No matched jobs yet. Run a search from **Discover**, "
            "or lower the threshold slider in the sidebar."
        )
    else:
        strong = [j for j in unseen if (j.match_score or 0) >= 0.65]
        medium = [j for j in unseen if 0.40 <= (j.match_score or 0) < 0.65]
        low    = [j for j in unseen if (j.match_score or 0) < 0.40]

        with col_ctrl1:
            st.caption(
                f"**{len(unseen)} jobs** waiting · "
                f":green[{len(strong)} strong] · "
                f":orange[{len(medium)} medium] · "
                f":red[{len(low)} low]"
            )

        if strong:
            st.markdown(f"### 🟢 Strong matches — {len(strong)}")
            for job in sorted(strong, key=lambda j: j.match_score or 0, reverse=True):
                _render_job_card(job, store, profile)

        if medium:
            st.markdown(f"### 🟠 Medium matches — {len(medium)}")
            for job in sorted(medium, key=lambda j: j.match_score or 0, reverse=True):
                _render_job_card(job, store, profile)

        if low and threshold <= 0.35:
            st.markdown(f"### 🔴 Low matches — {len(low)}")
            for job in sorted(low, key=lambda j: j.match_score or 0, reverse=True):
                _render_job_card(job, store, profile)


# ── Tailor & Draft ───────────────────────────────────────────────────────────
with tab_tailor:
    st.header("Tailor & Draft")

    active_job = st.session_state.get("active_job")

    if active_job is None:
        st.info("Go to **Review Queue**, open any job, and click **'Tailor & Draft →'**.")
    else:
        color, label = _score_label(active_job.match_score or 0)
        st.markdown(f":{color}[●] **{active_job.title}** @ **{active_job.company}** — {label}")
        st.markdown(f"[Open Job Posting ↗]({active_job.url})")
        st.markdown("---")

        stale = st.session_state.get("tailored_for_job") != active_job.dedup_key
        regen_clicked = st.button(
            "Regenerate" if not stale else "Generate Tailored Resume + Cover Letter",
            type="primary",
        )

        if stale or regen_clicked:
            spinner_msg = (
                "Rewriting your resume + cover letter for this job with Claude…"
                if _has_llm else "Tailoring your resume to this job…"
            )
            with st.spinner(spinner_msg):
                # Cheap local pass — always computed, used for the "matched
                # keywords" evidence shown in Match Notes regardless of mode.
                kw_tailored = keyword_tailor(active_job, profile)

                if _has_llm:
                    from draft.agent import DraftAgent
                    from tailor.agent import TailoringAgent
                    from tailor.rewrite_agent import RewriteAgent

                    # Real selection: Claude reads the full JD (incl. any company
                    # overview) and picks/orders real bullets by genuine relevance,
                    # instead of raw keyword-frequency overlap.
                    tailored = TailoringAgent().tailor(active_job)
                    tailored["matched_keywords"] = kw_tailored["matched_keywords"]

                    all_bullets, slots = [], []
                    for exp in tailored["experiences"]:
                        for i in range(len(exp.get("selected_bullets", []))):
                            all_bullets.append(exp["selected_bullets"][i])
                            slots.append((exp["selected_bullets"], i))
                    for proj in tailored["projects"]:
                        for i in range(len(proj.get("selected_bullets", []))):
                            all_bullets.append(proj["selected_bullets"][i])
                            slots.append((proj["selected_bullets"], i))

                    if all_bullets:
                        rewritten = RewriteAgent().rewrite_bullets(all_bullets, active_job)
                        for (container, i), new_bullet in zip(slots, rewritten):
                            container[i] = new_bullet
                    tailored["method"] = "Claude-select + Claude-rewrite"

                    cover = DraftAgent().draft_cover_letter(active_job, tailored)
                else:
                    tailored = kw_tailored
                    cover = build_cover_letter(active_job, profile, tailored)
            st.session_state["tailored"] = tailored
            st.session_state["cover"] = cover
            st.session_state["tailored_for_job"] = active_job.dedup_key

        if "tailored" in st.session_state and st.session_state.tailored:
            tailored = st.session_state["tailored"]
            cover = st.session_state["cover"]

            sub_resume, sub_cover, sub_outreach, sub_notes = st.tabs(
                ["Tailored Resume", "Cover Letter", "Contact Recruiter", "Match Notes"]
            )

            with sub_resume:
                if tailored.get("method", "keyword") == "keyword":
                    st.warning("Review every bullet before sending. Edit directly in the box below.")
                else:
                    st.warning(
                        "Bullets were rewritten by Claude to echo this job's language — "
                        "facts/numbers are guarded, but review before sending. Edit directly below."
                    )
                resume_text = _build_resume_text(profile, tailored)
                edited_resume = st.text_area(
                    "Tailored resume (edit freely)",
                    value=resume_text,
                    height=500,
                    key=f"resume_text_{active_job.dedup_key}",
                )
                st.download_button(
                    "Download tailored resume (.txt)",
                    data=edited_resume,
                    file_name=f"resume_{active_job.company.replace(' ','_').replace('/','_')}.txt",
                    mime="text/plain",
                )

            with sub_cover:
                st.warning("Review every sentence before sending. Edit anything in [brackets].")
                edited = st.text_area(
                    "Cover letter (edit freely)",
                    value=cover,
                    height=500,
                    key=f"cover_text_{active_job.dedup_key}",
                )
                st.download_button(
                    "Download cover letter (.txt)",
                    data=edited,
                    file_name=f"cover_{active_job.company.replace(' ','_').replace('/','_')}.txt",
                    mime="text/plain",
                )

            with sub_outreach:
                from outreach.company_lookup import (
                    find_published_contact_email,
                    linkedin_search_url,
                    resolve_company_domain,
                )
                from outreach.extract import find_emails_in_text

                st.markdown("#### Contact a recruiter or hiring manager")
                st.caption(
                    "No recruiter contact info exists in job board APIs, and this won't "
                    "scrape LinkedIn or guess email addresses — but here's a fast way to "
                    "find and verify a real one."
                )

                job_key = active_job.dedup_key
                name_widget_key = f"rec_name_{job_key}"
                email_widget_key = f"rec_email_{job_key}"

                # Seed once per job from any email already published in the JD text.
                # After that, only an explicit action (lookup button) or the user's
                # own typing changes it — never overwritten on a plain rerun.
                if email_widget_key not in st.session_state:
                    jd_emails = find_emails_in_text(active_job.description)
                    st.session_state[email_widget_key] = jd_emails[0] if jd_emails else ""
                    if jd_emails:
                        st.info(f"Found in this posting: {', '.join(jd_emails)}")

                li_url = linkedin_search_url(active_job.company)
                st.markdown(f"🔍 [Search LinkedIn for recruiters at {active_job.company} ↗]({li_url})")

                domain, is_guess = resolve_company_domain(active_job.company, active_job.url)
                if domain:
                    label = f"Check {domain} (guessed) for a published contact email" if is_guess \
                        else f"Check {domain} for a published contact email"
                    if st.button(label, key=f"lookup_{job_key}"):
                        with st.spinner(f"Checking {domain}…"):
                            site_emails = find_published_contact_email(domain)
                        if site_emails:
                            st.session_state[email_widget_key] = site_emails[0]
                            st.success(f"Found on {domain}: {', '.join(site_emails)}")
                        else:
                            st.warning(
                                "No published contact email found there — try the search links above, "
                                "or enter one manually below."
                            )

                st.markdown("##### Compose")
                c1, c2 = st.columns(2)
                with c1:
                    recruiter_name = st.text_input(
                        "Recruiter / hiring manager name (optional)", key=name_widget_key
                    )
                with c2:
                    recruiter_email = st.text_input(
                        "Recruiter / hiring manager email", key=email_widget_key
                    )

                draft_key = f"outreach_draft_{job_key}"
                subj_widget_key = f"subj_{job_key}"
                body_widget_key = f"body_{job_key}"

                if st.button("Draft outreach email", key=f"draft_btn_{job_key}"):
                    if _has_llm:
                        with st.spinner("Drafting outreach email with Claude…"):
                            from draft.agent import DraftAgent
                            new_draft = DraftAgent().draft_outreach_email(
                                active_job, recruiter_name
                            )
                    else:
                        new_draft = {
                            "subject": f"Interest in {active_job.title} at {active_job.company}",
                            "body": (
                                f"Dear {recruiter_name or 'Hiring Team'},\n\n"
                                "[Add ANTHROPIC_API_KEY in .env to auto-draft this, "
                                "or write your own message here.]\n\n"
                                f"Best,\n{profile.name}"
                            ),
                        }
                    st.session_state[draft_key] = new_draft
                    # Widgets below are keyed and already rendered once this session —
                    # passing a new `value=` on rerun is ignored by Streamlit, so the
                    # draft has to be pushed into the widget's own state directly.
                    st.session_state[subj_widget_key] = new_draft.get("subject", "")
                    st.session_state[body_widget_key] = new_draft.get("body", "")

                subject = st.text_input("Subject", key=subj_widget_key)
                body = st.text_area(
                    "Email body (edit freely)", height=350, key=body_widget_key
                )

                st.warning("Review every sentence before sending — this goes directly to a real person.")

                already_sent = st.session_state.get(f"sent_{job_key}")
                if already_sent:
                    st.success(f"Already sent to {already_sent} in this session.")

                confirm = st.checkbox(
                    "I've reviewed this email and want to send it",
                    key=f"confirm_{job_key}",
                )
                can_send = bool(recruiter_email and subject and body and confirm)
                if st.button("Send Email via Gmail", disabled=not can_send, key=f"send_{job_key}"):
                    from outreach.mcp_client import send_email as mcp_send_email

                    with st.spinner("Sending…"):
                        result = mcp_send_email(recruiter_email, subject, body)
                    if result.get("sent"):
                        st.session_state[f"sent_{job_key}"] = recruiter_email
                        st.success(f"Sent to {recruiter_email}")
                    else:
                        st.error(f"Failed to send: {result.get('error')}")

            with sub_notes:
                st.caption(f"Method: {tailored.get('method', 'keyword')}")
                st.caption(f"Tailoring notes: {tailored.get('tailoring_notes', '')}")
                if tailored.get("matched_keywords"):
                    st.caption(
                        "Genuinely matched to your skills/technologies: "
                        + ", ".join(tailored["matched_keywords"])
                    )
                if not _has_llm:
                    st.caption("To have Claude rewrite bullets in this job's language, set ANTHROPIC_API_KEY in .env.")


# ── Profile ──────────────────────────────────────────────────────────────────
with tab_profile:
    st.header("My Profile")
    st.subheader(profile.name)
    st.caption(f"{profile.email} · {profile.location}")

    with st.expander("Work Authorization — LOCKED (verbatim only)"):
        st.warning(
            f"Visa: **{profile.work_auth.visa_type}** · "
            f"Authorized: {profile.work_auth.authorized_to_work_us} · "
            f"Requires sponsorship: {profile.work_auth.will_require_sponsorship}"
        )

    with st.expander("Experience", expanded=True):
        for exp in profile.experiences:
            st.markdown(f"**{exp.title}** @ {exp.company} ({exp.dates.start}–{exp.dates.end or 'Present'})")
            for b in exp.bullets:
                st.markdown(f"- {b}")

    with st.expander("Projects", expanded=True):
        for proj in profile.projects:
            st.markdown(f"**{proj.name}**: {proj.description}")
            for b in proj.bullets:
                st.markdown(f"- {b}")

    with st.expander("Skills"):
        st.write(", ".join(profile.skills))
