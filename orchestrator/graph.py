"""LangGraph state machine: search → match → tailor → draft → prefill → human review.

State transitions:
  INGEST  → MATCH  → [below threshold → DISCARD]
                    → TAILOR → DRAFT → PREFILL → HUMAN_REVIEW → DONE
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from candidate.kb import load_profile
from core.config import settings
from core.schemas import Job, Profile
from draft.agent import DraftAgent
from match.agent import MatchAgent
from tailor.agent import TailoringAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ApplicationState(TypedDict):
    job: Job
    profile: Profile
    match_score: float
    tailored: dict
    cover_letter: str
    answers: dict[str, str]
    prefill_data: dict
    human_approved: bool
    skip_reason: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def node_match(state: ApplicationState) -> ApplicationState:
    agent = MatchAgent(state["profile"], threshold=settings.match_score_threshold)
    job = agent.score(state["job"])
    state["job"] = job
    state["match_score"] = job.match_score or 0.0
    if not agent.is_worth_applying(job):
        state["skip_reason"] = f"Match score {job.match_score:.2f} below threshold {settings.match_score_threshold}"
    logger.info("Match: %s @ %s → %.2f", job.title, job.company, job.match_score or 0)
    return state


def node_tailor(state: ApplicationState) -> ApplicationState:
    agent = TailoringAgent()
    state["tailored"] = agent.tailor(state["job"])
    logger.info("Tailored resume for %s @ %s", state["job"].title, state["job"].company)
    return state


def node_draft(state: ApplicationState) -> ApplicationState:
    agent = DraftAgent()
    state["cover_letter"] = agent.draft_cover_letter(state["job"], state.get("tailored"))
    logger.info("Cover letter drafted for %s @ %s", state["job"].title, state["job"].company)
    return state


def node_human_review(state: ApplicationState) -> ApplicationState:
    # In the Streamlit UI this is a blocking checkpoint — the graph yields here
    # and waits for the human to approve/edit before continuing.
    # For CLI runs, auto-approve for inspection; always False in prod until UI wires this.
    state["human_approved"] = False
    logger.info("Waiting for human review: %s @ %s", state["job"].title, state["job"].company)
    return state


def node_discard(state: ApplicationState) -> ApplicationState:
    logger.info("Discarded: %s — %s", state["job"].title, state.get("skip_reason", ""))
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_match(state: ApplicationState) -> str:
    if state.get("skip_reason"):
        return "discard"
    return "tailor"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    g = StateGraph(ApplicationState)

    g.add_node("match", node_match)
    g.add_node("tailor", node_tailor)
    g.add_node("draft", node_draft)
    g.add_node("human_review", node_human_review)
    g.add_node("discard", node_discard)

    g.add_edge(START, "match")
    g.add_conditional_edges("match", route_after_match, {"tailor": "tailor", "discard": "discard"})
    g.add_edge("tailor", "draft")
    g.add_edge("draft", "human_review")
    g.add_edge("human_review", END)
    g.add_edge("discard", END)

    return g.compile()


# Compiled singleton — import this in the Streamlit app
application_graph = build_graph()


async def run_pipeline(job: Job) -> ApplicationState:
    """Entry point: run a single job through the full pipeline."""
    profile = load_profile()
    initial: ApplicationState = {
        "job": job,
        "profile": profile,
        "match_score": 0.0,
        "tailored": {},
        "cover_letter": "",
        "answers": {},
        "prefill_data": {},
        "human_approved": False,
        "skip_reason": "",
    }
    result = await application_graph.ainvoke(initial)
    return result
