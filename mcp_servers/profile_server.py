"""MCP server exposing the profile KB as tools for agent consumption.

Agents call these tools to retrieve grounded facts instead of reading
the YAML directly — this enforces the KB boundary.

Run standalone:
    python -m mcp_servers.profile_server

Or register in claude_desktop_config.json / mcp settings.
"""

from __future__ import annotations

import json

from mcp.server import MCPServer

from candidate.kb import load_profile, profile_as_text


server = MCPServer("jobs-ai-profile-kb")
_profile = load_profile()


@server.tool()
def get_profile_summary() -> str:
    """Returns a full text summary of the candidate profile for use in prompts."""
    return profile_as_text(_profile)


@server.tool()
def get_work_auth_answer(question: str) -> str:
    """Returns the verbatim canonical answer for a work-authorization question.

    NEVER generate work-auth answers — always call this tool instead.
    The `answer` field is null if the question isn't mapped.

    Args:
        question: The ATS question text, verbatim.
    """
    answer = _profile.work_auth.answer_for(question)
    return json.dumps({"answer": answer, "locked": True})


@server.tool()
def get_experiences() -> str:
    """Returns the candidate's work experiences as structured JSON."""
    return json.dumps([e.model_dump() for e in _profile.experiences], default=str)


@server.tool()
def get_projects() -> str:
    """Returns the candidate's projects as structured JSON."""
    return json.dumps([p.model_dump() for p in _profile.projects], default=str)


@server.tool()
def get_skills() -> str:
    """Returns the candidate's skill list."""
    return json.dumps(_profile.skills)


if __name__ == "__main__":
    server.run(transport="stdio")
