"""Sync MCP client for the profile knowledge-base server.

Spawns mcp_servers/profile_server.py as a subprocess over stdio and calls its
tools. Agents go through this instead of importing candidate.kb directly —
see the module docstring in mcp_servers/profile_server.py for why.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_REPO_ROOT = Path(__file__).parent.parent
_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_servers.profile_server"],
    cwd=str(_REPO_ROOT),
)


async def _call_tools(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            results = []
            for name, arguments in calls:
                result = await session.call_tool(name, arguments)
                results.append(result.content[0].text)
            return results


def call_tools(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Run one or more MCP tool calls against the profile server in a single session."""
    return asyncio.run(_call_tools(calls))


def get_profile_summary() -> str:
    return call_tools([("get_profile_summary", {})])[0]


def get_work_auth_answer(question: str) -> str | None:
    raw = call_tools([("get_work_auth_answer", {"question": question})])[0]
    return json.loads(raw)["answer"]


def get_tailoring_inventory() -> tuple[list[dict], list[dict]]:
    """Experiences and projects, as returned by the MCP profile server."""
    exp_raw, proj_raw = call_tools([("get_experiences", {}), ("get_projects", {})])
    return json.loads(exp_raw), json.loads(proj_raw)
