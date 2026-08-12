"""Sync MCP client for the Gmail outreach server.

Mirrors candidate/mcp_client.py's pattern: spawn mcp_servers/gmail_server.py
over stdio, call its tool, return the parsed result.
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
    args=["-m", "mcp_servers.gmail_server"],
    cwd=str(_REPO_ROOT),
)


async def _call_tool(name: str, arguments: dict[str, Any]) -> str:
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return result.content[0].text


def send_email(to_email: str, subject: str, body: str) -> dict:
    """Returns {"sent": True, "to": ...} or {"sent": False, "error": ...}."""
    raw = asyncio.run(_call_tool("send_email", {
        "to_email": to_email, "subject": subject, "body": body,
    }))
    return json.loads(raw)
