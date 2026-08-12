"""MCP server exposing a Gmail-send tool for agent/UI consumption.

Sends via Gmail's SMTP endpoint using an App Password — no OAuth app needed.

Setup (one-time, in your own Google account — this app cannot do it for you):
  1. Turn on 2-Step Verification: https://myaccount.google.com/security
  2. Create an App Password: https://myaccount.google.com/apppasswords
  3. Add to .env:
       GMAIL_ADDRESS=you@gmail.com
       GMAIL_APP_PASSWORD=<the 16-character app password, no spaces>

This tool performs a real send with no additional confirmation step of its
own — the calling agent/UI is responsible for getting human review before
invoking it.

Run standalone:
    python -m mcp_servers.gmail_server
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv()

server = MCPServer("jobs-ai-gmail")


@server.tool()
def send_email(to_email: str, subject: str, body: str) -> str:
    """Send a plain-text email from the user's own Gmail account.

    Args:
        to_email: Recipient email address.
        subject: Email subject line.
        body: Plain-text email body.
    """
    address = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")
    if not address or not app_password:
        return json.dumps({
            "sent": False,
            "error": "GMAIL_ADDRESS/GMAIL_APP_PASSWORD not set in .env — see this file's docstring.",
        })

    msg = EmailMessage()
    msg["From"] = address
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(address, app_password)
            smtp.send_message(msg)
    except Exception as exc:
        return json.dumps({"sent": False, "error": str(exc)})

    return json.dumps({"sent": True, "to": to_email})


if __name__ == "__main__":
    server.run(transport="stdio")
