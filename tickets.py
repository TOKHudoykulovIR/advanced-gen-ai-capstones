import os
import requests

from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")


def tool_create_github_ticket(name: str, email: str, title: str, description: str) -> dict:
    """
    Tool: create_ticket
    Creates a GitHub Issue.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {"ok": False, "error": "Missing GITHUB_TOKEN or GITHUB_REPO in .env"}

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    body = (
        f"**Customer name:** {name}\n"
        f"**Customer email:** {email}\n\n"
        f"---\n\n"
        f"{description}"
    )

    payload = {"title": title, "body": body}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        return {"ok": False, "error": f"GitHub API error {r.status_code}", "details": r.text}

    data = r.json()
    return {"ok": True, "issue_url": data.get("html_url"), "issue_number": data.get("number")}