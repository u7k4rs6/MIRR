#!/usr/bin/env python3
"""
Post today's challenge to Slack Incoming Webhook (optional).

Requires: SLACK_WEBHOOK_URL
Optional: HF_SPACE_URL (link shown in the message)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.daily import daily_challenge_banner, utc_date_string  # noqa: E402
from env.scenarios import daily_rotation_choices  # noqa: E402


def main() -> int:
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        print("Set SLACK_WEBHOOK_URL to a Slack Incoming Webhook URL.", file=sys.stderr)
        return 1
    pairs = daily_rotation_choices()
    seed, sid, md = daily_challenge_banner([(t, i) for t, i in pairs])
    space = os.environ.get("HF_SPACE_URL", "").strip()
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "MIRR — daily incident challenge"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": md.replace("**", "*")}},
    ]
    if space:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{space}|Open incident lab>"},
            }
        )
    payload = json.dumps({"blocks": blocks}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        _ = resp.read()
    print(f"Posted daily challenge for UTC {utc_date_string()} seed={seed} scenario={sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
