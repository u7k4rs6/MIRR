"""Deterministic daily challenge seed (UTC) for leaderboards and social posts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional


def utc_date_string(when: Optional[datetime] = None) -> str:
    dt = when or datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def daily_challenge_seed(date_str: Optional[str] = None) -> int:
    """Stable int 0..9999 for a calendar day (UTC)."""
    key = (date_str or utc_date_string()).encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    return int(h[:8], 16) % 10000


def daily_scenario_rotation_index(n_scenarios: int, date_str: Optional[str] = None) -> int:
    if n_scenarios <= 0:
        return 0
    h = hashlib.sha256((date_str or utc_date_string() + "|scenario").encode()).hexdigest()
    return int(h[:8], 16) % n_scenarios


def daily_challenge_banner(
    scenario_titles: list[tuple[str, str]],
    date_str: Optional[str] = None,
) -> tuple[int, str, str]:
    """
    Returns (seed, scenario_id, markdown blurb).
    scenario_titles: list of (title, id) excluding pure RNG if you want; we rotate over all.
    """
    d = date_str or utc_date_string()
    seed = daily_challenge_seed(d)
    if not scenario_titles:
        return seed, "surprise", f"**UTC {d}** · seed `{seed}` · scenario `surprise`"
    idx = daily_scenario_rotation_index(len(scenario_titles), d)
    _title, sid = scenario_titles[idx]
    return (
        seed,
        sid,
        f"**Daily challenge (UTC {d})**\n\n"
        f"- **Seed:** `{seed}`\n"
        f"- **Scenario:** `{sid}`\n\n"
        "Same problem for everyone today — compare rewards and step counts with friends.",
    )
