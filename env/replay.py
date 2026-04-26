"""Compact episode traces for sharing, leaderboards, or training data."""

from __future__ import annotations

import copy
import json
from typing import Any, Optional


def summarize_obs(obs: dict) -> dict:
    """Strip bulky fields for replay JSON."""
    metrics = obs.get("metrics") or {}
    cpu_top = sorted(
            metrics.keys(),
            key=lambda s: float((metrics.get(s) or {}).get("cpu") or 0),
            reverse=True,
        )[:3]
    return {
        "step": obs.get("step"),
        "system_health_score": obs.get("system_health_score"),
        "recent_alerts": (obs.get("recent_alerts") or [])[:4],
        "metric_trend": obs.get("metric_trend"),
        "cpu_hot_services": cpu_top,
        "diagnosis_made": obs.get("diagnosis_made"),
    }


def append_step(trace: list[dict], action: dict, reward: float, obs_after: dict) -> None:
    lr = obs_after.get("last_action_result") or ""
    trace.append(
        {
            "action": action,
            "reward": round(float(reward), 4),
            "observation": summarize_obs(obs_after),
            "last_action_result": lr[:2000] if isinstance(lr, str) else str(lr)[:2000],
        }
    )


def build_episode_document(
    *,
    scenario_id: str,
    seed: Optional[int],
    trace: list[dict],
    outcome: Optional[str],
    total_reward: float,
    reveal: bool = True,
    root_cause: Optional[str] = None,
    failure_mode: Optional[str] = None,
    incident_cost: Optional[float] = None,
    explanation_score: Optional[float] = None,
    compound_legs: Optional[int] = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "version": 1,
        "scenario_id": scenario_id,
        "seed": seed,
        "total_reward": round(float(total_reward), 4),
        "outcome": outcome,
        "steps": trace,
    }
    if incident_cost is not None:
        doc["incident_cost"] = round(float(incident_cost), 4)
    if explanation_score is not None:
        doc["explanation_score"] = round(float(explanation_score), 4)
    if compound_legs is not None:
        doc["compound_legs"] = int(compound_legs)
    if reveal and root_cause is not None:
        doc["ground_truth"] = {"root_cause": root_cause, "failure_mode": failure_mode}
    return doc


def dumps_pretty(doc: dict) -> str:
    return json.dumps(doc, indent=2)


def parse_episode_document(text: str) -> dict[str, Any]:
    """Parse JSON from a pasted replay string."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Replay JSON must be an object")
    return data


def _clean_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"type": "no_op"}
    a = {k: v for k, v in action.items() if not str(k).startswith("_")}
    return a


def recompute_episode(doc: dict[str, Any], *, rich_ui: bool = False) -> dict[str, Any]:
    """
    Deterministically replay `steps` against the current simulator using doc seed + scenario_id.
    Returns a report dict (for UI or tests).
    """
    from env.environment import IncidentResponseEnv

    steps = doc.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")

    scenario_id = str(doc.get("scenario_id") or "surprise")
    seed = doc.get("seed")
    n = len(steps)
    env = IncidentResponseEnv(max_steps=max(30, n + 10))
    obs, _ = env.reset(
        seed=seed,
        options={"scenario_id": scenario_id, "rich_ui": rich_ui},
    )
    total_reward = 0.0
    health_series = [float(obs["system_health_score"])]
    metrics_hist: list[dict[str, Any]] = []

    metrics_hist.append(copy.deepcopy(obs.get("metrics") or {}))

    log_lines: list[str] = []
    done = False
    info: dict[str, Any] = {}
    for i, row in enumerate(steps):
        if not isinstance(row, dict):
            raise ValueError(f"steps[{i}] must be an object")
        action = _clean_action(row.get("action"))
        obs, reward, done, _, info = env.step(action)
        total_reward += float(reward)
        health_series.append(float(obs["system_health_score"]))
        metrics_hist.append(copy.deepcopy(obs.get("metrics") or {}))
        log_lines.append(f"step {i + 1}: {action} -> reward {reward:+.3f} health {obs['system_health_score']:.3f}")
        if done:
            break

    orig_out = doc.get("outcome")
    rep_out = info.get("outcome")
    orig_r = float(doc.get("total_reward") or 0.0)
    reward_close = abs(orig_r - total_reward) < 0.05 * max(1.0, abs(orig_r)) + 2.0
    outcome_match = orig_out == rep_out if orig_out and rep_out else False

    return {
        "ok": True,
        "steps_executed": min(len(steps), len(log_lines)),
        "episode_finished": bool(done),
        "original_outcome": orig_out,
        "replay_outcome": rep_out,
        "outcome_match": outcome_match,
        "original_total_reward": orig_r,
        "replay_total_reward": round(total_reward, 4),
        "reward_close": reward_close,
        "incident_cost": info.get("incident_cost"),
        "log_lines": log_lines,
        "health_series": health_series,
        "metrics_hist": metrics_hist,
        "ground_truth_doc": doc.get("ground_truth"),
        "final_info": {k: info[k] for k in ("root_cause", "failure_mode", "diagnosis_correct", "compound_legs") if k in info},
    }
