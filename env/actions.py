"""Runbook-style names map to canonical action types (implementation stays Python)."""

from __future__ import annotations

# Aliases accepted from humans, LLMs, or external tools — normalized in IncidentResponseEnv.step
ALIASES: dict[str, str] = {
    "restart_pod": "restart_service",
    "kubectl_rollout_restart": "restart_service",
    "kubectl_restart_deployment": "restart_service",
    "kubectl_rollout_undo": "rollback_deploy",
    "kubectl_undo_rollout": "rollback_deploy",
    "scale_deployment": "scale_up",
    "kubectl_scale_deployment": "scale_up",
    "kubectl_logs": "check_logs",
    "kubectl_top_pods": "check_logs",
    "stern_tail": "check_logs",
    "tail_logs": "check_logs",
    "open_trace": "check_logs",
}

RUNBOOK_LABELS: dict[str, str] = {
    "check_logs": "kubectl logs --tail=200 / tail",
    "diagnose": "diagnose (commit root cause + mode)",
    "restart_service": "kubectl rollout restart deployment",
    "rollback_deploy": "kubectl rollout undo deployment",
    "scale_up": "kubectl scale deployment — more replicas",
    "enable_circuit_breaker": "mesh / breaker — isolate dependency",
    "defer": "defer — gather more data (safe hold)",
    "no_op": "no_op",
}

_KNOWN = frozenset(
    {
        "check_logs",
        "diagnose",
        "restart_service",
        "rollback_deploy",
        "scale_up",
        "enable_circuit_breaker",
        "defer",
        "no_op",
    }
)


def normalize_action(action: dict | None) -> dict:
    if not action or not isinstance(action, dict):
        return {"type": "no_op"}
    raw = action.get("type", "no_op")
    if not isinstance(raw, str):
        raw = "no_op"
    mapped = ALIASES.get(raw, raw)
    if mapped not in _KNOWN:
        return {"type": "no_op", "_unknown": raw}
    out = dict(action)
    out["type"] = mapped
    return out


def runbook_dropdown_choices() -> list[tuple[str, str]]:
    """(label, canonical_type) for Gradio Dropdown."""
    order = [
        "check_logs",
        "diagnose",
        "restart_service",
        "rollback_deploy",
        "scale_up",
        "enable_circuit_breaker",
        "defer",
        "no_op",
    ]
    return [(RUNBOOK_LABELS.get(c, c), c) for c in order]
