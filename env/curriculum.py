"""Optional curriculum stages for harder POMDPs (more metric noise, stricter time)."""

from __future__ import annotations

from typing import Any, Optional


def noise_multiplier(stage: Optional[int]) -> float:
    if stage is None:
        return 1.0
    try:
        s = int(stage)
    except (TypeError, ValueError):
        return 1.0
    return {1: 1.0, 2: 1.22, 3: 1.48, 4: 1.75}.get(s, 1.0)


def max_steps_for_stage(base_max_steps: int, stage: Optional[int]) -> int:
    """Later stages slightly shorter episodes."""
    m = noise_multiplier(stage)
    if m <= 1.0:
        return base_max_steps
    if m >= 1.6:
        return max(12, base_max_steps - 6)
    return max(15, base_max_steps - 3)


def curriculum_reset_options(stage: Optional[int], base_options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    out = dict(base_options or {})
    if stage is not None:
        out["curriculum_stage"] = int(stage)
    return out
