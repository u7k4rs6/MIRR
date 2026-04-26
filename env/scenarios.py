"""Named incident scenarios — narrative wrapper over the same simulator."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    title: str
    tagline: str
    root_cause: Optional[str] = None
    failure_mode: Optional[str] = None
    # After the primary failure is fixed correctly, this secondary failure activates (same episode).
    compound_secondary: Optional[tuple[str, str]] = None
    # Extra alert lines mixed into paging (may mislead; not tied to true root).
    red_herring_alerts: tuple[str, ...] = ()
    curriculum_noise_mult: float = 1.0


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        id="surprise",
        title="Surprise incident",
        tagline="Classic drill: RNG picks the broken service and how it broke. No spoilers.",
    ),
    ScenarioSpec(
        id="flash_checkout",
        title="Flash sale — checkout melting",
        tagline="Marketing flipped a banner; edge traffic piles on the gateway. SREs see latency first, not the DB.",
        root_cause="api-gateway",
        failure_mode="overloaded",
        red_herring_alerts=(
            "cache: ephemeral hotkeys expiring (benign churn)",
            "worker: batch lag within SLO — triage later",
        ),
    ),
    ScenarioSpec(
        id="auth_memory_cliff",
        title="Auth pod memory cliff",
        tagline="Login succeeds then 502s creep in. Heap keeps growing until the node OOMs.",
        root_cause="auth-service",
        failure_mode="memory_leak",
    ),
    ScenarioSpec(
        id="db_ghost",
        title="Database ghost process",
        tagline="Replicas look fine on the dashboard until a worker stalls on a dead connection.",
        root_cause="database",
        failure_mode="crashed",
    ),
    ScenarioSpec(
        id="cache_stampede",
        title="Cache stampede",
        tagline="Hot keys expiring together; p99 explodes. Restart alone will not rebalance load.",
        root_cause="cache",
        failure_mode="overloaded",
    ),
    ScenarioSpec(
        id="worker_bad_cut",
        title="Friday deploy — worker cut bad",
        tagline="Canary looked green; batch worker picked up a bad config. Rollback is the real fix.",
        root_cause="worker",
        failure_mode="bad_deploy",
    ),
    ScenarioSpec(
        id="gateway_then_auth_leak",
        title="Two-act incident",
        tagline="The gateway surge is real; once load is shed, a latent auth heap leak shows up.",
        root_cause="api-gateway",
        failure_mode="overloaded",
        compound_secondary=("auth-service", "memory_leak"),
    ),
)

_BY_ID = {s.id: s for s in SCENARIOS}


def resolve_scenario(scenario_id: Optional[str]) -> ScenarioSpec:
    if not scenario_id:
        return _BY_ID["surprise"]
    return _BY_ID.get(scenario_id, _BY_ID["surprise"])


def scenario_dropdown_choices() -> list[tuple[str, str]]:
    return [(s.title, s.id) for s in SCENARIOS]


def daily_rotation_choices() -> list[tuple[str, str]]:
    """Scenarios with fixed roots — better for daily leaderboards than pure RNG."""
    return [(s.title, s.id) for s in SCENARIOS if s.root_cause is not None]
