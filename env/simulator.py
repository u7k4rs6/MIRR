import numpy as np
from dataclasses import dataclass, field
from typing import Optional

SERVICES = ["api-gateway", "auth-service", "database", "cache", "worker"]
FAILURE_MODES = ["crashed", "memory_leak", "overloaded", "bad_deploy"]

# matrix[i][j] = fraction of service i's degradation that bleeds to service j per step
PROPAGATION = np.array([
    [1.0, 0.3, 0.0, 0.0, 0.0],  # api-gateway
    [0.4, 1.0, 0.0, 0.0, 0.0],  # auth-service
    [0.5, 0.2, 1.0, 0.3, 0.6],  # database
    [0.2, 0.0, 0.1, 1.0, 0.3],  # cache
    [0.3, 0.0, 0.0, 0.1, 1.0],  # worker
])

FIX_MAP = {
    "crashed": "restart_service",
    "memory_leak": "restart_service",
    "overloaded": "scale_up",
    "bad_deploy": "rollback_deploy",
}


@dataclass
class SimState:
    root_cause: str
    failure_mode: str
    true_health: dict  # service -> float [0,1]
    lagged_health: dict  # 1-step behind, what agent can "observe"
    circuit_breakers: set = field(default_factory=set)
    memory_leak_timer: int = 0  # steps since last restart (memory_leak recurrence)
    step: int = 0
    compound_secondary: Optional[tuple[str, str]] = None  # (service, failure_mode) after primary fix
    metric_noise_std: float = 0.15


class Simulator:
    def __init__(self):
        self.state: Optional[SimState] = None
        self.rng: Optional[np.random.Generator] = None

    def reset(
        self,
        seed=None,
        root_cause: Optional[str] = None,
        failure_mode: Optional[str] = None,
        compound_secondary: Optional[tuple[str, str]] = None,
        metric_noise_std: float = 0.15,
    ) -> SimState:
        self.rng = np.random.default_rng(seed)
        if root_cause is not None or failure_mode is not None:
            if root_cause is None or failure_mode is None:
                raise ValueError("root_cause and failure_mode must both be set or both omitted")
            if root_cause not in SERVICES:
                raise ValueError(f"root_cause must be one of {SERVICES}")
            if failure_mode not in FAILURE_MODES:
                raise ValueError(f"failure_mode must be one of {FAILURE_MODES}")
        else:
            root_cause = str(self.rng.choice(SERVICES))
            failure_mode = str(self.rng.choice(FAILURE_MODES))

        # Start healthy
        true_health = {s: 1.0 for s in SERVICES}
        # Root cause immediately degraded
        true_health[root_cause] = 0.2

        csec = compound_secondary
        if csec is not None:
            if len(csec) != 2 or csec[0] not in SERVICES or csec[1] not in FAILURE_MODES:
                raise ValueError("compound_secondary must be (service, failure_mode) in valid sets")
            if csec[0] == root_cause:
                raise ValueError("compound_secondary must differ from primary root_cause")

        self.state = SimState(
            root_cause=root_cause,
            failure_mode=failure_mode,
            true_health=true_health.copy(),
            lagged_health=true_health.copy(),
            compound_secondary=csec,
            metric_noise_std=float(metric_noise_std),
        )
        return self.state

    def tick(self):
        """Propagate failure one step. Call after updating root cause health."""
        s = self.state
        svc_idx = {svc: i for i, svc in enumerate(SERVICES)}

        # memory_leak recurrence: re-degrade after 4 steps post-restart
        if s.failure_mode == "memory_leak" and s.memory_leak_timer > 0:
            s.memory_leak_timer += 1
            if s.memory_leak_timer >= 4:
                s.true_health[s.root_cause] = max(0.0, s.true_health[s.root_cause] - 0.3)
                s.memory_leak_timer = 0

        # propagate degradation (weakens when root is mostly healthy so correct fixes can clear success)
        rc_idx = svc_idx[s.root_cause]
        rc_degradation = 1.0 - s.true_health[s.root_cause]
        if s.true_health[s.root_cause] >= 0.83:
            rc_degradation *= 0.12

        for j, svc in enumerate(SERVICES):
            if svc == s.root_cause:
                continue
            if svc in s.circuit_breakers:
                continue
            bleed = PROPAGATION[rc_idx][j] * rc_degradation * 0.25
            s.true_health[svc] = max(0.0, s.true_health[svc] - bleed)

        # circuit breaker caps health at 0.75 but stops further bleed
        for svc in s.circuit_breakers:
            s.true_health[svc] = min(0.75, s.true_health[svc] + 0.05)

        # update lagged health
        s.lagged_health = s.true_health.copy()
        s.step += 1

    def apply_fix(self, action_type: str, target: str) -> tuple[bool, str, bool]:
        """Returns (was_effective, result_message, compound_activated)."""
        s = self.state
        correct_fix = FIX_MAP.get(s.failure_mode, "")
        is_correct_target = target == s.root_cause
        is_correct_fix = action_type == correct_fix

        if is_correct_target and is_correct_fix:
            if s.failure_mode == "bad_deploy":
                # rollback: full restore
                s.true_health[s.root_cause] = 0.95
            elif s.failure_mode == "overloaded":
                s.true_health[s.root_cause] = 0.90
            elif s.failure_mode in ("crashed", "memory_leak"):
                s.true_health[s.root_cause] = 0.95
                if s.failure_mode == "memory_leak":
                    s.memory_leak_timer = 1  # start recurrence clock
            # Ease cascade damage on dependents once the root cause is remediated.
            for svc in SERVICES:
                if svc == s.root_cause:
                    continue
                s.true_health[svc] = min(1.0, s.true_health[svc] + 0.075)
            compound_activated = False
            pending = s.compound_secondary
            if pending is not None:
                sec_s, sec_m = pending
                s.compound_secondary = None
                if sec_s != s.root_cause:
                    s.root_cause = sec_s
                    s.failure_mode = sec_m
                    s.memory_leak_timer = 0
                    s.true_health[sec_s] = min(s.true_health.get(sec_s, 1.0), 0.34)
                    compound_activated = True
            msg = f"{action_type} on {target}: service recovering"
            if compound_activated:
                msg += f" // CASCADE: new primary {s.root_cause} ({s.failure_mode})"
            return True, msg, compound_activated
        elif is_correct_target and action_type == "restart_service" and s.failure_mode == "bad_deploy":
            # restart on bad_deploy WORSENS health
            s.true_health[target] = max(0.0, s.true_health[target] - 0.1)
            return False, f"{action_type} on {target}: made things worse", False
        elif is_correct_target and not is_correct_fix:
            return False, f"{action_type} on {target}: no significant change", False
        else:
            return False, f"{action_type} on {target}: wrong service, no change", False

    def enable_circuit_breaker(self, target: str) -> str:
        self.state.circuit_breakers.add(target)
        return f"circuit breaker enabled on {target}: propagation stopped, health capped at 0.75"

    def get_noisy_metrics(self) -> dict:
        s = self.state
        metrics = {}
        for svc in SERVICES:
            h = s.lagged_health[svc]
            noise = self.rng.normal(0, s.metric_noise_std)
            effective_h = max(0.0, min(1.0, h + noise))
            metrics[svc] = {
                "cpu": round(min(1.0, (1.0 - effective_h) * 0.9 + 0.1), 2),
                "error_rate": round(max(0.0, (1.0 - effective_h) * 0.6), 2),
                "latency_ms": round(50 + (1.0 - effective_h) * 1950),
                "queue_depth": round((1.0 - effective_h) * 400),
            }
        return metrics

    def get_trends(self, prev_health: dict) -> dict:
        trends = {}
        for svc in SERVICES:
            delta = self.state.true_health[svc] - prev_health.get(svc, 1.0)
            if delta < -0.05:
                trends[svc] = "degrading"
            elif delta > 0.05:
                trends[svc] = "recovering"
            else:
                trends[svc] = "stable"
        return trends

    def get_log_hints(self, target: str, step: int) -> list[str]:
        """Seeded, deterministic log hints. No LLM call needed."""
        s = self.state
        seed_val = hash((target, s.failure_mode, step)) % 1000
        rng = np.random.default_rng(seed_val)

        TEMPLATES = {
            "crashed": [
                f"{target}: process exited unexpectedly (signal 11)",
                f"{target}: out-of-memory kill detected",
                f"{target}: segfault in worker thread",
            ],
            "memory_leak": [
                f"{target}: heap usage growing 12MB/min",
                f"{target}: GC pressure increasing",
                f"{target}: memory usage at 87% of limit",
            ],
            "overloaded": [
                f"{target}: request queue depth 340, dropping connections",
                f"{target}: CPU throttling active",
                f"{target}: p99 latency spike 2.1s",
            ],
            "bad_deploy": [
                f"{target}: config parse error in v2.4.1",
                f"{target}: rollback available: v2.3.9",
                f"{target}: startup probe failing after deploy",
            ],
        }
        NOISE = [
            f"{target}: normal periodic checkpoint",
            f"{target}: health check passed",
            f"{target}: routine connection recycled",
        ]

        if target == s.root_cause:
            hints = TEMPLATES[s.failure_mode]
            chosen = list(rng.choice(hints, size=min(2, len(hints)), replace=False))
            noise = list(rng.choice(NOISE, size=1))
            return chosen + noise
        else:
            return list(rng.choice(NOISE, size=2, replace=True))

    @property
    def system_health(self) -> float:
        return float(np.mean(list(self.state.true_health.values())))
