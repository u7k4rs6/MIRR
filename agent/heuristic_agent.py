from env.simulator import SERVICES


class HeuristicAgent:
    """Log-driven triage: full service scans, robust log capture, mode inference, recovery + fallbacks."""

    def __init__(self):
        self._diagnosed = False
        self._checked: set[str] = set()
        self._suspected = None
        self._failure_mode_guess = "crashed"
        self._log_by_service: dict[str, str] = {}
        self._last_check_target: str | None = None
        self._stabilize = False
        self._bad_restart_count = 0
        self._no_effect_restarts = 0
        self._issued_fix = False

    def reset(self):
        self._diagnosed = False
        self._checked = set()
        self._suspected = None
        self._failure_mode_guess = "crashed"
        self._log_by_service = {}
        self._last_check_target = None
        self._stabilize = False
        self._bad_restart_count = 0
        self._no_effect_restarts = 0
        self._issued_fix = False

    def _capture_previous_logs(self, obs: dict) -> None:
        raw = obs.get("last_action_result")
        lr = "" if raw is None else (raw if isinstance(raw, str) else str(raw))
        head = lr.lstrip()[:24].upper()
        looks_like_logs = "LOGS:" in lr or head.startswith("LOGS")
        if self._last_check_target and looks_like_logs:
            self._log_by_service[self._last_check_target] = lr
        self._last_check_target = None

    @staticmethod
    def _candidate_services(obs: dict) -> list[str]:
        metrics = obs["metrics"]
        by_cpu = sorted(SERVICES, key=lambda s: metrics[s]["cpu"], reverse=True)
        alert_svcs: list[str] = []
        for a in obs.get("recent_alerts", []) or []:
            if isinstance(a, str) and ":" in a:
                name = a.split(":", 1)[0].strip()
                if name in SERVICES and name not in alert_svcs:
                    alert_svcs.append(name)
        seen: set[str] = set()
        out: list[str] = []
        for s in alert_svcs + by_cpu:
            if s not in seen:
                seen.add(s)
                out.append(s)
        for s in SERVICES:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def _score_incident_keywords(text: str) -> int:
        if not text:
            return 0
        t = text.lower()
        score = 0
        for kw in (
            "rollback available",
            "config parse",
            "startup probe failing",
            "deploy",
            "v2.4.1",
            "v2.3.9",
            "queue depth",
            "throttling",
            "latency spike",
            "dropping connections",
            "heap usage",
            "gc pressure",
            "memory usage at",
            "exited unexpectedly",
            "oom kill",
            "segfault",
        ):
            if kw in t:
                score += 4
        return score

    @staticmethod
    def _text_suggests_mode(t: str, mode: str) -> bool:
        t = (t or "").lower()
        if mode == "bad_deploy":
            return any(
                x in t
                for x in (
                    "rollback available",
                    "config parse",
                    "startup probe failing",
                    "deploy",
                    "v2.4.1",
                    "v2.3.9",
                )
            )
        if mode == "overloaded":
            return any(x in t for x in ("queue depth", "throttling", "latency spike", "dropping connections"))
        if mode == "memory_leak":
            return any(x in t for x in ("heap usage", "gc pressure", "memory usage at"))
        return any(x in t for x in ("exited unexpectedly", "oom kill", "segfault"))

    @classmethod
    def _infer_best_failure_mode(cls, log_by_service: dict[str, str]) -> str:
        parts = [t for t in log_by_service.values() if t]
        parts = sorted(parts, key=lambda t: cls._score_incident_keywords(t), reverse=True)
        parts.append("\n".join(log_by_service.values()))
        for mode in ("bad_deploy", "overloaded", "memory_leak", "crashed"):
            for ch in parts:
                if cls._text_suggests_mode(ch, mode):
                    return mode
        return "crashed"

    def _pick_suspect(self, candidates: list[str]) -> str:
        best, best_score = None, -1
        for svc in self._checked:
            sc = self._score_incident_keywords(self._log_by_service.get(svc, ""))
            if sc > best_score:
                best_score = sc
                best = svc
        if best is not None and best_score > 0:
            return best
        for svc in candidates:
            if svc in self._checked:
                return svc
        return candidates[0] if candidates else SERVICES[0]

    def act(self, obs: dict) -> dict:
        self._capture_previous_logs(obs)
        lr_prev_raw = obs.get("last_action_result")
        lr_prev = (
            ""
            if lr_prev_raw is None
            else (lr_prev_raw if isinstance(lr_prev_raw, str) else str(lr_prev_raw))
        ).lower()

        if "cascade" in lr_prev or "new primary" in lr_prev:
            self._diagnosed = False
            self._stabilize = False
            self._issued_fix = False
            self._checked.clear()
            self._suspected = None
            self._log_by_service = {}
            self._failure_mode_guess = "crashed"
            self._bad_restart_count = 0
            self._no_effect_restarts = 0

        health = float(obs.get("system_health_score", 0) or 0)
        recovering_msg = (
            "service recovering" in lr_prev
            or ": service recovering" in lr_prev
        )

        # Do not lock on "recovering" while mean health is still bad (would no_op until timeout).
        if self._diagnosed:
            if self._stabilize and health < 0.872:
                self._stabilize = False
            if (recovering_msg and health >= 0.893) or (
                self._issued_fix and health >= 0.905
            ):
                self._stabilize = True
        if self._diagnosed and self._stabilize:
            return {"type": "no_op"}

        metrics = obs["metrics"]

        if not self._diagnosed:
            candidates = self._candidate_services(obs)
            for svc in candidates:
                if svc not in self._checked:
                    self._checked.add(svc)
                    self._last_check_target = svc
                    return {"type": "check_logs", "target": svc}

            self._suspected = self._pick_suspect(candidates)
            self._failure_mode_guess = self._infer_best_failure_mode(self._log_by_service)
            svc_log = self._log_by_service.get(self._suspected or "", "")
            for mode in ("bad_deploy", "overloaded", "memory_leak", "crashed"):
                if self._text_suggests_mode(svc_log, mode):
                    self._failure_mode_guess = mode
                    break
            self._diagnosed = True
            return {
                "type": "diagnose",
                "target": self._suspected,
                "failure_mode": self._failure_mode_guess,
            }

        if not self._suspected:
            return {"type": "no_op"}

        if self._failure_mode_guess == "crashed" and "no significant change" in lr_prev:
            self._no_effect_restarts += 1
            if self._no_effect_restarts >= 2:
                sl = (self._log_by_service.get(self._suspected, "") or "").lower()
                if any(
                    x in sl
                    for x in ("queue depth", "throttling", "latency spike", "dropping connections")
                ):
                    self._failure_mode_guess = "overloaded"
                self._no_effect_restarts = 0

        if self._failure_mode_guess == "crashed" and "made things worse" in lr_prev:
            self._bad_restart_count += 1
            if self._bad_restart_count >= 2:
                sus_log = (self._log_by_service.get(self._suspected, "") or "").lower()
                if any(
                    k in sus_log
                    for k in ("config parse", "rollback", "probe failing", "v2.4", "v2.3", "deploy")
                ):
                    self._failure_mode_guess = "bad_deploy"
                self._bad_restart_count = 0

        fm = self._failure_mode_guess
        self._issued_fix = True
        if fm == "bad_deploy":
            return {"type": "rollback_deploy", "target": self._suspected}
        if fm == "overloaded":
            return {"type": "scale_up", "target": self._suspected}
        return {"type": "restart_service", "target": self._suspected}
