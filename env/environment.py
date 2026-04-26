import json
from typing import Optional

from env.actions import normalize_action
from env.curriculum import max_steps_for_stage, noise_multiplier
from env.explanation import score_diagnosis_evidence
from env.scenarios import resolve_scenario
from env.simulator import Simulator, SERVICES, FAILURE_MODES


class IncidentResponseEnv:
    """Episodic microservice incident simulator (reset / step / render)."""

    metadata = {"render_modes": ["human", "json"]}
    env_id = "incident-response-v1"

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.sim = Simulator()
        self._effective_max_steps = max_steps
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._step = 0
        self._done = False
        self._diagnosis_made = False
        self._diagnosis_correct = False
        self._diagnosis_evidence = ""
        self._prev_health = {s: 1.0 for s in SERVICES}
        self._consecutive_healthy = 0
        self._consecutive_collapsed = 0
        self._last_action_result = "Episode started. Observe the system."
        self._health_history = []
        self._scenario_meta: dict = {}
        self._scenario_red_herrings: tuple[str, ...] = ()
        self._rich_ui = False
        self._log_pulls = 0
        self._wrong_fix_targets = 0
        self._healthy_fix_penalties = 0
        self._defer_count = 0
        self._alert_feed: list[str] = []
        self._true_health_trace: list[dict[str, float]] = []
        self._compound_legs_seen = 0
        self._last_alerts: list[str] = []
        self._diagnosis_target: Optional[str] = None
        self._diagnosis_mode: Optional[str] = None

    def reset(self, seed=None, options=None):
        self._reset_episode_state()
        options = options or {}
        self._rich_ui = bool(options.get("rich_ui", False))
        self._effective_max_steps = max_steps_for_stage(self.max_steps, options.get("curriculum_stage"))
        spec = resolve_scenario(options.get("scenario_id"))
        self._scenario_meta = {
            "scenario_id": spec.id,
            "scenario_title": spec.title,
            "scenario_tagline": spec.tagline,
        }
        self._scenario_red_herrings = tuple(spec.red_herring_alerts or ())
        noise = (
            0.15
            * float(spec.curriculum_noise_mult)
            * noise_multiplier(options.get("curriculum_stage"))
        )
        self.sim.reset(
            seed=seed,
            root_cause=spec.root_cause,
            failure_mode=spec.failure_mode,
            compound_secondary=spec.compound_secondary,
            metric_noise_std=noise,
        )
        self._last_alerts = self._get_alerts()
        self._append_alert_feed(self._last_alerts)
        obs = self._get_observation()
        return obs, {}

    def step(self, action: dict):
        assert not self._done, "Episode done — call reset()"

        reward = 0.0
        info: dict = {}
        action = normalize_action(action)
        action_type = action.get("type", "no_op")
        target = action.get("target", None)
        failure_mode = action.get("failure_mode", None)

        prev_health = self.sim.system_health

        compound_activated = False

        # --- Reward logic ---
        if action_type == "diagnose" and not self._diagnosis_made:
            self._diagnosis_made = True
            self._diagnosis_target = target
            self._diagnosis_mode = failure_mode
            self._diagnosis_evidence = (
                action.get("evidence")
                or action.get("rationale")
                or action.get("reasoning")
                or ""
            )
            if not isinstance(self._diagnosis_evidence, str):
                self._diagnosis_evidence = str(self._diagnosis_evidence)
            s = self.sim.state
            if target == s.root_cause and failure_mode == s.failure_mode:
                reward += 8.0
                self._diagnosis_correct = True
                self._last_action_result = f"Diagnosis: {target} / {failure_mode} — CORRECT"
            else:
                reward -= 2.0
                self._last_action_result = f"Diagnosis: {target} / {failure_mode} — INCORRECT"

        elif action_type in ("restart_service", "rollback_deploy", "scale_up"):
            if target and self.sim.state.lagged_health.get(target, 0) > 0.85:
                reward -= 1.5  # fixing healthy service
                self._healthy_fix_penalties += 1
            effective, msg, compound_activated = self.sim.apply_fix(action_type, target)
            self._last_action_result = msg
            if compound_activated:
                self._compound_legs_seen += 1
                self._diagnosis_made = False
                self._diagnosis_correct = False
                self._diagnosis_evidence = ""
                self._diagnosis_target = None
                self._diagnosis_mode = None
                info["compound_transition"] = True
            if effective:
                reward += 10.0 if self._diagnosis_correct else 6.0
            else:
                if target != self.sim.state.root_cause:
                    reward -= 2.0
                    self._wrong_fix_targets += 1

        elif action_type == "enable_circuit_breaker":
            msg = self.sim.enable_circuit_breaker(target)
            self._last_action_result = msg

        elif action_type == "check_logs":
            self._log_pulls += 1
            hints = self.sim.get_log_hints(target, self._step)
            self._last_action_result = "LOGS:\n" + "\n".join(hints)
            if self._log_pulls > 8:
                reward -= 0.25 * (self._log_pulls - 8)

        elif action_type == "defer":
            self._defer_count += 1
            reason = action.get("reason") or action.get("message")
            if reason and isinstance(reason, str):
                self._last_action_result = f"Deferred: {reason}"
            else:
                self._last_action_result = "Deferred: gathering more signal before a risky change."
            reward -= 0.06

        elif action_type == "no_op":
            reward -= 0.5
            self._last_action_result = "No action taken."

        # Tick simulator
        health_before_tick = {s: self.sim.state.true_health[s] for s in SERVICES}
        self.sim.tick()
        self._step += 1
        self._prev_health = health_before_tick

        # Health delta reward (capped)
        new_health = self.sim.system_health
        delta = new_health - prev_health
        if delta > 0:
            reward += min(delta * 2.0, 3.0)

        # Termination checks
        system_h = self.sim.system_health
        self._health_history.append(system_h)

        th = {s: float(self.sim.state.true_health[s]) for s in SERVICES}
        self._true_health_trace.append(th)

        self._last_alerts = self._get_alerts()
        self._append_alert_feed(self._last_alerts)

        if system_h >= 0.888:
            self._consecutive_healthy += 1
        else:
            self._consecutive_healthy = 0

        if system_h <= 0.10:
            self._consecutive_collapsed += 1
        else:
            self._consecutive_collapsed = 0

        success = self._consecutive_healthy >= 2
        collapsed = self._consecutive_collapsed >= 3
        timeout = self._step >= self._effective_max_steps

        if success:
            efficiency_bonus = (self._effective_max_steps - self._step) * 0.3
            reward += 20.0 + efficiency_bonus
            self._done = True
            info["outcome"] = "success"
        elif collapsed:
            reward -= 15.0
            self._done = True
            info["outcome"] = "collapsed"
        elif timeout:
            self._done = True
            info["outcome"] = "timeout"

        s_final = self.sim.state
        info["diagnosis_correct"] = self._diagnosis_correct
        info["root_cause"] = s_final.root_cause
        info["failure_mode"] = s_final.failure_mode
        if self._done:
            info["explanation_score"] = score_diagnosis_evidence(
                self._diagnosis_evidence,
                target=self._diagnosis_target or "",
                failure_mode=self._diagnosis_mode or "",
                root_cause=s_final.root_cause,
                true_mode=s_final.failure_mode,
            )
        info["incident_cost"] = self._compute_incident_cost()
        info["compound_legs"] = self._compound_legs_seen

        obs = self._get_observation()
        return obs, reward, self._done, False, info

    def _compute_incident_cost(self) -> float:
        return (
            self._log_pulls * 0.35
            + self._wrong_fix_targets * 1.2
            + self._healthy_fix_penalties * 1.0
            + self._defer_count * 0.15
            + max(0, self._step - 12) * 0.08
        )

    def _append_alert_feed(self, lines: list[str]) -> None:
        for line in lines:
            if line and line not in self._alert_feed[-3:]:
                self._alert_feed.append(line)
        self._alert_feed = self._alert_feed[-24:]

    def _get_alerts_core(self) -> list[str]:
        alerts = []
        for svc, h in self.sim.state.true_health.items():
            if h < 0.5:
                alerts.append(f"{svc}: health critical ({h:.2f})")
            elif h < 0.75:
                alerts.append(f"{svc}: degraded ({h:.2f})")
        return alerts[:4]

    def _get_alerts(self) -> list[str]:
        alerts = list(self._get_alerts_core())
        rng = self.sim.rng
        # Rotate red herrings so they do not fire every step
        if self._scenario_red_herrings and rng.random() < 0.45:
            pick = str(rng.choice(self._scenario_red_herrings))
            if pick not in alerts:
                alerts.append(pick)
        return alerts[:6]

    def _get_observation(self) -> dict:
        obs = {
            "step": self._step,
            "max_steps": self._effective_max_steps,
            "system_health_score": round(self.sim.system_health, 3),
            "metrics": self.sim.get_noisy_metrics(),
            "metric_trend": self.sim.get_trends(self._prev_health),
            "recent_alerts": list(self._last_alerts),
            "last_action_result": self._last_action_result,
            "diagnosis_made": self._diagnosis_made,
            "services": SERVICES,
            "failure_modes": FAILURE_MODES,
            "valid_actions": [
                "check_logs(service) | aliases: kubectl_logs, tail_logs",
                "diagnose(service, failure_mode, evidence?)",
                "restart_service(service) | aliases: kubectl_rollout_restart",
                "rollback_deploy(service) | aliases: kubectl_rollout_undo",
                "scale_up(service) | aliases: kubectl_scale_deployment",
                "enable_circuit_breaker(service)",
                "defer(reason?)",
                "no_op()",
            ],
            "compound_legs": self._compound_legs_seen,
            "incident_cost": round(self._compute_incident_cost(), 3),
        }
        if self._scenario_meta:
            obs.update(self._scenario_meta)
        if self._rich_ui:
            obs["alert_feed"] = list(self._alert_feed)
            obs["cost_breakdown"] = {
                "log_pulls": self._log_pulls,
                "wrong_fix_targets": self._wrong_fix_targets,
                "healthy_fix_penalties": self._healthy_fix_penalties,
                "defer_events": self._defer_count,
                "steps": self._step,
            }
            obs["true_health_by_service"] = {s: round(self.sim.state.true_health[s], 4) for s in SERVICES}
            if self._true_health_trace:
                obs["true_health_history"] = self._true_health_trace[-32:]
        return obs

    def render(self, mode="human") -> str:
        obs = self._get_observation()
        return json.dumps(obs, indent=2)

    def close(self):
        pass
