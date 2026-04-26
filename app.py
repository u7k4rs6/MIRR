import copy
import json
import os
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agent.heuristic_agent import HeuristicAgent
from agent.llm_agent import LLMAgent
from agent.random_agent import RandomAgent
from env import replay as replay_mod
from env.actions import runbook_dropdown_choices
from env.daily import daily_challenge_banner, utc_date_string
from env.environment import IncidentResponseEnv
from env.scenarios import daily_rotation_choices, scenario_dropdown_choices
from env.simulator import FAILURE_MODES, SERVICES

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

semaphore = threading.Semaphore(1)


def _health_figure(health: list[float]):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    if not health:
        ax.text(0.5, 0.5, "No data", ha="center")
        fig.tight_layout()
        return fig
    ax.plot(range(len(health)), health, color="#2563eb", marker="o", markersize=4, linewidth=1.5)
    ax.fill_between(range(len(health)), health, alpha=0.12, color="#2563eb")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("System health (mean)")
    ax.set_title("Mean service health (true, instructor view when rich UI is on)")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def _service_cpu_figure(metrics_hist: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    if not metrics_hist:
        ax.text(0.5, 0.5, "Run steps to plot noisy CPU", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    xs = list(range(len(metrics_hist)))
    for svc in SERVICES:
        ys = [float((row.get(svc) or {}).get("cpu") or 0) for row in metrics_hist]
        ax.plot(xs, ys, marker="o", markersize=2, linewidth=1.0, label=svc)
    ax.set_xlabel("Step")
    ax.set_ylabel("Observed CPU (noisy)")
    ax.set_title("Per-service observed CPU")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def _scenario_banner(obs: dict) -> str:
    title = obs.get("scenario_title") or "Incident"
    tag = obs.get("scenario_tagline") or ""
    sid = obs.get("scenario_id") or "surprise"
    return f"### {title}\n*{tag}*\n`scenario_id={sid}`"


def _obs_brief(obs: dict) -> str:
    lines = [
        _scenario_banner(obs),
        "",
        f"**Step** {obs.get('step', 0)} / {obs.get('max_steps', '?')} · **Health** {obs.get('system_health_score', 0):.3f}",
        f"**Alerts:** {', '.join(obs.get('recent_alerts') or ['none'])}",
    ]
    if obs.get("incident_cost") is not None:
        lines.append(f"**Incident cost** {float(obs['incident_cost']):.2f} _logs + wrong fixes + time_")
    cl = obs.get("compound_legs")
    if cl:
        lines.append(f"**Cascades resolved** {cl}")
    feed = obs.get("alert_feed") or []
    if feed:
        lines.extend(["", "**Alert stream**"])
        for line in feed[-10:]:
            lines.append(f"- {line}")
    lines.extend(["", "**Per-service CPU (noisy)**"])
    m = obs.get("metrics") or {}
    for s in SERVICES:
        row = m.get(s) or {}
        lines.append(f"- `{s}`: cpu={row.get('cpu', '?')} latency_ms={row.get('latency_ms', '?')}")
    th = obs.get("true_health_by_service")
    if th:
        lines.extend(["", "**True health (rich UI — teaching overlay)**"])
        for s in SERVICES:
            lines.append(f"- `{s}`: {th.get(s, '?')}")
    lines.extend(["", "**Last result**", obs.get("last_action_result", "—")])
    return "\n".join(lines)


def _replay_extras(info: dict) -> dict:
    return {
        "incident_cost": info.get("incident_cost"),
        "explanation_score": info.get("explanation_score"),
        "compound_legs": info.get("compound_legs"),
    }


def run_episode_demo(agent_choice: str, scenario_id: str, seed: int):
    if not semaphore.acquire(blocking=False):
        yield (
            "⏳ Another episode is running — try again in a few seconds.",
            "",
            "",
            None,
            None,
            "",
        )
        return

    try:
        env = IncidentResponseEnv(max_steps=30)

        if agent_choice == "Random":
            agent = RandomAgent()
        elif agent_choice == "Heuristic":
            agent = HeuristicAgent()
        else:
            if not os.environ.get("GROQ_API_KEY"):
                yield (
                    "**GROQ_API_KEY** is not set. Add it in Space secrets or your shell for **LLM (Groq)**.",
                    "",
                    "",
                    None,
                    None,
                    "",
                )
                return
            agent = LLMAgent()

        if hasattr(agent, "reset"):
            agent.reset()

        obs, _ = env.reset(seed=int(seed), options={"scenario_id": scenario_id})
        log: list[str] = []
        trace: list[dict] = []
        metrics_hist: list[dict] = [copy.deepcopy(obs.get("metrics") or {})]
        total_reward = 0.0
        health_series = [float(obs["system_health_score"])]

        log.append(_scenario_banner(obs))
        log.append(f"**Seed** `{int(seed)}` · **Agent** {agent_choice}")
        log.append("---")

        done = False
        step_idx = 0
        info: dict = {}
        while not done:
            try:
                action = agent.act(obs)
            except Exception as e:
                log.append(f"**Agent error:** `{e}`")
                done = True
                info = {
                    "outcome": "agent_error",
                    "root_cause": env.sim.state.root_cause,
                    "failure_mode": env.sim.state.failure_mode,
                    "diagnosis_correct": False,
                    "incident_cost": env._compute_incident_cost(),
                    "compound_legs": env._compound_legs_seen,
                }
                break
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step_idx += 1
            replay_mod.append_step(trace, action, reward, obs)
            health_series.append(float(obs["system_health_score"]))
            metrics_hist.append(copy.deepcopy(obs.get("metrics") or {}))

            log.append(f"**Step {step_idx}** `{json.dumps(action)}`")
            log.append(f"↳ {obs.get('last_action_result', '')}")
            log.append(
                f"↳ Health **{obs['system_health_score']:.3f}** · reward **{reward:+.1f}** · cost **{info.get('incident_cost', 0):.2f}**"
            )
            log.append("")
            yield (
                "\n\n".join(log),
                f"{total_reward:.1f}",
                "",
                _health_figure(health_series),
                _service_cpu_figure(metrics_hist),
                "",
            )

        outcome = info.get("outcome", "unknown")
        rc = info.get("root_cause", "?")
        fm = info.get("failure_mode", "?")
        diag = "correct diagnosis" if info.get("diagnosis_correct") else "missed diagnosis"
        ex = info.get("explanation_score")
        ex_s = f" · explanation **{ex:.2f}**" if isinstance(ex, (int, float)) else ""

        log.append("---")
        log.append(f"## Outcome: **{outcome.upper()}**")
        log.append(
            f"Ground truth: `{rc}` / `{fm}` · {diag} · reward **{total_reward:.1f}** · cost **{info.get('incident_cost', 0):.2f}**{ex_s}"
        )

        extras = _replay_extras(info)
        doc = replay_mod.build_episode_document(
            scenario_id=scenario_id,
            seed=int(seed),
            trace=trace,
            outcome=outcome,
            total_reward=total_reward,
            reveal=True,
            root_cause=rc,
            failure_mode=fm,
            **{k: v for k, v in extras.items() if v is not None},
        )
        replay_json = replay_mod.dumps_pretty(doc)

        yield (
            "\n\n".join(log),
            f"{total_reward:.1f}",
            f"{outcome.upper()} — {rc} / {fm}",
            _health_figure(health_series),
            _service_cpu_figure(metrics_hist),
            replay_json,
        )
    finally:
        semaphore.release()


def _build_human_action(
    action_type: str, target: str, failure_mode: str, evidence: str, defer_reason: str
) -> dict:
    if action_type == "no_op":
        return {"type": "no_op"}
    if action_type == "defer":
        out: dict = {"type": "defer"}
        if defer_reason.strip():
            out["reason"] = defer_reason.strip()
        return out
    if action_type == "diagnose":
        out = {"type": "diagnose", "target": target, "failure_mode": failure_mode}
        if evidence.strip():
            out["evidence"] = evidence.strip()
        return out
    if action_type in ("restart_service", "rollback_deploy", "scale_up", "check_logs", "enable_circuit_breaker"):
        return {"type": action_type, "target": target}
    return {"type": "no_op"}


def human_start(scenario_id: str, seed: int):
    env = IncidentResponseEnv(max_steps=30)
    obs, _ = env.reset(seed=int(seed), options={"scenario_id": scenario_id, "rich_ui": True})
    state = {
        "env": env,
        "obs": obs,
        "trace": [],
        "health": [float(obs["system_health_score"])],
        "metrics_hist": [copy.deepcopy(obs.get("metrics") or {})],
        "total_reward": 0.0,
        "done": False,
        "scenario_id": scenario_id,
        "seed": int(seed),
        "info": {},
    }
    return (
        _obs_brief(obs),
        _health_figure(state["health"]),
        _service_cpu_figure(state["metrics_hist"]),
        state,
        "",
        "",
        gr.update(interactive=True),
    )


def human_step(
    state,
    action_type: str,
    target: str,
    failure_mode: str,
    evidence: str,
    defer_reason: str,
    baseline_record,
):
    if not state:
        return (
            "_Click **Start incident** first._",
            None,
            None,
            state,
            "",
            "",
            gr.update(interactive=True),
        )
    if state.get("done"):
        return (
            "_Episode finished — press **Start incident** for a new run._",
            None,
            None,
            state,
            "",
            "",
            gr.update(interactive=False),
        )

    env = state["env"]
    obs = state["obs"]
    action = _build_human_action(action_type, target, failure_mode, evidence, defer_reason)
    obs2, reward, done, _, info = env.step(action)
    replay_mod.append_step(state["trace"], action, reward, obs2)

    state["obs"] = obs2
    state["total_reward"] += float(reward)
    state["health"].append(float(obs2["system_health_score"]))
    state["metrics_hist"].append(copy.deepcopy(obs2.get("metrics") or {}))
    state["done"] = bool(done)
    state["info"] = info

    outcome = ""
    rj = ""
    if done:
        oc = info.get("outcome", "?")
        rc = info.get("root_cause", "?")
        fm = info.get("failure_mode", "?")
        outcome = (
            f"**{oc.upper()}** · ground truth `{rc}` / `{fm}` · reward **{state['total_reward']:.1f}** "
            f"· cost **{info.get('incident_cost', 0):.2f}**"
        )
        ex = info.get("explanation_score")
        if isinstance(ex, (int, float)):
            outcome += f" · explanation **{ex:.2f}**"
        extras = _replay_extras(info)
        doc = replay_mod.build_episode_document(
            scenario_id=state["scenario_id"],
            seed=state["seed"],
            trace=state["trace"],
            outcome=oc,
            total_reward=state["total_reward"],
            reveal=True,
            root_cause=rc,
            failure_mode=fm,
            **{k: v for k, v in extras.items() if v is not None},
        )
        rj = replay_mod.dumps_pretty(doc)

        if baseline_record and isinstance(baseline_record, dict):
            br = float(baseline_record.get("reward", 0))
            bs = int(baseline_record.get("steps", 0))
            bc = float(baseline_record.get("incident_cost") or 0)
            hs = len(state["trace"])
            outcome += (
                "\n\n### vs baseline (same seed & scenario)\n"
                f"| | You | Baseline |\n|--:|--:|--:|\n"
                f"| Reward | {state['total_reward']:.1f} | {br:.1f} |\n"
                f"| Steps | {hs} | {bs} |\n"
                f"| Incident cost | {float(info.get('incident_cost') or 0):.2f} | {bc:.2f} |\n"
            )

    return (
        _obs_brief(obs2),
        _health_figure(state["health"]),
        _service_cpu_figure(state["metrics_hist"]),
        state,
        outcome,
        rj,
        gr.update(interactive=not done),
    )


def toggle_action_extras(action_type: str):
    is_diag = action_type == "diagnose"
    is_def = action_type == "defer"
    return (
        gr.update(visible=is_diag),
        gr.update(visible=is_diag),
        gr.update(visible=is_def),
    )


def compare_human_start(scenario_id: str, seed: int):
    return human_start(scenario_id, seed)


def run_baseline_only(agent_choice: str, scenario_id: str, seed: int):
    if not semaphore.acquire(blocking=False):
        return (
            "⏳ Wait for the other run to finish.",
            "",
            None,
        )
    try:
        env = IncidentResponseEnv(max_steps=30)
        if agent_choice == "Random":
            agent = RandomAgent()
        elif agent_choice == "Heuristic":
            agent = HeuristicAgent()
        else:
            if not os.environ.get("GROQ_API_KEY"):
                return ("**GROQ_API_KEY** not set for LLM baseline.", "", None)
            agent = LLMAgent()
        if hasattr(agent, "reset"):
            agent.reset()
        obs, _ = env.reset(seed=int(seed), options={"scenario_id": scenario_id})
        trace: list[dict] = []
        total_reward = 0.0
        step_idx = 0
        done = False
        info: dict = {}
        while not done:
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step_idx += 1
            replay_mod.append_step(trace, action, reward, obs)
        oc = info.get("outcome", "?")
        rc = info.get("root_cause", "?")
        fm = info.get("failure_mode", "?")
        extras = _replay_extras(info)
        doc = replay_mod.build_episode_document(
            scenario_id=scenario_id,
            seed=int(seed),
            trace=trace,
            outcome=oc,
            total_reward=total_reward,
            reveal=True,
            root_cause=rc,
            failure_mode=fm,
            **{k: v for k, v in extras.items() if v is not None},
        )
        replay_json = replay_mod.dumps_pretty(doc)
        md = (
            f"**Baseline** ({agent_choice}) seed `{int(seed)}` scenario `{scenario_id}`\n\n"
            f"- Outcome: **{oc}**\n- Reward: **{total_reward:.1f}**\n- Steps: **{step_idx}**\n"
            f"- Cost: **{float(info.get('incident_cost') or 0):.2f}**\n"
        )
        ex = info.get("explanation_score")
        if isinstance(ex, (int, float)):
            md += f"- Explanation: **{ex:.2f}**\n"
        record = {
            "reward": total_reward,
            "steps": step_idx,
            "outcome": oc,
            "incident_cost": float(info.get("incident_cost") or 0),
            "explanation_score": ex,
            "agent": agent_choice,
        }
        return md, replay_json, record
    finally:
        semaphore.release()


def replay_verify_ui(json_text: str, rich_plots: bool):
    text = (json_text or "").strip()
    if not text:
        return "Paste a replay JSON document first.", None, None
    try:
        doc = replay_mod.parse_episode_document(text)
        rep = replay_mod.recompute_episode(doc, rich_ui=bool(rich_plots))
    except Exception as e:
        return f"**Parse / replay error:** `{e}`", None, None

    lines = [
        "## Replay verification",
        f"- **Steps in file:** {len(doc.get('steps') or [])} · **Executed:** {rep['steps_executed']}",
        f"- **Episode finished in replay:** {rep['episode_finished']}",
        f"- **Outcome** file `{rep['original_outcome']}` vs replay `{rep['replay_outcome']}` → **{'MATCH' if rep['outcome_match'] else 'MISMATCH'}**",
        f"- **Total reward** file `{rep['original_total_reward']}` vs replay `{rep['replay_total_reward']}` → **{'close' if rep['reward_close'] else 'DIFF'}**",
    ]
    if rep.get("ground_truth_doc"):
        lines.append(f"- **Ground truth in file:** `{rep['ground_truth_doc']}`")
    if rep.get("final_info"):
        lines.append(f"- **Final sim state:** `{rep['final_info']}`")
    if rep.get("incident_cost") is not None:
        lines.append(f"- **Replay incident cost (end):** {rep['incident_cost']:.3f}")
    tail = "\n".join(rep["log_lines"][-40:])
    lines.extend(["", "### Step log", "```text", tail, "```"])
    return "\n".join(lines), _health_figure(rep["health_series"]), _service_cpu_figure(rep["metrics_hist"])


def daily_markdown():
    pairs = daily_rotation_choices()
    if not pairs:
        pairs = scenario_dropdown_choices()
    _seed, _sid, blurb = daily_challenge_banner([(t, i) for t, i in pairs])
    return (
        f"{blurb}\n\n---\n\n"
        f"_UTC today: **{utc_date_string()}** — rotate scenarios from the dropdown in other tabs._\n\n"
        "**Slack:** `python scripts/post_daily_slack.py` with `SLACK_WEBHOOK_URL` set (optional `HF_SPACE_URL`)."
    )


TOOLKIT_MD = """
### Runbook aliases (normalized automatically)

| Canonical `type` | Aliases |
|------------------|---------|
| `check_logs` | `kubectl_logs`, `tail_logs`, `stern_tail`, `open_trace` |
| `restart_service` | `kubectl_rollout_restart`, `restart_pod` |
| `rollback_deploy` | `kubectl_rollout_undo`, `kubectl_undo_rollout` |
| `scale_up` | `kubectl_scale_deployment`, `scale_deployment` |
| `defer` | safe hold — small time cost |

### Curriculum (training / eval)

Pass `options["curriculum_stage"]` in `1..4` on `reset()` for more metric noise and slightly shorter horizons (`env.curriculum`).

### Telemetry adapter

`env.telemetry.SimulatorTelemetry(sim)` and `FileTelemetry(path)` — JSONL lines with a `metrics` object.

### Compound scenarios

Scenario `gateway_then_auth_leak` applies a **second** failure after the first correct fix (`compound_secondary` in `env/scenarios.py`).

### Replay import

Use the **Replay import** tab, or call `env.replay.recompute_episode(doc)` from Python / tests.
"""


LEADERBOARD_MD = """
| Tab | Purpose |
|-----|---------|
| **Watch an agent** | Streaming log + mean health + **per-service CPU** chart + replay JSON |
| **Human incident room** | Rich UI: alert stream, true-health overlay, runbook action labels |
| **Compare vs baseline** | Run an agent once, then you play the **same seed** — table at the end |
| **Replay import** | Paste shared JSON — **re-simulate** and check outcome/reward vs file |
| **Daily challenge** | UTC-stable seed + rotating scenario |
| **Toolkit** | Aliases, curriculum, telemetry notes |
"""

RUNBOOK_CHOICES = runbook_dropdown_choices()

with gr.Blocks(title="Incident Response Lab") as demo:
    gr.Markdown(
        "# Incident response lab\n"
        "Microservice incident **sandbox**: scenarios, compound failures, runbook-shaped actions, "
        "cost accounting, replay JSON, and optional **daily** + **Slack** hooks."
    )

    baseline_snapshot = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Watch an agent"):
            gr.Markdown(LEADERBOARD_MD)
            with gr.Row():
                with gr.Column(scale=2):
                    scenario_dd = gr.Dropdown(
                        label="Scenario",
                        choices=scenario_dropdown_choices(),
                        value="surprise",
                    )
                    agent_select = gr.Radio(
                        ["Random", "Heuristic", "LLM (Groq)"],
                        label="Agent",
                        value="Heuristic",
                    )
                    seed_input = gr.Slider(0, 9999, value=42, step=1, label="Episode seed")
                    run_btn = gr.Button("Run episode", variant="primary")
                with gr.Column(scale=1):
                    reward_out = gr.Textbox(label="Total reward", interactive=False)
                    outcome_out = gr.Textbox(label="Outcome", interactive=False)
            health_plot = gr.Plot(label="Mean system health")
            cpu_plot = gr.Plot(label="Observed CPU by service")
            episode_log = gr.Markdown()
            replay_out = gr.Code(label="Replay JSON (shareable)", language="json")

            run_btn.click(
                fn=run_episode_demo,
                inputs=[agent_select, scenario_dd, seed_input],
                outputs=[episode_log, reward_out, outcome_out, health_plot, cpu_plot, replay_out],
            )

        with gr.Tab("Human incident room"):
            gr.Markdown(
                "Runbook labels in the action dropdown. **Diagnose** allows optional **evidence** (scored at episode end). "
                "**Rich UI** exposes an alert stream and true health (teaching overlay)."
            )
            human_state = gr.State(None)
            human_no_baseline = gr.State(None)
            with gr.Row():
                h_scenario = gr.Dropdown(
                    label="Scenario",
                    choices=scenario_dropdown_choices(),
                    value="flash_checkout",
                )
                h_seed = gr.Slider(0, 9999, value=7, step=1, label="Seed")
            h_start = gr.Button("Start incident", variant="primary")
            h_plot = gr.Plot(label="Mean system health")
            h_cpu = gr.Plot(label="Observed CPU by service")
            h_brief = gr.Markdown()
            with gr.Row():
                h_action = gr.Dropdown(
                    label="Action (runbook)",
                    choices=RUNBOOK_CHOICES,
                    value="check_logs",
                )
                h_target = gr.Dropdown(label="Target service", choices=SERVICES, value=SERVICES[0])
                h_fm = gr.Dropdown(
                    label="Failure mode (diagnose only)",
                    choices=FAILURE_MODES,
                    value=FAILURE_MODES[0],
                    visible=False,
                )
            h_evidence = gr.Textbox(
                label="Evidence / rationale (diagnose only)",
                placeholder="Optional — improves explanation score",
                visible=False,
                lines=2,
            )
            h_defer_reason = gr.Textbox(
                label="Defer reason (optional)",
                visible=False,
                lines=1,
            )
            h_go = gr.Button("Execute action", variant="secondary")
            h_outcome = gr.Markdown()
            h_replay = gr.Code(label="Replay JSON", language="json")

            h_start.click(
                fn=human_start,
                inputs=[h_scenario, h_seed],
                outputs=[h_brief, h_plot, h_cpu, human_state, h_outcome, h_replay, h_go],
            )
            h_action.change(
                fn=toggle_action_extras,
                inputs=[h_action],
                outputs=[h_fm, h_evidence, h_defer_reason],
            )
            h_go.click(
                fn=human_step,
                inputs=[
                    human_state,
                    h_action,
                    h_target,
                    h_fm,
                    h_evidence,
                    h_defer_reason,
                    human_no_baseline,
                ],
                outputs=[h_brief, h_plot, h_cpu, human_state, h_outcome, h_replay, h_go],
            )

        with gr.Tab("Compare vs baseline"):
            gr.Markdown(
                "1) Pick scenario + seed and run a **baseline** agent.\n"
                "2) **Start your incident** with the same settings and play manually.\n"
                "3) When you finish, we append a **comparison table** if a baseline exists."
            )
            cmp_human_state = gr.State(None)
            with gr.Row():
                c_scenario = gr.Dropdown(
                    label="Scenario",
                    choices=scenario_dropdown_choices(),
                    value="gateway_then_auth_leak",
                )
                c_seed = gr.Slider(0, 9999, value=11, step=1, label="Seed")
            with gr.Row():
                c_agent = gr.Radio(["Random", "Heuristic", "LLM (Groq)"], label="Baseline agent", value="Heuristic")
                c_run_bl = gr.Button("Run baseline agent", variant="primary")
            c_bl_md = gr.Markdown()
            c_bl_replay = gr.Code(label="Baseline replay JSON", language="json")
            c_run_bl.click(
                fn=run_baseline_only,
                inputs=[c_agent, c_scenario, c_seed],
                outputs=[c_bl_md, c_bl_replay, baseline_snapshot],
            )

            gr.Markdown("### Your run (same scenario + seed)")
            ch_start = gr.Button("Start your incident", variant="secondary")
            ch_plot = gr.Plot(label="Mean health")
            ch_cpu = gr.Plot(label="CPU by service")
            ch_brief = gr.Markdown()
            with gr.Row():
                ch_action = gr.Dropdown(label="Action", choices=RUNBOOK_CHOICES, value="check_logs")
                ch_target = gr.Dropdown(label="Target", choices=SERVICES, value=SERVICES[0])
                ch_fm = gr.Dropdown(
                    label="Failure mode",
                    choices=FAILURE_MODES,
                    value=FAILURE_MODES[0],
                    visible=False,
                )
            ch_evidence = gr.Textbox(label="Evidence (diagnose)", visible=False, lines=2)
            ch_defer = gr.Textbox(label="Defer reason", visible=False, lines=1)
            ch_go = gr.Button("Execute action", variant="secondary")
            ch_out = gr.Markdown()
            ch_rep = gr.Code(label="Your replay JSON", language="json")

            ch_start.click(
                fn=compare_human_start,
                inputs=[c_scenario, c_seed],
                outputs=[ch_brief, ch_plot, ch_cpu, cmp_human_state, ch_out, ch_rep, ch_go],
            )
            ch_action.change(
                fn=toggle_action_extras,
                inputs=[ch_action],
                outputs=[ch_fm, ch_evidence, ch_defer],
            )
            ch_go.click(
                fn=human_step,
                inputs=[
                    cmp_human_state,
                    ch_action,
                    ch_target,
                    ch_fm,
                    ch_evidence,
                    ch_defer,
                    baseline_snapshot,
                ],
                outputs=[ch_brief, ch_plot, ch_cpu, cmp_human_state, ch_out, ch_rep, ch_go],
            )

        with gr.Tab("Replay import"):
            gr.Markdown(
                "Paste a **replay JSON** export (from any tab). We re-run the same `scenario_id`, `seed`, "
                "and action sequence against the **current** simulator build — useful for challenges and regressions."
            )
            replay_paste = gr.Textbox(
                label="Replay JSON",
                placeholder='{"version": 1, "scenario_id": "...", "seed": 0, "steps": [...]}',
                lines=18,
            )
            replay_rich = gr.Checkbox(label="Rich plots (true health overlay in step log context)", value=False)
            replay_go = gr.Button("Verify / re-simulate", variant="primary")
            replay_report = gr.Markdown()
            replay_h = gr.Plot(label="Mean health (replay)")
            replay_cpu = gr.Plot(label="CPU by service (replay)")
            replay_go.click(
                fn=replay_verify_ui,
                inputs=[replay_paste, replay_rich],
                outputs=[replay_report, replay_h, replay_cpu],
            )

        with gr.Tab("Daily challenge"):
            dm = gr.Markdown(value=daily_markdown())
            gr.Button("Refresh blurb").click(fn=daily_markdown, outputs=[dm])

        with gr.Tab("Toolkit"):
            gr.Markdown(TOOLKIT_MD)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft())
