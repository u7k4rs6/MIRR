---
title: MIRR — Incident Response Lab
emoji: 🚨
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
---

# MIRR - Microservice Incident Response & Recovery

> *At 3 AM, your payment service starts failing. Orders are queuing. Health checks are lying. You have five microservices, one silent killer, and no idea where to start.*
>
> *Your agent has 30 steps.*

---

## What is this?

A playground for agents that think under pressure.

MIRR is a partially observable microservice environment where classical rules-based agents and LLM agents diagnose and recover a broken distributed system - before it cascades into total failure. The environment is GRPO-ready and `train.ipynb` scaffolds a training loop, but no model has been trained against it yet and no checkpoint exists.

Five services. One hidden fault. Noisy metrics. A diagnosis action that forces the agent to commit its reasoning before it acts.

You can watch it happen live in the Gradio demo, step through episodes frame by frame, or wire your own model into the training scaffold.

---

## Links

| Deliverable | Link |
|---|---|
| HF Space (live demo) | Create a Space from this repo, then paste your URL here |
| Training Notebook (Colab) | Open in Colab - or upload `train.ipynb` from this clone |
| Source / updates | [github.com/u7k4rs6/MIRR](https://github.com/u7k4rs6/MIRR) |
| Trained Model | Run Step 3 in `train.ipynb` after training. Set `HF_TOKEN` + `HF_HUB_USERNAME`. Default: `YOUR_USERNAME/incident-response-grpo` |
| Episode Rollouts (Dataset) | Step 4 in `train.ipynb`. Default: `YOUR_USERNAME/incident-response-rollouts` |

**Hub uploads:** Set `HF_TOKEN` and `HF_HUB_USERNAME` in Colab (or `.env` locally), then run Steps 3 and 4 of `train.ipynb`. Copy `.env.example` to `.env` for local runs - it's gitignored.

---

## The Setup

Here's the actual problem the agent faces each episode:

```
Five microservices. One is failing silently.
Metrics are noisy (±15%). Logs cost a step to read.
You don't know which service is broken - and neither do your metrics.
```

The agent's sequence:
1. **Observe** - degraded health metrics arrive with noise baked in
2. **Investigate** - call `check_logs()` to narrow it down (costs a step)
3. **Diagnose** - explicitly commit to a root cause before touching anything
4. **Fix** - `restart`, `rollback`, or `scale_up` the right service
5. **Confirm** - watch recovery propagate, or watch it get worse

The diagnosis step is the whole game. It's what separates a reasoning agent from a lucky guesser.

---

## Why the Diagnose Action Changes Everything

Here's what brute-forcing looks like on the reward function:

```
Brute-force: tries all 5 services
  → -2.0 × 4 wrong fix attempts
  → +6.0 on the lucky final hit
  = -2.0 total
```

Here's what actually reasoning looks like:

```
Reasoning agent: commits to the right diagnosis first
  → +8.0 correct diagnosis
  → +10.0 correct fix
  → +20.0 full recovery
  = 38.0+
```

**That's a 40-point gap from one design decision.** Scoring diagnosis separately from the fix means you can't hide shallow reasoning behind a lucky action. The environment punishes confident wrongness and rewards structured thinking.

---

## Failure Modes

Not all failures are created equal. Three modes, three different twists:

| Mode | Correct Fix | The Catch |
|---|---|---|
| `crashed` | `restart` | Clean. Straightforward. |
| `memory_leak` | `restart` | Works - but it comes back after 4 steps. |
| `overloaded` | `scale_up` | Restart does nothing. Watch agents flail. |
| `bad_deploy` | `rollback` | Restart actively makes it worse. |

The `bad_deploy` mode is the one that breaks naive heuristics. If your agent's mental model is "crashed = restart," it'll restart a bad deploy and tank the health score further. This is intentional.

---

## Results

<!-- BEGIN GENERATED RESULTS (scripts/sync_readme_results.py) -->
| Agent | Success Rate | Diagnosis Acc. | Mean Reward |
|---|---|---|---|
| Random | 26% | 4% | -18.08 |
| Heuristic (log-aware) | 100% | 100% | 55.98 |

Measured over n=100 episodes per agent, seed=12345, `max_steps=30` - the same horizon the demo runs. Code fingerprint `sha256:621558321891274f`. Reproduce with `python eval/evaluate.py`; raw output is committed at [`eval/results.json`](eval/results.json).

LLM agents are deliberately absent from this table. They call a live third-party API, so their scores are not reproducible from this repo alone.
<!-- END GENERATED RESULTS -->

---

## Environment Design

```
openenv.yaml          - Env metadata (id, thresholds, service list)
env/environment.py    - Episodic API: reset / step / render
env/simulator.py      - Hidden state, failure propagation, health logic
agent/                - Random, heuristic, and LLM agents
eval/evaluate.py      - Evaluation loop (writes eval/results.json)
eval/results.json     - Committed baseline measurements (the only valid numbers)
scripts/              - README results sync + check, daily Slack poster
train.ipynb           - GRPO training scaffold (Colab-ready; no trained model yet)
app.py                - Gradio live demo
```

The environment is OpenEnv-compliant. `reset()` / `step()` / `render()` are implemented per spec. Drop in any compatible agent and it runs.

---

## Setup

**Local:**
```bash
pip install -r requirements.txt

# Windows
set GROQ_API_KEY=your_key_here
set PYTHONPATH=%CD%

# Linux / macOS
export GROQ_API_KEY=your_key_here
export PYTHONPATH="$(pwd)"

python eval/evaluate.py
python app.py
```

**HF Space:** add `GROQ_API_KEY` under Space secrets. The app listens on `PORT` (default `7860`).

---

## Release Checklist

- [ ] Public HF Space - smoke-test from incognito
- [ ] `openenv.yaml` at repo root
- [ ] `environment.py` implements `reset()` / `step()` / `render()`
- [ ] `train.ipynb` runnable end-to-end; Colab copy in sync
- [ ] README links point at live URLs

---

## Why I built this

Most RL environments are either too clean (CartPole, Atari) or too opaque (production infra you can't open up).

MIRR sits in the middle - messy enough that brute force fails, structured enough that you can actually measure reasoning. The `diagnose()` action exists because I wanted to see if forcing an explicit commitment step changed how agents behave. It does.

The GRPO scaffolding is in place - reward function, rollout format, and a Colab notebook. I have not trained a model against it yet, so there is no checkpoint and no training curve to show. Bring your own model and point it at the rollout format.

---

<p align="center">
  <sub>Built by <a href="https://github.com/u7k4rs6">Utkarsh Bahuguna</a> &nbsp;·&nbsp; PRs welcome &nbsp;·&nbsp; Star if it taught you something</sub>
</p>
