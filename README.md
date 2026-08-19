---
title: MIRR — Incident Response Lab
emoji: 🚨
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.24.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# MIRR - Microservice Incident Response & Recovery

> *At 3 AM, your payment service starts failing. Orders are queuing. Health checks are lying. You have five microservices, one silent killer, and no idea where to start.*
>
> *Your agent has 30 steps.*

*Originally built for the Meta x PyTorch Hackathon.*

---

## What is this?

A playground for agents that think under pressure.

MIRR is a partially observable microservice environment where classical rules-based agents and LLM agents diagnose and recover a broken distributed system - before it cascades into total failure. The environment is GRPO-ready and `train.ipynb` scaffolds a training loop, but no model has been trained against it yet and no checkpoint exists.

Five services. One hidden fault. Noisy metrics. A diagnosis action that forces the agent to commit its reasoning before it acts.

You can watch an agent run live in the Gradio demo, play the same incident yourself in the human incident room, replay a saved episode, or wire your own model into the training scaffold.

---

## Links

| Deliverable | Link |
|---|---|
| HF Space (live demo) | [huggingface.co/spaces/u7k4rs6/Metafinal](https://huggingface.co/spaces/u7k4rs6/Metafinal) |
| Training Notebook (Colab) | Open in Colab - or upload `train.ipynb` from this clone |
| Source / updates | [github.com/u7k4rs6/MIRR](https://github.com/u7k4rs6/MIRR) |
| Episode Rollouts (Dataset) | Step 4 in `train.ipynb`. Default: `YOUR_USERNAME/incident-response-rollouts` |

The Space is named **Metafinal** because it was originally built for the Meta x PyTorch Hackathon and is grandfathered onto free CPU hardware; it serves the current MIRR code.

**Hub uploads:** Step 4 of `train.ipynb` publishes heuristic-agent rollouts as a dataset. Step 3 pushes a *trained model* - it has nothing to push until you actually train one, and will skip itself until then.

Set `HF_TOKEN` and `HF_HUB_USERNAME` via Colab Secrets, or export them in the shell that launches Jupyter. **`.env` does not work for these** - only `app.py` and `eval/evaluate.py` call `load_dotenv`; `train.ipynb` never does, so values placed there are silently ignored. `.env` is read for `GROQ_API_KEY` (and `GROQ_MODEL`) when running the app locally; copy `.env.example` to `.env` for that - it's gitignored.

---

## The Setup

Here's the actual problem the agent faces each episode:

```
Five microservices. One is failing silently.
Metrics are noisy (±15% base, scaled up by scenarios and curriculum stages).
Logs cost a step to read - and past 8 pulls an escalating -0.25 per pull.
You don't know which service is broken - and neither do your metrics.
```

The agent's sequence:
1. **Observe** - degraded health metrics arrive with noise baked in
2. **Investigate** - call `check_logs()` to narrow it down (costs a step)
3. **Diagnose** - explicitly commit to a root cause before touching anything
4. **Fix** - `restart_service`, `rollback_deploy`, or `scale_up` the right service
5. **Confirm** - watch recovery propagate, or watch it get worse

The diagnosis step is the whole game. It's what separates a reasoning agent from a lucky guesser.

---

## Why the Diagnose Action Changes Everything

Here's what brute-forcing looks like on the reward function:

```
Brute-force: tries all 5 services
  → -2.0 × 4 wrong fix attempts
  → +6.0 on the lucky final hit
  = -2.0 total (optimistic: also trips -1.5 per fix aimed
    at a healthy service, so real brute force scores lower)
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

Not all failures are created equal. Four modes, four different twists:

| Mode | Correct Fix | The Catch |
|---|---|---|
| `crashed` | `restart_service` | Clean. Straightforward. |
| `memory_leak` | `restart_service` | Works - but it comes back 3 ticks after the fix. |
| `overloaded` | `scale_up` | Restart does nothing. Watch agents flail. |
| `bad_deploy` | `rollback_deploy` | Restart actively makes it worse. |

The `bad_deploy` mode is the one that breaks naive heuristics. If your agent's mental model is "crashed = restart," it'll restart a bad deploy and tank the health score further. This is intentional.

---

## Results

<!-- BEGIN GENERATED RESULTS (scripts/sync_readme_results.py) -->
| Agent | Success Rate | Diagnosis Acc. | Mean Reward |
|---|---|---|---|
| Random | 22% | 3% | -21.69 |
| Heuristic (log-aware) | 100% | 100% | 55.98 |

Measured over n=100 episodes per agent, seed=12345, `max_steps=30` - the same horizon the demo runs. Measurement fingerprint `sha256:78a08d089b95e0ad`. Reproduce with `python eval/evaluate.py`; raw output is committed at [`eval/results.json`](eval/results.json).

Episodes run across worker processes on one machine. Each episode's seed derives from the base seed and its index, so the numbers above are identical at any worker count - verified byte-identical at 1, 4 and 8. Timing is not recorded here because wall-clock is not reproducible; see [`eval/bench.json`](eval/bench.json).

LLM agents are deliberately absent from this table. They call a live third-party API, so their scores are not reproducible from this repo alone.
<!-- END GENERATED RESULTS -->

**Why the Random figures changed.** Earlier published Random numbers were
`26% / 4% / -18.08`. They differed because the old sequential runner seeded the
global `random` module **once per agent** and let a single RNG stream span all
100 episodes - so episode *n*'s actions depended on how many episodes had run
before it. That is order-dependent by construction and cannot survive being split
across workers. The runner now derives a separate seed per episode index, making
each episode self-contained and the aggregate order-independent; this is verified
byte-identical at 1, 4 and 8 workers. The Heuristic figures are unchanged, because
that agent draws no randomness. The difference is a fix to the measurement
method, not a change in agent behaviour.

**What the fingerprint covers, and why it had to grow.** `measurement_fingerprint`
hashes `env/`, `agent/` **and** `eval/evaluate.py`. It originally covered only the
code under test, which was not enough: the change described above moved the
published Random numbers with no change whatsoever to `env/` or `agent/`. A
fingerprint excluding the runner would have certified an artifact it could not
vouch for - the sequential and parallel results are different measurements of
identical code, and a fingerprint that cannot tell them apart is not evidence of
anything. The runner's seeding and aggregation are part of the measurement, so
they are part of the hash. `scripts/sync_readme_results.py` recomputes it and
fails if the artifact does not match the current tree, so a stale artifact is
rejected rather than republished.

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

The environment implements `reset()` / `step()` / `render()` with Gym-style return types, and `openenv.yaml` describes it. That interface is what the repo demonstrates; conformance to the full OpenEnv spec is not verified here. Drop in any agent exposing `act(obs) -> dict` and it runs.

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

**Evaluation runner.** `eval/evaluate.py` spreads episodes across worker
processes on the machine it runs on. This is local multiprocessing via the
standard library's `concurrent.futures` - there is no cluster, no scheduler, and
no distributed framework, and adding machines does nothing.

```bash
python eval/evaluate.py                 # workers default to CPU count
python eval/evaluate.py --workers 1     # serial
python eval/evaluate.py --no-speedup    # skip the 1-worker baseline pass
```

Each episode's seed derives from the base seed and the episode index, so results
are identical at any worker count - only wall-clock changes. Completed episodes
are written to `eval/.runs/` (gitignored), so an interrupted run resumes and
finishes the remainder rather than starting over.

`--bench` additionally times a 1-worker pass against an N-worker pass and writes
`eval/bench.json`. That file is gitignored by default, because wall-clock is not
reproducible and would dirty `results.json` on every run. One snapshot is
committed deliberately as a documented benchmark, with the machine it was
measured on:

| | |
|---|---|
| CPU | 12th Gen Intel Core i5-12450HX, 12 logical cores |
| RAM | 11 GB |
| Python | 3.11.15, Linux x86_64 |
| Episodes | 100 per agent, 2 agents |
| 1 worker | 0.504 s |
| 12 workers | 0.173 s |
| **Speedup** | **2.91x** |

Well short of 12x, and that is expected: the whole workload is under a second, so
process startup and result marshalling are a large fraction of it. The gap
narrows as the run grows - measured on the same machine, roughly 3.9x at 1000
episodes per agent and 4.6x at 4000. Re-run `--bench` on your own hardware rather
than trusting these numbers; they describe one laptop.

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
