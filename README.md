# MIRR - Microservice Incident Response & Recovery

> *At 3 AM, your payment service starts failing. Orders are queuing. Health checks are lying. You have five microservices, one silent killer, and no idea where to start.*
>
> *Your agent has 30 steps.*

---

## What is this?

A playground for agents that think under pressure.

MIRR is a partially observable microservice environment where classical rules-based agents, LLM agents, and GRPO-trained models compete to diagnose and recover a broken distributed system - before it cascades into total failure.

Five services. One hidden fault. Noisy metrics. A diagnosis action that forces the agent to commit its reasoning before it acts.

You can watch it happen live in the Gradio demo, step through episodes frame by frame, or train your own model and see the reward curves climb.

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

## Training Curves

<table>
<tr>
<td><img src="training_curves/reward_curve.png" alt="Reward Curve" width="400"/></td>
<td><img src="training_curves/loss_curve.png" alt="Loss Curve" width="400"/></td>
</tr>
<tr>
<td align="center"><em>Reward Curve</em></td>
<td align="center"><em>Loss Curve</em></td>
</tr>
</table>

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

| Agent | Success Rate | Diagnosis Acc. | Mean Reward |
|---|---|---|---|
| Random | 10% | 5% | -8.2 |
| Heuristic (log-aware) | ~68% | ~99% | ~81 |
| Trained LLM | 68% | 61% | 22.7 |

The heuristic agent has near-perfect diagnosis accuracy because it directly pattern-matches logs - it knows exactly what to look for. The trained LLM matches its success rate but gets there differently: messier diagnosis, better generalization. The gap in diagnosis accuracy (99% vs 61%) while achieving the same success rate tells you something interesting about how LLMs recover from wrong beliefs mid-episode.

---

## Environment Design

```
openenv.yaml          - Env metadata (id, thresholds, service list)
env/environment.py    - Episodic API: reset / step / render
env/simulator.py      - Hidden state, failure propagation, health logic
agent/                - Random, heuristic, and LLM agents
eval/evaluate.py      - Evaluation loop + curve generation
train.ipynb           - GRPO training notebook (Colab-ready)
app.py                - Gradio live demo
training_curves/      - reward_curve.png, loss_curve.png
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
- [ ] `training_curves/reward_curve.png` and `loss_curve.png` committed
- [ ] `train.ipynb` runnable end-to-end; Colab copy in sync
- [ ] README links point at live URLs

---

## Why I built this

Most RL environments are either too clean (CartPole, Atari) or too opaque (production infra you can't open up).

MIRR sits in the middle - messy enough that brute force fails, structured enough that you can actually measure reasoning. The `diagnose()` action exists because I wanted to see if forcing an explicit commitment step changed how agents behave. It does.

GRPO training hooks are built in. Bring your own model, point it at the rollout format, and watch whether it learns to think before it acts.

---

<p align="center">
  <sub>Built by <a href="https://github.com/u7k4rs6">Utkarsh Bahuguna</a> &nbsp;·&nbsp; PRs welcome &nbsp;·&nbsp; Star if it taught you something</sub>
</p>
