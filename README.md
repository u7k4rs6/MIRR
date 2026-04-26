# Incident Response Agent — OpenEnv

**Meta × PyTorch Hackathon Submission**

> An LLM agent trained with GRPO to diagnose and resolve production incidents in a partially observable microservices environment. Reasoning quality is scored separately from fix success — making causal reasoning measurable and trainable.

## Links

| Deliverable | Link |
|-------------|------|
| HF Space (live demo) | Create a Space from this repo, then paste your URL here (e.g. `huggingface.co/spaces/YOUR_USERNAME/...`) |
| Training Notebook (Colab) | [Open in Colab](https://colab.research.google.com/drive/16Rq5AQ3yvXiKh_3Chs1fx41YK7isWNJp?usp=sharing) — or upload `train.ipynb` from this clone |
| Source / updates | [github.com/u7k4rs6/MIRR](https://github.com/u7k4rs6/MIRR) |
| Trained Model | Run **Step 3** in `train.ipynb` after training. Set `HF_TOKEN` + `HF_HUB_USERNAME` (or `HF_MODEL_REPO`). Default id: `YOUR_USERNAME/incident-response-grpo`. Optional: [new model repo](https://huggingface.co/new). |
| Episode rollouts (Dataset) | **Step 4** in `train.ipynb` — default id: `YOUR_USERNAME/incident-response-rollouts`, or set `HF_DATASET_REPO`. Optional: [new dataset repo](https://huggingface.co/new-dataset). |

### Hub uploads (after training)

In Colab (or locally), set a **write** token as `HF_TOKEN` and your Hub username as `HF_HUB_USERNAME` ([token settings](https://huggingface.co/settings/tokens)), then run **Step 3** (model) and **Step 4** (rollouts) at the end of `train.ipynb`. URLs will be `https://huggingface.co/YOUR_USERNAME/incident-response-grpo` and `https://huggingface.co/datasets/YOUR_USERNAME/incident-response-rollouts` unless you override with `HF_MODEL_REPO` / `HF_DATASET_REPO`. Copy `.env.example` to `.env` for local runs (`.env` is gitignored).

## Training Curves

### Reward Curve

![Reward Curve](training_curves/reward_curve.png)

### Loss Curve

![Loss Curve](training_curves/loss_curve.png)

## Environment Design

Five microservices, one silent failure. The agent must:

1. Observe degraded metrics (±15% noise)
2. Gather information via `check_logs()`
3. **Explicitly commit to a diagnosis** — scored separately from the fix
4. Apply the correct fix (`restart`, `rollback`, or `scale_up`)
5. Confirm recovery

### Why the `diagnose()` action matters

The reward gap between a reasoning agent and a brute-force guesser:

- Brute-force: tries all 5 services → `-2.0 × 4` wrong penalties + `+6.0` lucky fix = **-2.0**
- Reasoning: diagnoses correctly → `+8.0` + `+10.0` fix + `+20.0` success = **38.0+**

### Failure Modes

| Mode | Correct Fix | Twist |
|------|-------------|-------|
| `crashed` | restart | Clean fix |
| `memory_leak` | restart | Recurs after 4 steps |
| `overloaded` | scale_up | Restart has no effect |
| `bad_deploy` | rollback | Restart worsens health |

## Results

| Agent | Success Rate | Diagnosis Acc. | Mean Reward |
|-------|-------------|----------------|-------------|
| Random | 10% | 5% | -8.2 |
| Heuristic (log-aware) | ~68% | ~99% | ~81 |
| **Trained LLM** | **68%** | **61%** | **22.7** |

## Setup

```bash
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here
set PYTHONPATH=%CD%
python eval/evaluate.py
python app.py
```

On Linux or macOS, use `export GROQ_API_KEY=...` and `export PYTHONPATH="$(pwd)"` from the repo root so `python eval/evaluate.py` resolves the `env` and `agent` packages.

**HF Space:** add `GROQ_API_KEY` under Space secrets. The app listens on `PORT` (default `7860`).

## File Structure

```
openenv.yaml          — OpenEnv grader config
env/environment.py    — OpenEnv interface (reset/step/render)
env/simulator.py      — Hidden state, propagation, failure logic
agent/                — Random, heuristic, LLM agents
eval/evaluate.py      — Evaluation + curve generation
train.ipynb           — GRPO training notebook (Colab)
app.py                — Gradio demo
training_curves/      — Committed reward/loss PNGs
```

## Validation Checklist

- [ ] Public HF Space — test from a **logged-out** browser
- [ ] `openenv.yaml` at repo root
- [ ] `environment.py` implements `reset()` / `step()` / `render()`
- [ ] `training_curves/reward_curve.png` and `loss_curve.png` committed
- [ ] `train.ipynb` runnable; Colab link in this README
- [ ] README links and embedded plots updated for judges

Double-check every link in a **logged-out** browser before submit. Confirm the **model** and **dataset** repos exist after you run the Hub cells in Colab.
