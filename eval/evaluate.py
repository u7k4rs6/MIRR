import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

from agent.heuristic_agent import HeuristicAgent  # noqa: E402
from agent.random_agent import RandomAgent  # noqa: E402
from env.environment import IncidentResponseEnv  # noqa: E402

# Fixed for reproducibility. RandomAgent draws from the `random` module, so the
# module-level RNG has to be seeded too — env seeding alone is not enough.
SEED = 12345
N_EPISODES = 100
# Match the horizon the app actually runs: every app.py entry point constructs
# IncidentResponseEnv(max_steps=30). The class default is 20, and measuring at 20
# would publish numbers that do not describe what a visitor gets. The efficiency
# bonus scales with the horizon, so this materially changes mean reward.
MAX_STEPS = 30
RESULTS_PATH = ROOT / "eval" / "results.json"


def run_episodes(agent, n=N_EPISODES, seed_offset=0, max_steps=MAX_STEPS):
    env = IncidentResponseEnv(max_steps=max_steps)
    rewards, successes, diag_correct = [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed_offset + i)
        if hasattr(agent, "reset"):
            agent.reset()
        total_reward = 0.0
        done = False
        info = {}
        while not done:
            action = agent.act(obs)
            obs, r, done, _, info = env.step(action)
            total_reward += r
        rewards.append(total_reward)
        successes.append(1 if info.get("outcome") == "success" else 0)
        diag_correct.append(1 if info.get("diagnosis_correct") else 0)
    return rewards, successes, diag_correct


def _code_fingerprint() -> str:
    """Content hash of the code under test (env/ + agent/). Unlike a commit SHA,
    this is stable across rebase/amend and can be recomputed by anyone, so the
    artifact identifies the exact code it measured rather than pointing at a
    commit that may no longer exist."""
    h = hashlib.sha256()
    files = sorted(
        list((ROOT / "env").glob("*.py")) + list((ROOT / "agent").glob("*.py"))
    )
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return False


def run_leaderboard(n=N_EPISODES, seed=SEED):
    """Measure the deterministic baselines. LLM agents are not evaluated here:
    they depend on a live third-party API, so their numbers are not reproducible
    from this repo alone."""
    print(f"\n=== LEADERBOARD (n={n} per agent, seed={seed}, max_steps={MAX_STEPS}) ===")
    results = {}
    for name, factory in [("Random", RandomAgent), ("Heuristic", HeuristicAgent)]:
        random.seed(seed)
        np.random.seed(seed)
        rewards, successes, diag = run_episodes(factory(), n=n, seed_offset=seed)
        results[name] = {
            "mean_reward": round(float(np.mean(rewards)), 2),
            "success_rate": round(float(np.mean(successes)), 3),
            "diagnosis_accuracy": round(float(np.mean(diag)), 3),
        }
        print(f"{name}: {json.dumps(results[name])}")

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_fingerprint": _code_fingerprint(),
        "git_sha_at_measurement": _git_sha(),
        "git_dirty_at_measurement": _git_dirty(),
        "seed": seed,
        "episodes_per_agent": n,
        "max_steps": MAX_STEPS,
        "agents_evaluated": ["Random", "Heuristic"],
        "results": results,
    }
    return payload


if __name__ == "__main__":
    payload = run_leaderboard()
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {RESULTS_PATH.relative_to(ROOT)}")
