import argparse
import hashlib
import json
import os
import random
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
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
# Per-episode records for resume. Gitignored: this is scratch state for an
# in-progress run, not a published artifact.
MANIFEST_ROOT = ROOT / "eval" / ".runs"
# Timing lives apart from results.json. Wall-clock is not reproducible, so mixing
# it into the published artifact would dirty the tree on every re-run. Written
# only under --bench, and gitignored unless deliberately committed.
BENCH_PATH = ROOT / "eval" / "bench.json"


AGENTS = {"Random": RandomAgent, "Heuristic": HeuristicAgent}


def _env_seed(base_seed: int, idx: int) -> int:
    """Seed for episode `idx`. Depends only on (base_seed, idx) - never on
    execution order - so an episode is identical at any worker count."""
    return base_seed + idx


def _agent_seed(base_seed: int, idx: int) -> int:
    """Separate deterministic stream for the agent's own RNG.

    RandomAgent draws from the global `random` module. Seeding it once per run
    and letting the stream continue across episodes would make every episode
    depend on how many ran before it - which is exactly what breaks when work is
    distributed over workers. Deriving a per-episode seed makes each episode
    self-contained and order-independent.
    """
    return (base_seed * 1_000_003 + idx * 31 + 17) % (2**32)


def run_one_episode(task: tuple) -> dict:
    """Run a single episode. Top-level and pure so it is picklable and carries no
    state between episodes. Returns a record, not an aggregate."""
    agent_name, idx, base_seed, max_steps = task

    # Each episode reseeds both RNGs from its own index.
    agent_rng_seed = _agent_seed(base_seed, idx)
    random.seed(agent_rng_seed)
    np.random.seed(agent_rng_seed)

    env = IncidentResponseEnv(max_steps=max_steps)
    agent = AGENTS[agent_name]()
    if hasattr(agent, "reset"):
        agent.reset()

    obs, _ = env.reset(seed=_env_seed(base_seed, idx))
    total_reward = 0.0
    done = False
    info = {}
    while not done:
        obs, r, done, _, info = env.step(agent.act(obs))
        total_reward += r
    return {
        "agent": agent_name,
        "index": idx,
        "reward": total_reward,
        "success": 1 if info.get("outcome") == "success" else 0,
        "diagnosis_correct": 1 if info.get("diagnosis_correct") else 0,
    }


def _run_id(n: int, seed: int, max_steps: int) -> str:
    """Identifies a run configuration. Resuming only reuses records produced by
    the same config AND the same code, so a code change starts a fresh run
    instead of silently mixing measurements."""
    key = f"{_code_fingerprint()}|{n}|{seed}|{max_steps}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _record_path(run_dir: Path, agent_name: str, idx: int) -> Path:
    return run_dir / f"{agent_name}-{idx:06d}.json"


def _write_record(run_dir: Path, rec: dict) -> None:
    """Atomic: write to a temp file then rename. A kill mid-write leaves either
    nothing or a complete record - never a truncated one that would be read back
    as valid on resume."""
    dst = _record_path(run_dir, rec["agent"], rec["index"])
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, dst)


def _load_records(run_dir: Path, agent_name: str, n: int) -> dict:
    """Completed episodes for this agent, keyed by index. Unreadable records are
    ignored so a corrupt file causes a re-run, not a crash or a bad aggregate."""
    done = {}
    for idx in range(n):
        f = _record_path(run_dir, agent_name, idx)
        if not f.is_file():
            continue
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        if rec.get("index") == idx and rec.get("agent") == agent_name:
            done[idx] = rec
    return done


def _chunks(tasks: list, workers: int) -> list:
    """Contiguous slices, one per worker, so IPC is paid per worker rather than
    per episode. Episodes are short; per-task dispatch would dominate."""
    if workers <= 1:
        return [tasks]
    size = max(1, (len(tasks) + workers - 1) // workers)
    return [tasks[i : i + size] for i in range(0, len(tasks), size)]


def _run_chunk(chunk: list) -> list:
    return [run_one_episode(task) for task in chunk]


def run_episodes(agent_name: str, n=N_EPISODES, base_seed=SEED, max_steps=MAX_STEPS,
                 workers: int = 1, run_dir: Path = None) -> list:
    """Run n episodes, optionally across a local process pool.

    Returns records sorted by episode index, never by completion order - so the
    aggregate is bit-identical regardless of worker count. When `run_dir` is set,
    each completed episode is written out and already-completed ones are skipped,
    making an interrupted run resumable.
    """
    done = _load_records(run_dir, agent_name, n) if run_dir else {}
    todo = [(agent_name, i, base_seed, max_steps) for i in range(n) if i not in done]
    if done:
        print(f"  {agent_name}: resuming, {len(done)}/{n} already complete")

    fresh = []
    if todo:
        if workers <= 1:
            for task in todo:
                rec = run_one_episode(task)
                if run_dir:
                    _write_record(run_dir, rec)
                fresh.append(rec)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for batch in pool.map(_run_chunk, _chunks(todo, workers)):
                    for rec in batch:
                        if run_dir:
                            _write_record(run_dir, rec)
                    fresh.extend(batch)

    records = list(done.values()) + fresh
    records.sort(key=lambda r: r["index"])
    return records


def _aggregate(records: list) -> dict:
    """Aggregate in index order. Float addition is not associative, so summing in
    completion order would make the mean depend on scheduling."""
    ordered = sorted(records, key=lambda r: r["index"])
    return {
        "mean_reward": round(float(np.mean([r["reward"] for r in ordered])), 2),
        "success_rate": round(float(np.mean([r["success"] for r in ordered])), 3),
        "diagnosis_accuracy": round(
            float(np.mean([r["diagnosis_correct"] for r in ordered])), 3
        ),
    }


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


# Git context is anchored to the measured code, not to the moment the script ran.
# A run-time HEAD changes on every unrelated commit, so the artifact would dirty
# the working tree on every re-run. These fields move only when env/ or agent/
# move - the same condition that changes code_fingerprint.
CODE_PATHS = ("env", "agent")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _code_git_sha() -> str:
    """Last commit that touched the code under test."""
    return _git("log", "-1", "--format=%h", "--", *CODE_PATHS) or "unknown"


def _code_git_dirty() -> bool:
    """True if the code under test has uncommitted changes."""
    return bool(_git("status", "--porcelain", "--", *CODE_PATHS))


def default_workers() -> int:
    return os.cpu_count() or 1


def _machine_spec() -> dict:
    """A speedup figure without hardware is not reproducible."""
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    return {
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _timed_pass(n: int, seed: int, workers: int, run_dir: Path) -> tuple:
    """One full pass over both agents. Returns (results, wall_clock_seconds).

    Timing excludes nothing: it is measured around the actual work, so the
    recorded speedup is what a user would observe, including pool startup.
    """
    started = time.perf_counter()
    results = {}
    for name in AGENTS:
        records = run_episodes(
            name, n=n, base_seed=seed, workers=workers, run_dir=run_dir
        )
        results[name] = _aggregate(records)
    return results, time.perf_counter() - started


def run_benchmark(n=N_EPISODES, seed=SEED, workers: int = 1) -> dict:
    """Measure wall-clock at 1 worker versus `workers`, on scratch manifests so a
    resumed run cannot make either pass look instant. Separate from
    run_leaderboard because timing is not part of the published result."""
    with tempfile.TemporaryDirectory() as scratch:
        _, serial_seconds = _timed_pass(n, seed, 1, Path(scratch))
    with tempfile.TemporaryDirectory() as scratch:
        _, parallel_seconds = _timed_pass(n, seed, workers, Path(scratch))
    return {
        "machine": _machine_spec(),
        "code_fingerprint": _code_fingerprint(),
        "seed": seed,
        "episodes_per_agent": n,
        "max_steps": MAX_STEPS,
        "workers": workers,
        "wall_clock_seconds_1_worker": round(serial_seconds, 3),
        "wall_clock_seconds": round(parallel_seconds, 3),
        "speedup_vs_1_worker": round(serial_seconds / parallel_seconds, 2),
    }


def run_leaderboard(n=N_EPISODES, seed=SEED, workers: int = 1, resume: bool = True):
    """Measure the deterministic baselines. LLM agents are not evaluated here:
    they depend on a live third-party API, so their numbers are not reproducible
    from this repo alone."""
    print(
        f"\n=== LEADERBOARD (n={n} per agent, seed={seed}, "
        f"max_steps={MAX_STEPS}, workers={workers}) ==="
    )
    run_dir = None
    if resume:
        run_dir = MANIFEST_ROOT / _run_id(n, seed, MAX_STEPS)
        run_dir.mkdir(parents=True, exist_ok=True)

    results, _ = _timed_pass(n, seed, workers, run_dir)
    for name, r in results.items():
        print(f"{name}: {json.dumps(r)}")

    # No timestamp: the artifact is identified by code_fingerprint, not by when it
    # was produced. A wall-clock field would make every re-run dirty the tree even
    # when nothing measured has changed.
    payload = {
        "code_fingerprint": _code_fingerprint(),
        "code_git_sha": _code_git_sha(),
        "code_git_dirty": _code_git_dirty(),
        "seed": seed,
        "episodes_per_agent": n,
        "max_steps": MAX_STEPS,
        "agents_evaluated": list(AGENTS),
        "results": results,
    }
    return payload


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Measure MIRR baselines.")
    ap.add_argument(
        "--workers", type=int, default=default_workers(),
        help="Local worker processes (default: CPU count). Results are identical "
             "at any worker count; only wall-clock changes.",
    )
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--bench", action="store_true",
                    help="Also measure 1-worker vs N wall-clock and write eval/bench.json. "
                         "Off by default: timing is not reproducible and does not belong "
                         "in results.json.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore and do not write per-episode records.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    payload = run_leaderboard(
        n=args.episodes, seed=args.seed, workers=args.workers,
        resume=not args.no_resume,
    )
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.out}")

    if args.bench:
        bench = run_benchmark(n=args.episodes, seed=args.seed, workers=args.workers)
        BENCH_PATH.write_text(json.dumps(bench, indent=2) + "\n")
        print(
            f"bench: {bench['wall_clock_seconds']}s at {bench['workers']} workers vs "
            f"{bench['wall_clock_seconds_1_worker']}s at 1 -> "
            f"{bench['speedup_vs_1_worker']}x"
        )
        print(f"Wrote {BENCH_PATH}")
