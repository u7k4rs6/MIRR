#!/usr/bin/env python3
"""Regenerate the README Results block from eval/results.json.

The README and the app both publish the same baseline numbers. The app reads the
artifact directly; Markdown cannot, so this script is the README's reader. Two
surfaces holding the same numbers with only one updated is exactly how the
fabricated results table survived for months.

    python scripts/sync_readme_results.py            # rewrite the block
    python scripts/sync_readme_results.py --check    # exit 1 if out of date
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.fingerprint import measurement_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "eval" / "results.json"

BEGIN = "<!-- BEGIN GENERATED RESULTS (scripts/sync_readme_results.py) -->"
END = "<!-- END GENERATED RESULTS -->"

LABELS = {"Heuristic": "Heuristic (log-aware)"}

# Mirrors the payload written by eval/evaluate.py. No timestamp field: the
# artifact is identified by content, so re-running leaves the tree clean.
REQUIRED_TOP = (
    "results",
    "seed",
    "episodes_per_agent",
    "max_steps",
    "measurement_fingerprint",
    "measurement_git_sha",
    "measurement_git_dirty",
)
REQUIRED_PER_AGENT = ("success_rate", "diagnosis_accuracy", "mean_reward")


def fail(msg: str) -> "NoReturn":
    """Every abnormal exit goes through here: a guard that cannot find what it is
    checking must fail loudly, never report success."""
    print(f"sync_readme_results: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_results() -> dict:
    if not RESULTS.is_file():
        fail(f"{RESULTS} is missing - run: python eval/evaluate.py")
    try:
        d = json.loads(RESULTS.read_text())
    except ValueError as e:
        fail(f"{RESULTS} is not valid JSON: {e}")
    if not isinstance(d, dict):
        fail(f"{RESULTS} must contain a JSON object, got {type(d).__name__}")
    missing = [k for k in REQUIRED_TOP if k not in d]
    if missing:
        fail(f"{RESULTS} is missing required key(s): {', '.join(missing)}")
    results = d["results"]
    if not isinstance(results, dict) or not results:
        fail(
            f"{RESULTS} has no measured agents - refusing to publish an empty "
            "results table. Run: python eval/evaluate.py"
        )
    recorded = d["measurement_fingerprint"]
    actual = measurement_fingerprint()
    if recorded != actual:
        fail(
            f"{RESULTS} was produced by a different measurement.\n"
            f"  recorded: {recorded}\n"
            f"  current:  {actual}\n"
            "The code under test or the runner changed since this artifact was "
            "written. Re-run: python eval/evaluate.py"
        )
    for agent, r in results.items():
        if not isinstance(r, dict):
            fail(f"{RESULTS}: entry for {agent!r} is not an object")
        bad = [k for k in REQUIRED_PER_AGENT if not isinstance(r.get(k), (int, float))]
        if bad:
            fail(f"{RESULTS}: agent {agent!r} missing/non-numeric: {', '.join(bad)}")
    return d


def render() -> str:
    d = load_results()
    rows = [
        "| Agent | Success Rate | Diagnosis Acc. | Mean Reward |",
        "|---|---|---|---|",
    ]
    for name, r in d["results"].items():
        rows.append(
            f"| {LABELS.get(name, name)} | {r['success_rate'] * 100:.0f}% | "
            f"{r['diagnosis_accuracy'] * 100:.0f}% | {r['mean_reward']} |"
        )
    rows += [
        "",
        f"Measured over n={d['episodes_per_agent']} episodes per agent, "
        f"seed={d['seed']}, `max_steps={d['max_steps']}` - the same horizon the demo "
        f"runs. Measurement fingerprint `{d['measurement_fingerprint']}`. Reproduce with "
        "`python eval/evaluate.py`; raw output is committed at "
        "[`eval/results.json`](eval/results.json).",
        "",
        "Episodes run across worker processes on one machine. Each episode's seed "
        "derives from the base seed and its index, so the numbers above are "
        "identical at any worker count - verified byte-identical at 1, 4 and 8. "
        "Timing is not recorded here because wall-clock is not reproducible; see "
        "[`eval/bench.json`](eval/bench.json).",
        "",
        "LLM agents are deliberately absent from this table. They call a live "
        "third-party API, so their scores are not reproducible from this repo alone.",
    ]
    return "\n".join(rows)


def splice(text: str, body: str) -> str:
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        fail(
            f"expected exactly one {BEGIN} ... {END} pair in {README.name} "
            f"(found {text.count(BEGIN)} begin / {text.count(END)} end markers)"
        )
    if text.index(BEGIN) > text.index(END):
        fail(f"{README.name}: END marker appears before BEGIN")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{body}\n{END}{tail}"


def main() -> int:
    if not README.is_file():
        fail(f"{README} is missing")
    current = README.read_text()
    body = render()
    if len([ln for ln in body.splitlines() if ln.startswith("| ")]) < 3:
        fail("rendered results block has no data rows - refusing to publish it")
    updated = splice(current, body)
    if "--check" in sys.argv:
        if current != updated:
            print(
                "README.md Results block is out of date with eval/results.json.\n"
                "Run: python scripts/sync_readme_results.py",
                file=sys.stderr,
            )
            return 1
        print("README.md Results block matches eval/results.json")
        return 0
    if current != updated:
        README.write_text(updated)
        print("README.md Results block updated from eval/results.json")
    else:
        print("README.md already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
