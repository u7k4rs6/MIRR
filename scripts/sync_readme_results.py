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

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "eval" / "results.json"

BEGIN = "<!-- BEGIN GENERATED RESULTS (scripts/sync_readme_results.py) -->"
END = "<!-- END GENERATED RESULTS -->"

LABELS = {"Heuristic": "Heuristic (log-aware)"}


def render() -> str:
    d = json.loads(RESULTS.read_text())
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
        f"runs. Code fingerprint `{d['code_fingerprint']}`. Reproduce with "
        "`python eval/evaluate.py`; raw output is committed at "
        "[`eval/results.json`](eval/results.json).",
        "",
        "LLM agents are deliberately absent from this table. They call a live "
        "third-party API, so their scores are not reproducible from this repo alone.",
    ]
    return "\n".join(rows)


def splice(text: str, body: str) -> str:
    if BEGIN not in text or END not in text:
        sys.exit(f"markers missing from {README.name}: expected {BEGIN} ... {END}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{body}\n{END}{tail}"


def main() -> int:
    current = README.read_text()
    updated = splice(current, render())
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
