"""
Telemetry adapter: default reads the live Simulator; optional file replay for demos.

Define a thin protocol so you can swap implementations without changing the env core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class TelemetryPort(Protocol):
    """Minimal surface for 'fake observability' layers."""

    def snapshot_metrics(self) -> dict[str, Any]:
        ...


class SimulatorTelemetry:
    """Wraps Simulator.get_noisy_metrics()."""

    def __init__(self, sim: Any):
        self._sim = sim

    def snapshot_metrics(self) -> dict[str, Any]:
        return dict(self._sim.get_noisy_metrics())


class FileTelemetry:
    """
    Reads the last object from a JSONL file produced offline (e.g. anonymized prod snapshot).
    Each line should be a JSON object with a 'metrics' key shaped like the env metrics dict.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cache: Optional[dict[str, Any]] = None

    def snapshot_metrics(self) -> dict[str, Any]:
        if self._cache is not None:
            return dict(self._cache)
        if not self.path.is_file():
            return {}
        last: dict[str, Any] = {}
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        metrics = last.get("metrics") if isinstance(last, dict) else None
        self._cache = metrics if isinstance(metrics, dict) else {}
        return dict(self._cache)
