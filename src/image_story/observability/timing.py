"""Timing helpers used by the orchestrator and benchmarks."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    """Accumulates per-stage wall-clock durations.

    Replaces the ad-hoc `self._runtime[...] = time.perf_counter() - t0` pattern
    that was repeated at every stage boundary in the orchestrator.
    """

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(time.perf_counter() - start, 2)

    def record(self, name: str, seconds: float) -> None:
        self.stages[name] = round(seconds, 2)

    def total(self) -> float:
        return round(sum(self.stages.values()), 2)

    def to_dict(self) -> dict[str, float]:
        return dict(self.stages)


@contextmanager
def timed(label: str, sink: dict[str, float] | None = None):
    """Time a block and optionally store the result in `sink` under `label`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - start, 2)
        if sink is not None:
            sink[label] = elapsed
