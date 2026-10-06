"""Minimal run tracing.

A run is a single invocation of the orchestrator. The tracer records stage
order and timings so artifacts carry an execution trace without requiring an
external tracing backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .timing import StageTimer


@dataclass
class RunTrace:
    """Ordered record of what a run did and how long each stage took."""

    run_id: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    timings: StageTimer = field(default_factory=StageTimer)

    def event(self, stage: str, **detail: Any) -> None:
        self.events.append(
            {"stage": stage, "t": round(time.time(), 3), **detail}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": self.events,
            "timings": self.timings.to_dict(),
            "total_s": self.timings.total(),
        }
