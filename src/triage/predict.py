"""Validate -> predict. No input values are logged or stored."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import build_frame, load_pipeline
from .validation import validate


@dataclass
class PredictionResult:
    acuity: int | None = None  # 1 = most urgent ... 5 = least urgent
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.acuity is not None


def predict(raw: dict[str, Any]) -> PredictionResult:
    v = validate(raw)
    if not v.ok:
        return PredictionResult(errors=v.errors, warnings=v.warnings)
    label = load_pipeline().predict(build_frame(v.clean))[0]
    return PredictionResult(acuity=int(label), warnings=v.warnings)
