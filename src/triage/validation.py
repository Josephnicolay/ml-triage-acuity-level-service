"""Input validation: hard errors (reject) and soft warnings (still predict).

Messages never echo free-text content, so they are safe to display or log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import schema as s


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clean: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _to_number(v: Any, integer: bool) -> float | int | None:
    """Return a number, or None if not convertible / not integral when required."""
    if isinstance(v, bool) or isinstance(v, str):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    if integer:
        if x != int(x):
            return None
        return int(x)
    return x


def _validate_numeric(f: s.NumField, raw: dict, res: ValidationResult) -> None:
    v = raw.get(f.name)
    if _is_missing(v):
        if f.optional:
            res.clean[f.name] = None
        else:
            res.errors.append(f"{f.name}: required.")
        return

    x = _to_number(v, f.integer)
    if x is None:
        kind = "an integer" if f.integer else "a number"
        res.errors.append(f"{f.name}: must be {kind}.")
        return

    if f.name == "pain_score":
        if x != s.PAIN_NOT_RECORDED and not 0 <= x <= 10:
            res.errors.append("pain_score: must be -1 (not recorded) or an integer from 0 to 10.")
            return
        res.clean[f.name] = x
        return

    below = x <= f.hard_min if f.hard_min_exclusive else x < f.hard_min
    if below or x > f.hard_max:
        lo = f"> {f.hard_min:g}" if f.hard_min_exclusive else f">= {f.hard_min:g}"
        res.errors.append(f"{f.name}: {x:g} is not physically plausible (must be {lo} and <= {f.hard_max:g}).")
        return

    if x < f.soft_min or x > f.soft_max:
        res.warnings.append(
            f"{f.name}: {x:g} is outside the range seen in training "
            f"({f.soft_min:g} to {f.soft_max:g}); the prediction may be unreliable."
        )
    res.clean[f.name] = x


def validate(raw: dict[str, Any]) -> ValidationResult:
    res = ValidationResult()

    unknown = sorted(set(raw) - s.ALL_FIELDS)
    if unknown:
        res.errors.append(f"Unknown fields: {', '.join(unknown)}.")

    for f in s.NUMERIC:
        _validate_numeric(f, raw, res)

    for name, choices in s.CATEGORICAL.items():
        v = raw.get(name)
        if v not in choices:
            res.errors.append(f"{name}: must be one of {', '.join(choices)}.")
        else:
            res.clean[name] = v

    for name in s.HISTORY_FLAGS:
        v = raw.get(name)
        if isinstance(v, bool):
            v = int(v)
        if v not in (0, 1):
            res.errors.append(f"{name}: must be 0 or 1.")
        else:
            res.clean[name] = int(v)

    text = raw.get(s.TEXT_FIELD)
    if not isinstance(text, str) or not text.strip():
        res.errors.append(f"{s.TEXT_FIELD}: required (non-empty text).")
    elif len(text) > s.MAX_TEXT_LEN:
        res.errors.append(f"{s.TEXT_FIELD}: too long (max {s.MAX_TEXT_LEN} characters).")
    else:
        text = text.strip()
        if len(text) > s.SOFT_TEXT_LEN:
            res.warnings.append(
                f"{s.TEXT_FIELD}: longer than any complaint seen in training "
                f"({s.SOFT_TEXT_LEN} characters); the prediction may be unreliable."
            )
        res.clean[s.TEXT_FIELD] = text

    sbp, dbp = res.clean.get("systolic_bp"), res.clean.get("diastolic_bp")
    if sbp is not None and dbp is not None and dbp > sbp:
        # Training data contains such rows (~2.6%), so warn instead of rejecting.
        res.warnings.append(
            "diastolic_bp is higher than systolic_bp, which is physiologically unusual; "
            "check the values (the prediction still runs)."
        )

    return res
