"""Model loading and input-frame construction."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import schema as s

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline_model.pkl"


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def load_pipeline():
    """Load the sklearn Pipeline once per process (a pickle: only load trusted files)."""
    return joblib.load(model_path())


def build_frame(clean: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame in the exact column order and dtypes used in training."""
    cols = list(load_pipeline().feature_names_in_)
    row: dict[str, Any] = {}
    for c in cols:
        v = clean[c]
        if c in s.INT_COLUMNS:
            row[c] = np.int64(v)
        elif c in s.FLOAT_COLUMNS:
            row[c] = np.float64(np.nan if v is None else v)
        else:
            row[c] = v
    return pd.DataFrame([row], columns=cols)
