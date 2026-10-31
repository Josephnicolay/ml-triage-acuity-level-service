import copy
from pathlib import Path

import pandas as pd
import pytest

from triage import predict
from triage.model import load_pipeline
from app.presets import PRESETS

DATA = Path(__file__).resolve().parents[2] / "data" / "processed_train.csv"
EXPECTED = [1, 2, 3, 5]  # acuity of the preset rows in training; model reproduces them


def test_presets_predict_expected_levels():
    got = [predict(copy.deepcopy(r)).acuity for r in PRESETS.values()]
    assert got == EXPECTED


def test_optional_vitals_missing_still_predicts():
    raw = copy.deepcopy(next(iter(PRESETS.values())))
    for f in ("systolic_bp", "diastolic_bp", "respiratory_rate", "temperature_c"):
        raw[f] = None
    r = predict(raw)
    assert r.ok and r.acuity in {1, 2, 3, 4, 5}


def test_invalid_input_returns_errors_not_prediction():
    raw = copy.deepcopy(next(iter(PRESETS.values())))
    raw["spo2"] = 150
    r = predict(raw)
    assert not r.ok and r.errors


@pytest.mark.skipif(not DATA.exists(), reason="training data not available (e.g. in CI/container)")
def test_golden_rows_match_direct_pipeline_call():
    """API path must equal calling the pipeline directly on the same rows."""
    pipe = load_pipeline()
    df = pd.read_csv(DATA).dropna(subset=list(pipe.feature_names_in_))
    df = df[(df.pain_score >= -1) & (df.diastolic_bp <= df.systolic_bp)].sample(100, random_state=0)
    direct = pipe.predict(df[list(pipe.feature_names_in_)])
    via_api = []
    for _, row in df[list(pipe.feature_names_in_)].iterrows():
        r = predict(row.to_dict())
        assert r.ok, r.errors
        via_api.append(r.acuity)
    assert list(direct) == via_api
