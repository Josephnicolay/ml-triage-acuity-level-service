import copy

import pytest

from triage import validate
from triage import schema as s
from app.presets import PRESETS


@pytest.fixture
def base():
    return copy.deepcopy(next(iter(PRESETS.values())))


def test_presets_are_valid(base):
    for raw in PRESETS.values():
        r = validate(copy.deepcopy(raw))
        assert r.ok, r.errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", -1), ("age", 121), ("weight_kg", 0), ("height_cm", 29), ("systolic_bp", 0),
        ("heart_rate", 301), ("respiratory_rate", 101), ("temperature_c", 24.9),
        ("temperature_c", 45.1), ("spo2", 100.1), ("spo2", -1), ("gcs_total", 2), ("gcs_total", 16),
        ("pain_score", -2), ("pain_score", 11), ("news2_score", 21), ("num_comorbidities", -1),
        ("age", 40.5), ("age", "forty"), ("sex", "X"), ("arrival_mode", "teleport"), ("hx_copd", 2),
    ],
)
def test_hard_rejects(base, field, value):
    base[field] = value
    r = validate(base)
    assert not r.ok
    assert any(e.startswith(field) for e in r.errors)


def test_diastolic_above_systolic_warns_but_passes(base):
    base["systolic_bp"], base["diastolic_bp"] = 100.0, 120.0
    r = validate(base)
    assert r.ok and any(w.startswith("diastolic_bp") for w in r.warnings)


def test_soft_bounds_warn_but_pass(base):
    base["age"] = 100
    r = validate(base)
    assert r.ok and any(w.startswith("age") for w in r.warnings)


def test_pain_not_recorded_sentinel_is_valid(base):
    base["pain_score"] = s.PAIN_NOT_RECORDED
    r = validate(base)
    assert r.ok and not r.warnings


def test_optional_vitals_may_be_blank(base):
    for f in ("systolic_bp", "diastolic_bp", "respiratory_rate", "temperature_c"):
        base[f] = None
    r = validate(base)
    assert r.ok and r.clean["systolic_bp"] is None


def test_required_field_missing(base):
    del base["heart_rate"]
    assert not validate(base).ok


def test_text_rules(base):
    base["chief_complaint_raw"] = "   "
    assert not validate(base).ok
    base["chief_complaint_raw"] = "x" * (s.MAX_TEXT_LEN + 1)
    assert not validate(base).ok
    base["chief_complaint_raw"] = "x" * (s.SOFT_TEXT_LEN + 1)
    r = validate(base)
    assert r.ok and r.warnings


def test_unknown_field_rejected(base):
    base["patient_id"] = 123
    assert not validate(base).ok


def test_error_messages_do_not_echo_free_text(base):
    base["chief_complaint_raw"] = "SECRET-NAME " * 30
    r = validate(base)
    assert not any("SECRET-NAME" in m for m in r.errors + r.warnings)
