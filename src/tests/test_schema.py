from triage import schema as s
from triage.model import load_pipeline


def test_schema_matches_pipeline_features():
    expected = set(load_pipeline().feature_names_in_)
    assert s.ALL_FIELDS == expected


def test_hx_flags_are_binary_ints_and_counted():
    assert len(s.HISTORY_FLAGS) == len(set(s.HISTORY_FLAGS))
    assert all(h.startswith("hx_") for h in s.HISTORY_FLAGS)
