"""Single source of truth for the model's input contract.

Soft bounds come from the observed range in ``data/processed_train.csv``.
Hard bounds mark physically impossible values (rejected). They are engineering
sanity checks for a demo, not clinical guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

PAIN_NOT_RECORDED = -1  # sentinel present in the training data

MAX_TEXT_LEN = 200  # hard limit (abuse/cost guard)
SOFT_TEXT_LEN = 77  # longest complaint seen in training


@dataclass(frozen=True)
class NumField:
    name: str
    integer: bool
    hard_min: float
    hard_max: float
    soft_min: float
    soft_max: float
    optional: bool = False
    hard_min_exclusive: bool = False  # True -> value must be > hard_min
    unit: str = ""


NUMERIC: tuple[NumField, ...] = (
    NumField("age", True, 0, 120, 1, 94, unit="years"),
    NumField("weight_kg", False, 0, 500, 2, 148.5, hard_min_exclusive=True, unit="kg"),
    NumField("height_cm", False, 30, 250, 45, 210, unit="cm"),
    NumField("systolic_bp", False, 0, 300, 40, 226.9, optional=True, hard_min_exclusive=True, unit="mmHg"),
    NumField("diastolic_bp", False, 0, 300, 20, 134.8, optional=True, hard_min_exclusive=True, unit="mmHg"),
    NumField("heart_rate", False, 0, 300, 30, 207.7, hard_min_exclusive=True, unit="bpm"),
    NumField("respiratory_rate", False, 0, 100, 8, 51.5, optional=True, hard_min_exclusive=True, unit="/min"),
    NumField("temperature_c", False, 25, 45, 35.1, 41.8, optional=True, unit="°C"),
    NumField("spo2", False, 0, 100, 60.4, 100, unit="%"),
    NumField("num_active_medications", True, 0, 1000, 0, 20),
    NumField("num_comorbidities", True, 0, 1000, 0, 20),
    NumField("gcs_total", True, 3, 15, 3, 15),
    # pain_score is special-cased: -1 (not recorded) or integer 0..10
    NumField("pain_score", True, -1, 10, -1, 10),
    NumField("news2_score", True, 0, 20, 0, 17),
    NumField("num_prior_ed_visits_12m", True, 0, 1000, 0, 11),
    NumField("num_prior_admissions_12m", True, 0, 1000, 0, 9),
)

CATEGORICAL: dict[str, tuple[str, ...]] = {
    "sex": ("F", "M", "Other"),
    "arrival_mode": ("ambulance", "brought_by_family", "helicopter", "police", "transfer", "walk-in"),
    "mental_status_triage": ("agitated", "alert", "confused", "drowsy", "unresponsive"),
    "pain_location": ("abdomen", "back", "chest", "extremity", "head", "multiple", "none", "pelvis", "unknown"),
    "chief_complaint_system": (
        "ENT", "cardiovascular", "dermatological", "endocrine", "gastrointestinal",
        "genitourinary", "infectious", "musculoskeletal", "neurological", "ophthalmic",
        "other", "psychiatric", "respiratory", "trauma",
    ),
    "transport_origin": ("home", "nursing_home", "other_hospital", "outdoor", "public_space", "school", "workplace"),
}

TEXT_FIELD = "chief_complaint_raw"

HISTORY_FLAGS: tuple[str, ...] = (
    "hx_hypertension", "hx_diabetes_type2", "hx_diabetes_type1", "hx_asthma", "hx_copd",
    "hx_heart_failure", "hx_atrial_fibrillation", "hx_ckd", "hx_liver_disease", "hx_malignancy",
    "hx_obesity", "hx_depression", "hx_anxiety", "hx_dementia", "hx_epilepsy",
    "hx_hypothyroidism", "hx_hyperthyroidism", "hx_hiv", "hx_coagulopathy", "hx_immunosuppressed",
    "hx_pregnant", "hx_substance_use_disorder", "hx_coronary_artery_disease", "hx_stroke_prior",
    "hx_peripheral_vascular_disease",
)

NUMERIC_BY_NAME = {f.name: f for f in NUMERIC}
INT_COLUMNS = frozenset([f.name for f in NUMERIC if f.integer] + list(HISTORY_FLAGS))
FLOAT_COLUMNS = frozenset(f.name for f in NUMERIC if not f.integer)
ALL_FIELDS: frozenset[str] = frozenset(
    list(NUMERIC_BY_NAME) + list(CATEGORICAL) + [TEXT_FIELD] + list(HISTORY_FLAGS)
)

# Acuity scale of the target (triage_acuity): 1 = most urgent ... 5 = least urgent.
ACUITY_MOST_URGENT = 1
ACUITY_LEAST_URGENT = 5
