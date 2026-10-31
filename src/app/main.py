"""Streamlit demo UI. All logic lives in the `triage` package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.presets import PRESETS  # noqa: E402
from triage import predict, schema as s  # noqa: E402
from triage.model import load_pipeline  # noqa: E402

st.set_page_config(page_title="Triage acuity demo", page_icon=None, layout="wide")

CUSTOM = "(blank form)"
LEVEL_HELP = "Acuity scale: **1 = most urgent**, **5 = least urgent**."
BLANK = {
    "age": 40, "weight_kg": 70.0, "height_cm": 170.0, "systolic_bp": 120.0, "diastolic_bp": 80.0,
    "heart_rate": 80.0, "respiratory_rate": 16.0, "temperature_c": 36.8, "spo2": 98.0,
    "num_active_medications": 0, "num_comorbidities": 0, "gcs_total": 15, "pain_score": 0,
    "news2_score": 0, "num_prior_ed_visits_12m": 0, "num_prior_admissions_12m": 0,
    "sex": "F", "arrival_mode": "walk-in", "mental_status_triage": "alert", "pain_location": "none",
    "chief_complaint_system": "other", "transport_origin": "home", "chief_complaint_raw": "",
    **{h: 0 for h in s.HISTORY_FLAGS},
}


@st.cache_resource(show_spinner="Loading model...")
def _warm():
    return load_pipeline()


_warm()

st.title("Triage acuity prediction (demo)")
st.warning(
    "Demo only, using a baseline model trained on synthetic data. Not medical advice and not for "
    "clinical use. **Do not enter real patient information.**"
)

preset_name = st.selectbox("Start from an example", [CUSTOM, *PRESETS], key="preset")
if st.session_state.get("_loaded") != preset_name:
    src = BLANK if preset_name == CUSTOM else PRESETS[preset_name]
    for k, v in src.items():
        st.session_state[f"f_{k}"] = bool(v) if k in s.HISTORY_FLAGS else v
    for k in ("systolic_bp", "diastolic_bp", "respiratory_rate", "temperature_c"):
        st.session_state[f"f_{k}"] = src[k]
        st.session_state[f"na_{k}"] = src[k] is None
    st.session_state["pain_nr"] = src["pain_score"] == s.PAIN_NOT_RECORDED
    st.session_state["f_pain_score"] = max(src["pain_score"], 0)
    st.session_state["_loaded"] = preset_name


def num(name: str, label: str, *, step, fmt: str | None = None, optional=False):
    """Numeric input; bounds are not enforced by the widget so the validator decides."""
    key = f"f_{name}"
    if optional:
        na = st.checkbox(f"{label}: not recorded", key=f"na_{name}")
        if na:
            return None
        if st.session_state.get(key) is None:
            st.session_state[key] = BLANK[name]
    return st.number_input(label, key=key, step=step, format=fmt)


raw: dict = {}
with st.form("patient"):
    st.subheader("Vitals and scores")
    c = st.columns(3)
    with c[0]:
        raw["age"] = num("age", "Age (years)", step=1)
        raw["weight_kg"] = num("weight_kg", "Weight (kg)", step=0.5, fmt="%.1f")
        raw["height_cm"] = num("height_cm", "Height (cm)", step=0.5, fmt="%.1f")
        raw["heart_rate"] = num("heart_rate", "Heart rate (bpm)", step=1.0, fmt="%.1f")
        raw["spo2"] = num("spo2", "SpO2 (%)", step=0.5, fmt="%.1f")
    with c[1]:
        raw["systolic_bp"] = num("systolic_bp", "Systolic BP (mmHg)", step=1.0, fmt="%.1f", optional=True)
        raw["diastolic_bp"] = num("diastolic_bp", "Diastolic BP (mmHg)", step=1.0, fmt="%.1f", optional=True)
        raw["respiratory_rate"] = num("respiratory_rate", "Respiratory rate (/min)", step=1.0, fmt="%.1f", optional=True)
        raw["temperature_c"] = num("temperature_c", "Temperature (°C)", step=0.1, fmt="%.1f", optional=True)
    with c[2]:
        raw["gcs_total"] = num("gcs_total", "GCS total (3-15)", step=1)
        raw["news2_score"] = num("news2_score", "NEWS2 score", step=1)
        st.caption(
            "NEWS2 is entered manually. The data lacks supplemental-oxygen and hypercapnic "
            "respiratory failure information needed to compute it exactly, so keep it consistent "
            "with the vitals yourself."
        )
        pain_nr = st.checkbox("Pain score: not recorded", key="pain_nr")
        raw["pain_score"] = s.PAIN_NOT_RECORDED if pain_nr else st.number_input(
            "Pain score (0-10)", key="f_pain_score", step=1
        )

    st.subheader("Presentation")
    c = st.columns(3)
    for col, (name, choices) in zip(c * 2, s.CATEGORICAL.items()):
        with col:
            raw[name] = st.selectbox(name.replace("_", " ").capitalize(), choices, key=f"f_{name}")
    raw["chief_complaint_raw"] = st.text_input(
        "Chief complaint (free text)", key="f_chief_complaint_raw", max_chars=s.MAX_TEXT_LEN
    )

    st.subheader("History and utilization")
    c = st.columns(3)
    with c[0]:
        raw["num_active_medications"] = num("num_active_medications", "Active medications", step=1)
    with c[1]:
        raw["num_comorbidities"] = num("num_comorbidities", "Comorbidities", step=1)
    with c[2]:
        raw["num_prior_ed_visits_12m"] = num("num_prior_ed_visits_12m", "ED visits, last 12 months", step=1)
        raw["num_prior_admissions_12m"] = num("num_prior_admissions_12m", "Admissions, last 12 months", step=1)

    with st.expander("Medical history flags", expanded=False):
        cols = st.columns(4)
        for i, h in enumerate(s.HISTORY_FLAGS):
            with cols[i % 4]:
                raw[h] = int(st.checkbox(h[3:].replace("_", " "), key=f"f_{h}"))

    submitted = st.form_submit_button("Predict acuity", type="primary")

if submitted:
    result = predict(raw)
    for w in result.warnings:
        st.warning(w)
    if result.ok:
        st.metric("Predicted acuity level", result.acuity)
        st.caption(LEVEL_HELP)
    else:
        for e in result.errors:
            st.error(e)
