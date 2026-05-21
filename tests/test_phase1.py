"""Smoke tests for Phase-1 scaffolding: connectors, no-show, API."""

import json
from pathlib import Path

import pandas as pd
import pytest

from bi_forecast.connectors import parse_hl7_message, hl7_to_appointment, hl7_to_admission, fhir_bundle_to_dataframe
from bi_forecast.noshow import NoShowClassifier, synthesize_appointments


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ---------- HL7 ----------

def test_parse_siu():
    text = (EXAMPLES / "hl7" / "siu_s12_sample.hl7").read_text()
    msg = parse_hl7_message(text)
    assert msg.message_type.startswith("SIU")
    row = hl7_to_appointment(msg)
    assert row["appointment_id"] == "APPT-1001"
    assert row["patient_id"] == "MRN12345"
    assert row["patient_name"] == "Doe"
    assert row["location"] == "CARDIO"
    assert row["source"] == "HL7:SIU"


def test_parse_adt():
    text = (EXAMPLES / "hl7" / "adt_a01_sample.hl7").read_text()
    msg = parse_hl7_message(text)
    assert msg.message_type.startswith("ADT")
    row = hl7_to_admission(msg)
    assert row["patient_id"] == "MRN67890"
    assert row["patient_class"] == "I"
    assert row["source"] == "HL7:ADT"


def test_hl7_rejects_non_hl7():
    with pytest.raises(ValueError):
        parse_hl7_message("hello world")


# ---------- FHIR ----------

def test_fhir_bundle_flatten():
    bundle = json.loads((EXAMPLES / "fhir" / "appointment_bundle.json").read_text())
    df = fhir_bundle_to_dataframe(bundle, resource_type="Appointment")
    assert len(df) == 2
    assert set(df["appointment_id"]) == {"appt-1001", "appt-1002"}
    assert df.iloc[0]["patient_id"] == "12345"
    assert df.iloc[0]["doctor"] == "1234"


# ---------- No-show classifier ----------

def test_noshow_fit_and_predict():
    df = synthesize_appointments(n=500, seed=1)
    clf = NoShowClassifier()
    evaluation = clf.fit(df, target="no_show")
    assert 0.5 <= evaluation.auc <= 1.0
    assert evaluation.feature_importance

    upcoming = synthesize_appointments(n=20, seed=2).drop(columns=["no_show"])
    scores = clf.predict(upcoming)
    assert len(scores) == 20
    for s in scores:
        assert 0.0 <= s.probability <= 1.0
        assert s.risk_band in {"low", "medium", "high"}
    actions = clf.recommend(scores)
    assert all("action" in a for a in actions)


# ---------- FastAPI ----------

def test_api_health_and_forecast():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True

    series = []
    base = pd.Timestamp("2025-01-01")
    for i in range(80):
        series.append({"ds": (base + pd.Timedelta(days=i)).isoformat(), "y": 50 + (i % 7) * 2 + i * 0.1})
    body = {"target": "admissions", "series": series, "horizon": 7}
    r = client.post("/forecast", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["model"] in {"seasonal_naive", "ets", "sarima", "gbm"}
    assert len(payload["forecast"]) == 7
    assert payload["recommendations"]


def test_api_noshow_and_hooks():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    client = TestClient(create_app())

    r = client.post("/noshow/retrain")
    assert r.status_code == 200
    assert 0.5 <= r.json()["auc"] <= 1.0

    appts = synthesize_appointments(n=5, seed=99).drop(columns=["no_show"]).to_dict(orient="records")
    r = client.post("/noshow/score", json={"appointments": appts})
    assert r.status_code == 200
    assert r.json()["n"] == 5

    hl7 = (EXAMPLES / "hl7" / "siu_s12_sample.hl7").read_text()
    r = client.post("/hooks/hl7", json={"message": hl7})
    assert r.status_code == 200
    assert r.json()["type"] == "appointment"

    fhir = json.loads((EXAMPLES / "fhir" / "appointment_bundle.json").read_text())
    r = client.post("/hooks/fhir", json={"resource": fhir, "resource_type": "Appointment"})
    assert r.status_code == 200
    assert r.json()["type"] == "bundle"
    assert len(r.json()["rows"]) == 2
