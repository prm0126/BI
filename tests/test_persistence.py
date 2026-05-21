"""Tests for the database persistence layer."""

import json
from pathlib import Path

import pandas as pd
import pytest

from bi_forecast.db import init_db, reset_for_tests, session_scope, repository
from bi_forecast.db.models import Patient, Appointment, Prediction, DeliveryLog, HL7Message, ModelVersion


@pytest.fixture(autouse=True)
def _fresh_inmemory_db():
    reset_for_tests("sqlite:///:memory:")


# ---------- Repository ----------

def test_upsert_patient_and_appointment_idempotent():
    with session_scope() as s:
        p1 = repository.upsert_patient(s, "MRN001", name="Jane", phone="+1555")
        p2 = repository.upsert_patient(s, "MRN001", name="Jane Doe")  # updates name
        assert p1.id == p2.id
        assert p2.name == "Jane Doe"
        assert p2.phone == "+1555"

        a1 = repository.upsert_appointment(s, "APT001", patient_id=p1.id, doctor="Dr. Smith")
        a2 = repository.upsert_appointment(s, "APT001", doctor="Dr. Smith Jr.")
        assert a1.id == a2.id and a2.doctor == "Dr. Smith Jr."


def test_record_prediction_and_query():
    with session_scope() as s:
        repository.record_prediction(s, "APT-NEW", probability=0.81, risk_band="high", top_factors=["lead_time_days"])
        repository.record_prediction(s, "APT-NEW", probability=0.85, risk_band="high", top_factors=["lead_time_days"])
    with session_scope() as s:
        rows = repository.recent_predictions(s)
        assert len(rows) == 2
        assert rows[0].probability >= rows[1].probability or rows[0].created_at >= rows[1].created_at


def test_ingest_dataframe():
    df = pd.DataFrame([
        {"appointment_id": "A1", "mrn": "M1", "name": "Jane", "phone": "+1555",
         "appointment_time": "2026-05-25 10:15", "doctor": "Dr. Smith", "location": "OPD-3"},
        {"appointment_id": "A2", "mrn": "M2", "name": "Sam", "phone": "+1556",
         "appointment_time": "2026-05-25 11:00", "doctor": "Dr. Khan", "location": "OPD-1"},
    ])
    with session_scope() as s:
        n = repository.ingest_appointment_dataframe(s, df, source="upload")
        assert n == 2
    with session_scope() as s:
        from sqlalchemy import select
        patients = list(s.scalars(select(Patient)))
        appointments = list(s.scalars(select(Appointment)))
        assert len(patients) == 2 and len(appointments) == 2
        assert appointments[0].patient_id is not None


# ---------- API integration ----------

def test_api_persists_noshow_score_and_delivery():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    # Re-init so the app picks up the in-memory DB.
    client = TestClient(create_app(database_url="sqlite:///:memory:"))

    # Train
    from bi_forecast.noshow import synthesize_appointments
    df = synthesize_appointments(n=300, seed=1)
    r = client.post(
        "/noshow/train",
        files={"file": ("train.csv", df.to_csv(index=False).encode(), "text/csv")},
        data={"target": "no_show"},
    )
    assert r.status_code == 200

    # Score
    upcoming = synthesize_appointments(n=5, seed=2).drop(columns=["no_show"]).to_dict(orient="records")
    r = client.post("/noshow/score", json={"appointments": upcoming})
    assert r.status_code == 200

    # Predictions appear in history.
    r = client.get("/history/predictions")
    assert r.status_code == 200
    assert r.json()["n"] == 5

    # Send messages (console = dry_run) and check delivery history.
    appts = [{
        "appointment_id": "A1", "name": "Jane", "phone": "+1555",
        "appointment_time": "Mon 10:15", "risk_band": "high",
    }]
    r = client.post("/engagement/send", json={"channel": "console", "appointments": appts})
    assert r.status_code == 200
    r = client.get("/history/deliveries")
    assert r.status_code == 200
    assert r.json()["n"] >= 1
    assert r.json()["items"][0]["status"] == "dry_run"


def test_api_persists_hl7_message():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    client = TestClient(create_app(database_url="sqlite:///:memory:"))
    EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
    hl7 = (EXAMPLES / "hl7" / "siu_s12_sample.hl7").read_text()

    r = client.post("/hooks/hl7", json={"message": hl7})
    assert r.status_code == 200
    r = client.get("/history/hl7")
    assert r.json()["n"] == 1
    assert r.json()["items"][0]["message_type"].startswith("SIU")


def test_api_history_models_after_training(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    client = TestClient(create_app(database_url="sqlite:///:memory:"))
    from bi_forecast.noshow import synthesize_appointments
    df = synthesize_appointments(n=200, seed=4)
    save_path = tmp_path / "m.joblib"
    r = client.post(
        "/noshow/train",
        files={"file": ("t.csv", df.to_csv(index=False).encode(), "text/csv")},
        data={"target": "no_show", "save_to": str(save_path)},
    )
    assert r.status_code == 200
    r = client.get("/history/models")
    assert r.json()["n"] == 1
    assert r.json()["items"][0]["name"] == "noshow"
    assert r.json()["items"][0]["path"] == str(save_path)
