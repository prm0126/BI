"""Tests for Phase-2: engagement layer + no-show persistence."""

import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from bi_forecast.engagement import (
    Dispatcher, AppointmentContext, ConsoleChannel,
    DEFAULT_TEMPLATES, render, pick_template,
)
from bi_forecast.noshow import NoShowClassifier, synthesize_appointments


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ---------- Templates ----------

def test_pick_template_returns_channel_specific():
    tpl = pick_template("high", channel="whatsapp")
    assert tpl is not None and tpl.channel == "whatsapp"


def test_render_substitutes_known_keys_and_safely_handles_missing():
    tpl = DEFAULT_TEMPLATES["noshow_high_24h_sms"]
    body = render(tpl, {"name": "Jane", "appointment_time": "Mon 10:15", "clinic": "Verdan"})
    assert "Jane" in body and "Mon 10:15" in body and "Verdan" in body
    # Missing keys collapse to '?' rather than crashing.
    assert "?" in body


# ---------- Dispatcher ----------

def test_dispatcher_preview_does_not_send():
    appts = [
        AppointmentContext("A1", "Jane", "+1555", "Mon 10:15", risk_band="high"),
        AppointmentContext("A2", "Sam", "+1556", "Mon 11:00", risk_band="low"),
    ]
    d = Dispatcher(default_channel="console")
    previews = d.preview(appts)
    assert len(previews) == 2
    assert previews[0].delivery.status == "dry_run"
    # Preview does not append to the live log.
    assert d.log == []


def test_dispatcher_send_uses_console_in_tests(capsys):
    appts = [AppointmentContext("A1", "Jane", "+1555", "Mon 10:15", risk_band="high")]
    d = Dispatcher(default_channel="console")
    results = d.send_batch(appts, channel="console")
    assert len(results) == 1
    assert results[0].delivery.status == "dry_run"
    out = capsys.readouterr().out
    assert "[console]" in out and "Jane" in out
    assert len(d.log) == 1


def test_dispatcher_skips_low_risk():
    appts = [
        AppointmentContext("A1", "Jane", "+1555", "Mon 10:15", risk_band="high"),
        AppointmentContext("A2", "Low", "+1556", "Mon 11:00", risk_band="low"),
    ]
    d = Dispatcher(default_channel="console")
    results = d.send_batch(appts, channel="console", skip_low_risk=True)
    ids = [r.appointment_id for r in results]
    assert ids == ["A1"]


def test_twilio_channels_dry_run_without_creds(monkeypatch, capsys):
    """When TWILIO_* env vars are missing, channels fall back to dry_run instead of failing."""
    from bi_forecast.engagement.channels import TwilioSMSChannel, TwilioWhatsAppChannel
    for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_SMS", "TWILIO_FROM_WHATSAPP"]:
        monkeypatch.delenv(k, raising=False)
    sms = TwilioSMSChannel().send("+15551234567", "hi")
    wa = TwilioWhatsAppChannel().send("+15551234567", "hi")
    assert sms.status == "dry_run" and wa.status == "dry_run"


# ---------- No-show persistence ----------

def test_noshow_save_and_load_roundtrip(tmp_path):
    df = synthesize_appointments(n=400, seed=42)
    clf = NoShowClassifier()
    clf.fit(df, target="no_show")
    upcoming = synthesize_appointments(n=20, seed=43).drop(columns=["no_show"])
    before = [s.probability for s in clf.predict(upcoming)]

    path = tmp_path / "noshow_v1.joblib"
    clf.save(str(path))
    assert path.exists()

    loaded = NoShowClassifier.load(str(path))
    after = [s.probability for s in loaded.predict(upcoming)]
    assert before == pytest.approx(after, rel=1e-9)


# ---------- API: upload, load, engagement ----------

def test_api_train_from_upload_and_load(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    client = TestClient(create_app())

    df = synthesize_appointments(n=300, seed=7)
    csv_bytes = df.to_csv(index=False).encode()
    save_path = tmp_path / "model.joblib"

    r = client.post(
        "/noshow/train",
        files={"file": ("train.csv", csv_bytes, "text/csv")},
        data={"target": "no_show", "save_to": str(save_path)},
    )
    assert r.status_code == 200, r.text
    assert save_path.exists()
    assert r.json()["saved_to"] == str(save_path)

    r2 = client.get("/noshow/status")
    assert r2.json()["loaded"] is True

    r3 = client.post("/noshow/load", data={"path": str(save_path)})
    assert r3.status_code == 200


def test_api_engagement_preview_and_send():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from bi_forecast.api.app import create_app

    client = TestClient(create_app())
    payload = {
        "channel": "console",
        "appointments": [
            {"appointment_id": "A1", "name": "Jane", "phone": "+1555",
             "appointment_time": "Mon 10:15", "doctor": "Dr. Smith",
             "location": "OPD-3", "clinic": "Verdan", "risk_band": "high"},
            {"appointment_id": "A2", "name": "Low", "phone": "+1556",
             "appointment_time": "Mon 11:00", "risk_band": "low"},
        ],
    }
    r = client.post("/engagement/preview", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 2
    assert all("rendered_body" in row for row in body["results"])
    assert any("Jane" in row["rendered_body"] for row in body["results"])

    r2 = client.post("/engagement/send", json=payload)
    assert r2.status_code == 200
    assert r2.json()["n"] == 2
    assert all(row["status"] == "dry_run" for row in r2.json()["results"])

    r3 = client.get("/engagement/log")
    assert r3.status_code == 200
    assert r3.json()["n"] >= 2
