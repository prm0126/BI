"""Tests for the Oracle HIS connector.

We can't run a real Oracle in CI, so we exercise:
  - SELECT-only validation (defensive guardrail)
  - Config loading from YAML
  - Column-map renaming
  - End-to-end fetch via an injected sqlite3 connection
"""

import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bi_forecast.connectors.oracle_his import (
    HISConfig, OracleHIS, _validate_select_only, _rename,
)


# ---------- guardrail ----------

def test_validate_rejects_writes():
    for bad in ["DELETE FROM patients", "UPDATE x SET y=1", "DROP TABLE x",
                "INSERT INTO y VALUES (1)", "MERGE INTO x USING y"]:
        with pytest.raises(ValueError):
            _validate_select_only(bad)


def test_validate_allows_select_and_cte():
    _validate_select_only("SELECT 1 FROM dual")
    _validate_select_only("WITH x AS (SELECT 1 FROM dual) SELECT * FROM x")
    _validate_select_only("-- a comment\nSELECT 1 FROM dual")


def test_rename_applies_column_map():
    df = pd.DataFrame({"appt_id": [1], "mrn": ["X"]})
    out = _rename(df, {"appointment_id": "APPT_ID", "patient_id": "MRN"})
    assert list(out.columns) == ["appointment_id", "patient_id"]


# ---------- config loading ----------

def test_load_yaml_config(tmp_path: Path):
    yml = tmp_path / "his.yaml"
    yml.write_text(textwrap.dedent("""
        queries:
          appointments: "SELECT 1 AS appointment_id FROM dual"
          patients:     "SELECT 1 AS patient_id FROM dual"
          admissions:   "SELECT 1 AS external_id FROM dual"
        column_map:
          appointments: {appointment_id: APPOINTMENT_ID}
    """).strip())
    cfg = HISConfig.load(str(yml))
    assert "appointment_id" in cfg.appointments_sql
    assert cfg.column_map["appointments"]["appointment_id"] == "APPOINTMENT_ID"


def test_load_yaml_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HISConfig.load(str(tmp_path / "no_such.yaml"))


# ---------- end-to-end fetch via sqlite ----------

@pytest.fixture
def sqlite_his(tmp_path: Path, monkeypatch) -> OracleHIS:
    """Build a real OracleHIS but swap the Oracle connection for SQLite."""
    db_path = tmp_path / "fake_his.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE HIS_PATIENTS (MRN TEXT PRIMARY KEY, FULL_NAME TEXT, MOBILE TEXT);
        CREATE TABLE HIS_APPOINTMENTS (
            APPT_ID TEXT PRIMARY KEY, MRN TEXT, APPT_DATETIME TEXT,
            DOCTOR_NAME TEXT, DEPARTMENT TEXT, LOCATION TEXT, STATUS TEXT
        );
        INSERT INTO HIS_PATIENTS VALUES ('MRN001', 'Jane Doe', '+1555');
        INSERT INTO HIS_APPOINTMENTS VALUES
            ('APT-1', 'MRN001', '2026-05-25 10:15', 'Dr. Smith', 'Cardio', 'OPD-3', 'SCHEDULED'),
            ('APT-2', 'MRN001', '2026-05-26 11:00', 'Dr. Khan',  'Derm',   'OPD-1', 'SCHEDULED');
    """)
    conn.commit()

    # SQLite uses :name binding too, so the same SQL works (with date as text).
    cfg = HISConfig(
        appointments_sql=textwrap.dedent("""
            SELECT a.APPT_ID AS appointment_id, a.MRN AS patient_id,
                   p.FULL_NAME AS name, p.MOBILE AS phone,
                   a.APPT_DATETIME AS appointment_time,
                   a.DOCTOR_NAME AS doctor, a.DEPARTMENT AS specialty,
                   a.LOCATION AS location, a.STATUS AS status
            FROM HIS_APPOINTMENTS a
            LEFT JOIN HIS_PATIENTS p ON a.MRN = p.MRN
            WHERE a.APPT_DATETIME BETWEEN :since AND :until
        """).strip(),
        patients_sql="SELECT MRN AS patient_id, FULL_NAME AS name FROM HIS_PATIENTS",
        admissions_sql="SELECT 1 AS external_id FROM HIS_PATIENTS LIMIT 0",
        column_map={"appointments": {}, "patients": {}, "admissions": {}},
    )

    monkeypatch.setenv("HIS_ORACLE_DSN", "x")
    monkeypatch.setenv("HIS_ORACLE_USER", "x")
    monkeypatch.setenv("HIS_ORACLE_PASSWORD", "x")
    his = OracleHIS(config=cfg)
    his._conn = conn  # inject the sqlite connection in place of oracledb
    return his


def test_fetch_appointments_via_sqlite(sqlite_his):
    df = sqlite_his.fetch(
        sqlite_his.config.appointments_sql,
        {"since": "2026-05-01", "until": "2026-12-31"},
    )
    assert len(df) == 2
    assert set(df["appointment_id"]) == {"APT-1", "APT-2"}
    assert df.iloc[0]["name"] == "Jane Doe"


def test_fetch_blocks_writes_at_runtime(sqlite_his):
    with pytest.raises(ValueError):
        sqlite_his.fetch("DELETE FROM HIS_PATIENTS WHERE MRN='MRN001'")


def test_missing_credentials_raises(monkeypatch, tmp_path):
    yml = tmp_path / "his.yaml"
    yml.write_text(
        "queries:\n  appointments: SELECT 1 FROM dual\n"
        "  patients: SELECT 1 FROM dual\n  admissions: SELECT 1 FROM dual\n"
    )
    monkeypatch.delenv("HIS_ORACLE_DSN", raising=False)
    monkeypatch.delenv("HIS_ORACLE_USER", raising=False)
    monkeypatch.delenv("HIS_ORACLE_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="Missing HIS Oracle credentials"):
        OracleHIS(config=HISConfig.load(str(yml)))
