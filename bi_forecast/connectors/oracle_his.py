"""Read-only Oracle HIS connector.

Architecture
------------
- The engine NEVER writes to the HIS. All queries are SELECT-only.
- Connection details + queries come from env vars + a YAML config file
  so the same code works against MLM / Cerner / Epic / custom HIS by
  swapping the config.
- Default driver is `oracledb` in thin mode — no Oracle Client install
  required. Set THICK_MODE=1 to use thick mode if your HIS needs it
  (e.g. for some encrypted connections).

Env vars (set these on the machine that runs the engine; never commit):
    HIS_ORACLE_DSN          host:port/service_name        (e.g. 192.168.129.30:1521/mhdb)
    HIS_ORACLE_USER         schema user                   (e.g. yasasii)
    HIS_ORACLE_PASSWORD     password
    HIS_CONFIG_PATH         optional path to YAML config; defaults to
                            ./his_config.yaml in CWD

The YAML config defines named queries that map to canonical fields. See
examples/his_config.example.yaml for the template.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml


_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class HISConfig:
    """Per-HIS config: tells the connector which tables/columns to query."""
    appointments_sql: str
    patients_sql: str
    admissions_sql: str
    column_map: Dict[str, Dict[str, str]]  # kind -> {canonical: source_column}

    @classmethod
    def load(cls, path: Optional[str] = None) -> "HISConfig":
        p = Path(path or os.environ.get("HIS_CONFIG_PATH", "his_config.yaml"))
        if not p.exists():
            raise FileNotFoundError(
                f"HIS config not found at {p}. Copy examples/his_config.example.yaml "
                f"to your project root and fill in your table/column names."
            )
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            appointments_sql=data["queries"]["appointments"],
            patients_sql=data["queries"]["patients"],
            admissions_sql=data["queries"]["admissions"],
            column_map=data.get("column_map", {}),
        )


def _validate_select_only(sql: str) -> None:
    """Defensive check — refuse anything that isn't a single SELECT."""
    cleaned = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    stripped = cleaned.strip().rstrip(";").strip().lower()
    if not stripped.startswith("select") and not stripped.startswith("with"):
        raise ValueError("HIS queries must be SELECT-only (read-only engine).")
    forbidden = ("insert ", "update ", "delete ", "drop ", "alter ", "truncate ",
                 "create ", "grant ", "revoke ", "merge ")
    for kw in forbidden:
        if kw in stripped:
            raise ValueError(f"HIS query contains forbidden keyword: {kw.strip().upper()}")


class OracleHIS:
    """Read-only connection to the HIS Oracle database."""

    def __init__(self, dsn: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, config: Optional[HISConfig] = None):
        self.dsn = dsn or os.environ.get("HIS_ORACLE_DSN")
        self.user = user or os.environ.get("HIS_ORACLE_USER")
        self.password = password or os.environ.get("HIS_ORACLE_PASSWORD")
        self.config = config or HISConfig.load()
        self._conn = None

        if not all([self.dsn, self.user, self.password]):
            raise RuntimeError(
                "Missing HIS Oracle credentials. Set HIS_ORACLE_DSN, "
                "HIS_ORACLE_USER, HIS_ORACLE_PASSWORD env vars."
            )

    def _connect(self):
        if self._conn is not None:
            return self._conn
        import oracledb  # imported lazily so tests don't need the driver
        if os.environ.get("THICK_MODE"):
            oracledb.init_oracle_client()
        self._conn = oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------- public read API -------

    def fetch(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Run a parameterized SELECT, return a DataFrame."""
        _validate_select_only(sql)
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(sql, params or {})
            cols = [d[0].lower() for d in cur.description]
            rows = cur.fetchall()
        finally:
            cur.close()
        return pd.DataFrame(rows, columns=cols)

    def fetch_appointments(self, since: Optional[datetime] = None,
                           until: Optional[datetime] = None) -> pd.DataFrame:
        since = since or (datetime.utcnow() - timedelta(days=7))
        until = until or (datetime.utcnow() + timedelta(days=30))
        df = self.fetch(self.config.appointments_sql, {"since": since, "until": until})
        return _rename(df, self.config.column_map.get("appointments", {}))

    def fetch_patients(self, mrns: Optional[Iterable[str]] = None) -> pd.DataFrame:
        df = self.fetch(self.config.patients_sql, {"mrns": list(mrns or [])})
        return _rename(df, self.config.column_map.get("patients", {}))

    def fetch_admissions(self, since: Optional[datetime] = None,
                         until: Optional[datetime] = None) -> pd.DataFrame:
        since = since or (datetime.utcnow() - timedelta(days=30))
        until = until or datetime.utcnow()
        df = self.fetch(self.config.admissions_sql, {"since": since, "until": until})
        return _rename(df, self.config.column_map.get("admissions", {}))

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def _rename(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename source columns to canonical names per the config map."""
    if not mapping:
        return df
    inverse = {src.lower(): canonical for canonical, src in mapping.items() if src}
    rename = {col: inverse[col] for col in df.columns if col in inverse}
    return df.rename(columns=rename)


def pull_to_dataframe(kind: str, since: Optional[datetime] = None,
                      until: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
    """Top-level helper: pull a single kind ('appointments' | 'patients' | 'admissions')."""
    with OracleHIS(**kwargs) as his:
        if kind == "appointments":
            return his.fetch_appointments(since=since, until=until)
        if kind == "patients":
            return his.fetch_patients(mrns=kwargs.get("mrns"))
        if kind == "admissions":
            return his.fetch_admissions(since=since, until=until)
        raise ValueError(f"Unknown kind: {kind}")
