"""Generate synthetic healthcare-operations sample CSVs for demo/testing."""

import numpy as np
import pandas as pd
from pathlib import Path


def make_admissions(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    weekday_effect = np.array([1.0, 1.05, 1.08, 1.10, 1.15, 0.85, 0.75])  # Mon..Sun
    base = 60
    trend = np.linspace(0, 10, n_days)
    seasonal = 8 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    weekday = np.array([weekday_effect[d.weekday()] for d in dates])
    noise = rng.normal(0, 4, n_days)
    y = np.clip((base + trend + seasonal) * weekday + noise, 0, None).round()
    return pd.DataFrame({"ds": dates, "admissions": y})


def make_occupancy(n_days: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    base = 72.0
    trend = np.linspace(0, 6, n_days)
    weekly = 4 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 2.5, n_days)
    y = np.clip(base + trend + weekly + noise, 20, 99).round(1)
    return pd.DataFrame({"ds": dates, "occupancy": y})


def make_wait_time(n_days: int = 120, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    base = 28.0
    surge = np.where(np.arange(n_days) > 80, 12, 0)  # surge starts day 80
    weekly = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 3, n_days)
    y = np.clip(base + surge + weekly + noise, 5, None).round(1)
    return pd.DataFrame({"ds": dates, "wait_time": y})


if __name__ == "__main__":
    here = Path(__file__).parent
    make_admissions().to_csv(here / "admissions.csv", index=False)
    make_occupancy().to_csv(here / "occupancy.csv", index=False)
    make_wait_time().to_csv(here / "wait_time.csv", index=False)
    print("Wrote:", list(here.glob("*.csv")))
