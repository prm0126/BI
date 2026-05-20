"""Explain a forecast.

Two layers of explanation:
  - quantitative drivers: feature importance from the chosen model
    (gradient boosting), and a permutation-style fallback otherwise.
  - calendar drivers: which days of the week / months drive the forecast.
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd

from .types import ForecastResult


def _permutation_drivers(series: pd.Series, max_features: int = 5) -> Dict[str, float]:
    """Rank simple calendar drivers by variance explained."""
    if not isinstance(series.index, pd.DatetimeIndex) or len(series) < 14:
        return {}
    s = series.dropna().astype(float)
    total_var = float(s.var()) or 1.0
    drivers = {}
    drivers["weekday_effect"] = float(s.groupby(s.index.dayofweek).mean().var() / total_var)
    drivers["month_effect"] = float(s.groupby(s.index.month).mean().var() / total_var)
    drivers["trend_effect"] = float(
        np.polyfit(np.arange(len(s)), s.values, 1)[0] ** 2 / total_var
    )
    drivers["recent_level"] = float(s.iloc[-7:].mean() / max(s.mean(), 1e-9))
    return dict(sorted(drivers.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_features])


def explain_forecast(result: ForecastResult, series: Optional[pd.Series] = None) -> dict:
    drivers = {}
    if result.feature_importance:
        top = list(result.feature_importance.items())[:5]
        drivers["model_features"] = {k: float(v) for k, v in top}

    if series is not None:
        drivers["calendar_drivers"] = _permutation_drivers(series)

    hist = np.asarray(result.history_values, dtype=float)
    fc = np.asarray(result.forecast_values, dtype=float)
    delta = float(fc.mean() - hist[-min(len(hist), len(fc)):].mean())
    pct = (delta / (hist.mean() or 1e-9)) * 100.0

    return {
        "model": result.model,
        "horizon": result.horizon,
        "expected_change_abs": delta,
        "expected_change_pct": pct,
        "interval_width": float(np.mean(np.asarray(result.upper) - np.asarray(result.lower))),
        "drivers": drivers,
        "backtest": {"rmse": result.rmse_cv, "mape_pct": result.mape_cv},
    }
