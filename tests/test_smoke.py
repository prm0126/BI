"""End-to-end smoke tests for the forecasting engine."""

import numpy as np
import pandas as pd
import pytest

from bi_forecast import ModelRouter, route_and_forecast, explain_forecast, recommend_actions
from bi_forecast.whatif import what_if, WhatIfScenario
from bi_forecast.features import extract_features


@pytest.fixture
def daily_admissions():
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    weekday_effect = np.array([1.0, 1.05, 1.1, 1.15, 1.2, 0.8, 0.7])
    weekday = np.array([weekday_effect[d.weekday()] for d in idx])
    y = 50 + np.linspace(0, 10, n) + weekday * 5 + rng.normal(0, 2, n)
    s = pd.Series(y, index=idx, name="admissions")
    return s


def test_extract_features(daily_admissions):
    f = extract_features(daily_admissions)
    assert f.n_obs == len(daily_admissions)
    assert f.seasonal_period == 7
    assert f.mean > 0


def test_router_chooses_a_model(daily_admissions):
    router = ModelRouter()
    decision, _ = router.route(daily_admissions)
    assert decision.chosen in {"seasonal_naive", "ets", "sarima", "gbm"}
    assert decision.chosen in decision.candidates
    assert decision.scores[decision.chosen]["rmse"] >= 0


def test_full_forecast_pipeline(daily_admissions):
    result = route_and_forecast(daily_admissions, horizon=14, target_name="admissions")
    assert len(result.forecast_values) == 14
    assert len(result.lower) == 14
    assert len(result.upper) == 14
    # Bounds sanity.
    for p, lo, hi in zip(result.forecast_values, result.lower, result.upper):
        assert lo <= p <= hi or np.isclose(lo, p) or np.isclose(p, hi)


def test_explain_and_recommend(daily_admissions):
    result = route_and_forecast(daily_admissions, horizon=14, target_name="admissions")
    explanation = explain_forecast(result, series=daily_admissions)
    assert "drivers" in explanation
    recs = recommend_actions(result, explanation)
    assert recs and all(r.severity in {"act", "watch", "info"} for r in recs)
    assert all(r.action for r in recs)


def test_whatif_shifts_forecast(daily_admissions):
    scenarios = [
        WhatIfScenario(name="surge", multiplier=1.20, on_last_n=30),
        WhatIfScenario(name="drop", multiplier=0.80, on_last_n=30),
    ]
    out = what_if(daily_admissions, horizon=7, scenarios=scenarios, target_name="admissions")
    assert "baseline" in out and "surge" in out and "drop" in out
    base_mean = np.mean(out["baseline"].forecast_values)
    surge_mean = np.mean(out["surge"].forecast_values)
    drop_mean = np.mean(out["drop"].forecast_values)
    assert surge_mean > base_mean > drop_mean


def test_router_respects_allow_list(daily_admissions):
    router = ModelRouter(allow=["seasonal_naive", "ets"])
    decision, _ = router.route(daily_admissions)
    assert decision.chosen in {"seasonal_naive", "ets"}


def test_short_series_does_not_crash():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2025-01-01", periods=5, freq="D"))
    result = route_and_forecast(s, horizon=3)
    assert len(result.forecast_values) == 3
