"""What-if scenario simulator.

Apply multiplicative / additive scenario adjustments to the history and re-run
the router to see how the forecast shifts.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

from .router import ModelRouter
from .types import ForecastResult


@dataclass
class WhatIfScenario:
    name: str
    multiplier: float = 1.0
    delta: float = 0.0
    on_last_n: Optional[int] = None  # apply scenario only to the tail of the series
    exog_overrides: Optional[Dict[str, float]] = None  # column -> value, applied to future exog


def _apply(series: pd.Series, scenario: WhatIfScenario) -> pd.Series:
    s = series.copy().astype(float)
    if scenario.on_last_n and scenario.on_last_n > 0 and scenario.on_last_n < len(s):
        idx = s.index[-scenario.on_last_n:]
        s.loc[idx] = s.loc[idx] * scenario.multiplier + scenario.delta
    else:
        s = s * scenario.multiplier + scenario.delta
    return s


def what_if(
    series: pd.Series,
    horizon: int,
    scenarios: list,
    target_name: str = "y",
    exog: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
) -> Dict[str, ForecastResult]:
    """Run baseline + one forecast per scenario. Returns dict keyed by scenario name."""
    out: Dict[str, ForecastResult] = {}
    router = ModelRouter()
    baseline, _, _ = router.forecast(
        series, horizon, target_name=target_name, exog=exog, exog_future=exog_future
    )
    out["baseline"] = baseline

    for sc in scenarios:
        scenario_series = _apply(series, sc)
        sc_exog_future = exog_future.copy() if exog_future is not None else None
        if sc.exog_overrides and sc_exog_future is not None:
            for col, val in sc.exog_overrides.items():
                if col in sc_exog_future.columns:
                    sc_exog_future[col] = val
        result, _, _ = router.forecast(
            scenario_series, horizon, target_name=target_name, exog=exog, exog_future=sc_exog_future
        )
        out[sc.name] = result
    return out
