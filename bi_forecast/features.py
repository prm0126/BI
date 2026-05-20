import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from .types import SeriesFeatures


def infer_seasonal_period(index: pd.DatetimeIndex) -> int:
    if index is None or len(index) < 4:
        return 1
    freq = pd.infer_freq(index)
    if freq is None:
        delta = (index[-1] - index[0]) / max(len(index) - 1, 1)
        days = delta.total_seconds() / 86400
        if days < 0.5:
            return 24
        if days < 2:
            return 7
        if days < 10:
            return 4
        return 12
    code = freq.upper()
    if code.startswith("H"):
        return 24
    if code.startswith("D"):
        return 7
    if code.startswith("W"):
        return 52
    if code.startswith("M"):
        return 12
    if code.startswith("Q"):
        return 4
    return 1


def extract_features(series: pd.Series) -> SeriesFeatures:
    s = series.dropna()
    n = len(s)
    if n < 4:
        return SeriesFeatures(
            n_obs=n,
            has_trend=False,
            has_seasonality=False,
            seasonal_period=None,
            stationarity_p=1.0,
            mean=float(s.mean()) if n else 0.0,
            std=float(s.std()) if n else 0.0,
            cv=0.0,
            missing_ratio=float(series.isna().mean()),
            zero_inflation=float((s == 0).mean()) if n else 0.0,
        )

    mean = float(s.mean())
    std = float(s.std())
    cv = std / mean if mean else 0.0

    try:
        adf = adfuller(s, autolag="AIC")
        p_value = float(adf[1])
    except Exception:
        p_value = 1.0

    x = np.arange(n)
    slope = np.polyfit(x, s.values, 1)[0]
    has_trend = abs(slope) > (std / max(n, 1)) * 0.5

    period = infer_seasonal_period(s.index if isinstance(s.index, pd.DatetimeIndex) else None)
    has_seasonality = False
    if period > 1 and n >= 2 * period:
        autocorr = s.autocorr(lag=period)
        if autocorr is not None and not np.isnan(autocorr):
            has_seasonality = autocorr > 0.3

    return SeriesFeatures(
        n_obs=n,
        has_trend=bool(has_trend),
        has_seasonality=bool(has_seasonality),
        seasonal_period=period if period > 1 else None,
        stationarity_p=p_value,
        mean=mean,
        std=std,
        cv=cv,
        missing_ratio=float(series.isna().mean()),
        zero_inflation=float((s == 0).mean()),
    )
