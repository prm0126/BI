"""Model router: picks the best base forecaster per input series.

Strategy
--------
1. Extract series features (length, trend, seasonality, stationarity, variance).
2. Use feature-based heuristics to assemble a *candidate set* of models —
   short series get simpler models, seasonal series get SARIMA/ETS, etc.
3. Backtest each candidate on a held-out tail of the series.
4. Pick the candidate with the lowest RMSE.
5. Refit on the full series and produce the final forecast.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math
import warnings
import numpy as np
import pandas as pd

from .features import extract_features
from .types import ForecastResult, SeriesFeatures
from .models import REGISTRY, BaseModel


@dataclass
class RouteDecision:
    chosen: str
    candidates: List[str]
    scores: dict  # model name -> {"rmse": .., "mape": ..}
    reason: str


def _candidate_models(features: SeriesFeatures) -> List[str]:
    n = features.n_obs
    if n < 14:
        return ["seasonal_naive", "ets"]
    candidates = ["seasonal_naive", "ets"]
    if features.has_seasonality or (features.seasonal_period and n >= 2 * features.seasonal_period):
        candidates.append("sarima")
    if n >= 30:
        candidates.append("gbm")
    return candidates


def _rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _backtest(model_name: str, series: pd.Series, period: Optional[int], holdout: int) -> Tuple[float, float]:
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m: BaseModel = REGISTRY[model_name](seasonal_period=period)
            m.fit(train)
            pred, _, _ = m.predict(holdout)
        return _rmse(test.values, pred), _mape(test.values, pred)
    except Exception:
        return float("inf"), float("nan")


class ModelRouter:
    def __init__(self, allow: Optional[List[str]] = None):
        self.allow = allow

    def route(self, series: pd.Series) -> Tuple[RouteDecision, SeriesFeatures]:
        feats = extract_features(series)
        candidates = _candidate_models(feats)
        if self.allow:
            candidates = [c for c in candidates if c in self.allow] or list(self.allow)

        holdout = max(3, min(14, int(feats.n_obs * 0.2)))
        scores = {}
        if feats.n_obs - holdout < 4:
            best = candidates[0]
            scores[best] = {"rmse": float("nan"), "mape": float("nan")}
            return RouteDecision(
                chosen=best,
                candidates=candidates,
                scores=scores,
                reason="series too short for backtest; defaulted to first candidate",
            ), feats

        for name in candidates:
            rmse, mape = _backtest(name, series, feats.seasonal_period, holdout)
            scores[name] = {"rmse": rmse, "mape": mape}

        ranked = sorted(scores.items(), key=lambda kv: kv[1]["rmse"])
        best = ranked[0][0]
        reason = (
            f"backtest on last {holdout} obs (n={feats.n_obs}, "
            f"seasonal={feats.has_seasonality}, period={feats.seasonal_period}); "
            f"{best} won with rmse={ranked[0][1]['rmse']:.3f}"
        )
        return RouteDecision(chosen=best, candidates=candidates, scores=scores, reason=reason), feats

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        target_name: str = "y",
        exog: Optional[pd.DataFrame] = None,
        exog_future: Optional[pd.DataFrame] = None,
    ) -> Tuple[ForecastResult, RouteDecision, SeriesFeatures]:
        decision, feats = self.route(series)
        model_cls = REGISTRY[decision.chosen]
        model: BaseModel = model_cls(seasonal_period=feats.seasonal_period)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(series, exog=exog)
            point, lower, upper = model.predict(horizon, exog_future=exog_future)

        last = series.index[-1]
        if isinstance(last, pd.Timestamp):
            freq = pd.infer_freq(series.index) or "D"
            future_idx = pd.date_range(last, periods=horizon + 1, freq=freq)[1:]
        else:
            future_idx = list(range(int(last) + 1, int(last) + 1 + horizon))

        score = decision.scores.get(decision.chosen, {})
        result = ForecastResult(
            target=target_name,
            model=decision.chosen,
            horizon=horizon,
            history_index=list(series.index),
            history_values=[float(v) for v in series.values],
            forecast_index=list(future_idx),
            forecast_values=[float(v) for v in point],
            lower=[float(v) for v in lower],
            upper=[float(v) for v in upper],
            rmse_cv=score.get("rmse"),
            mape_cv=score.get("mape"),
            feature_importance=model.feature_importance(),
            metadata={
                "candidates": decision.candidates,
                "scores": decision.scores,
                "reason": decision.reason,
                "series_features": feats.to_dict(),
            },
        )
        return result, decision, feats


def route_and_forecast(
    series: pd.Series,
    horizon: int,
    target_name: str = "y",
    exog: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    allow: Optional[List[str]] = None,
) -> ForecastResult:
    router = ModelRouter(allow=allow)
    result, _, _ = router.forecast(series, horizon, target_name=target_name, exog=exog, exog_future=exog_future)
    return result
