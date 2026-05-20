from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .base import BaseModel


def _build_features(series: pd.Series, period: int, lags: int = 7) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values}, index=series.index)
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    if period and period > 1:
        df[f"lag_{period}"] = df["y"].shift(period)
        df[f"lag_{period * 2}"] = df["y"].shift(period * 2)
    df["roll_mean_7"] = df["y"].shift(1).rolling(min(7, len(df))).mean()
    if isinstance(series.index, pd.DatetimeIndex):
        df["dow"] = series.index.dayofweek
        df["month"] = series.index.month
    return df


class GBMModel(BaseModel):
    name = "gbm"

    def __init__(self, seasonal_period: Optional[int] = None, lags: int = 7):
        super().__init__(seasonal_period=seasonal_period)
        self.lags = lags

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "GBMModel":
        s = series.dropna().astype(float)
        self._series = s
        period = self.seasonal_period or 1
        df = _build_features(s, period, self.lags)
        if exog is not None:
            df = df.join(exog, how="left")
        df = df.dropna()
        if len(df) < 10:
            # Not enough data — fall back to a constant predictor.
            self._mean = float(s.mean())
            self._model = None
            self._sigma = float(s.std())
            self._feature_cols = []
            self._fitted = True
            return self
        y = df["y"].values
        X = df.drop(columns=["y"])
        self._feature_cols = list(X.columns)
        self._model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
        ).fit(X, y)
        resid = y - self._model.predict(X)
        self._sigma = float(resid.std())
        self._exog = exog
        self._fitted = True
        return self

    def _step(self, history: pd.Series, exog_row: Optional[pd.Series]) -> float:
        period = self.seasonal_period or 1
        df = _build_features(history, period, self.lags)
        x = df.iloc[-1:].drop(columns=["y"])
        if exog_row is not None:
            for c, v in exog_row.items():
                x[c] = v
        x = x.reindex(columns=self._feature_cols, fill_value=0.0)
        return float(self._model.predict(x)[0])

    def predict(
        self, horizon: int, exog_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            base = np.repeat(self._mean, horizon)
            sigma = self._sigma * np.sqrt(np.arange(1, horizon + 1))
            return base, base - 1.96 * sigma, base + 1.96 * sigma

        history = self._series.copy()
        preds = []
        last_idx = history.index[-1]
        if isinstance(last_idx, pd.Timestamp):
            freq = pd.infer_freq(history.index) or "D"
            future_idx = pd.date_range(last_idx, periods=horizon + 1, freq=freq)[1:]
        else:
            future_idx = range(int(last_idx) + 1, int(last_idx) + 1 + horizon)

        for i, ts in enumerate(future_idx):
            exog_row = exog_future.iloc[i] if exog_future is not None and i < len(exog_future) else None
            yhat = self._step(history, exog_row)
            preds.append(yhat)
            history = pd.concat([history, pd.Series([yhat], index=[ts])])

        preds = np.array(preds, dtype=float)
        sigma = self._sigma * np.sqrt(np.arange(1, horizon + 1))
        return preds, preds - 1.96 * sigma, preds + 1.96 * sigma

    def feature_importance(self) -> dict:
        if self._model is None or not self._feature_cols:
            return {}
        imp = dict(zip(self._feature_cols, self._model.feature_importances_.tolist()))
        return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
