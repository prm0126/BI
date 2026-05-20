from typing import Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import BaseModel


class ETSModel(BaseModel):
    name = "ets"

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "ETSModel":
        s = series.dropna().astype(float)
        period = self.seasonal_period or 1
        seasonal = "add" if period > 1 and len(s) >= 2 * period else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = ExponentialSmoothing(
                s,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=period if seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
        resid = self._fit.resid.dropna()
        self._sigma = float(resid.std()) if len(resid) > 1 else 0.0
        self._fitted = True
        return self

    def predict(
        self, horizon: int, exog_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.asarray(self._fit.forecast(horizon), dtype=float)
        sigma = self._sigma * np.sqrt(np.arange(1, horizon + 1))
        return f, f - 1.96 * sigma, f + 1.96 * sigma
