from typing import Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseModel


class SARIMAModel(BaseModel):
    name = "sarima"

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "SARIMAModel":
        s = series.dropna().astype(float)
        period = self.seasonal_period or 1
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, period) if period > 1 and len(s) >= 2 * period else (0, 0, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit = SARIMAX(
                s,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        self._fitted = True
        return self

    def predict(
        self, horizon: int, exog_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = self._fit.get_forecast(steps=horizon, exog=exog_future)
        mean = np.asarray(pred.predicted_mean, dtype=float)
        ci = pred.conf_int(alpha=0.05)
        lower = np.asarray(ci.iloc[:, 0], dtype=float)
        upper = np.asarray(ci.iloc[:, 1], dtype=float)
        return mean, lower, upper
