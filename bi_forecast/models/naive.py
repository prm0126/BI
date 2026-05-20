from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseModel


class SeasonalNaive(BaseModel):
    name = "seasonal_naive"

    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "SeasonalNaive":
        self._series = series.dropna().astype(float)
        self._residual_std = float(np.diff(self._series.values).std()) if len(self._series) > 1 else 0.0
        self._fitted = True
        return self

    def predict(
        self, horizon: int, exog_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        period = self.seasonal_period or 1
        values = self._series.values
        if period >= len(values):
            base = np.repeat(values[-1], horizon)
        else:
            template = values[-period:]
            reps = int(np.ceil(horizon / period))
            base = np.tile(template, reps)[:horizon]
        sigma = self._residual_std * np.sqrt(np.arange(1, horizon + 1))
        return base, base - 1.96 * sigma, base + 1.96 * sigma
