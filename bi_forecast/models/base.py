from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    def __init__(self, seasonal_period: Optional[int] = None):
        self.seasonal_period = seasonal_period
        self._fitted = False

    @abstractmethod
    def fit(self, series: pd.Series, exog: Optional[pd.DataFrame] = None) -> "BaseModel":
        ...

    @abstractmethod
    def predict(
        self, horizon: int, exog_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point, lower, upper)."""

    def feature_importance(self) -> dict:
        return {}
