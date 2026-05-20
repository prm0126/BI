from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import numpy as np
import pandas as pd


@dataclass
class SeriesFeatures:
    n_obs: int
    has_trend: bool
    has_seasonality: bool
    seasonal_period: Optional[int]
    stationarity_p: float
    mean: float
    std: float
    cv: float
    missing_ratio: float
    zero_inflation: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForecastResult:
    target: str
    model: str
    horizon: int
    history_index: list
    history_values: list
    forecast_index: list
    forecast_values: list
    lower: list
    upper: list
    rmse_cv: Optional[float] = None
    mape_cv: Optional[float] = None
    feature_importance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ds": self.forecast_index,
                "yhat": self.forecast_values,
                "yhat_lower": self.lower,
                "yhat_upper": self.upper,
            }
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["history_index"] = [str(x) for x in d["history_index"]]
        d["forecast_index"] = [str(x) for x in d["forecast_index"]]
        return json.dumps(d, default=_json_default, indent=2)


@dataclass
class Recommendation:
    headline: str
    severity: str  # "info" | "watch" | "act"
    rationale: str
    action: str

    def to_dict(self) -> dict:
        return asdict(self)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(f"Not serializable: {type(o)}")
