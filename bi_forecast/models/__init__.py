from .base import BaseModel
from .naive import SeasonalNaive
from .ets import ETSModel
from .sarima import SARIMAModel
from .gbm import GBMModel

REGISTRY = {
    "seasonal_naive": SeasonalNaive,
    "ets": ETSModel,
    "sarima": SARIMAModel,
    "gbm": GBMModel,
}

__all__ = ["BaseModel", "SeasonalNaive", "ETSModel", "SARIMAModel", "GBMModel", "REGISTRY"]
