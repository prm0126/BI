"""Data-loading helpers for healthcare ops CSVs."""

from typing import Optional, Tuple
import pandas as pd


def load_series(
    path: str,
    date_col: str = "ds",
    target_col: str = "y",
    exog_cols: Optional[list] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not in {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    series = df[target_col].astype(float)
    series.name = target_col

    exog = None
    if exog_cols:
        missing = [c for c in exog_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Exogenous columns missing: {missing}")
        exog = df[exog_cols].astype(float)
    return series, exog
