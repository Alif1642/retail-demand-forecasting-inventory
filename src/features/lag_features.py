from __future__ import annotations

import pandas as pd

LAGS = (1, 2, 3, 7, 14, 28, 56)


def add_lag_features(frame: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    out = frame.sort_values(["series_id", "date"]).copy()
    grouped = out.groupby("series_id", observed=True)["demand"]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out
