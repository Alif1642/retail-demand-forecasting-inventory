from __future__ import annotations

import pandas as pd

MEAN_WINDOWS = (7, 14, 28, 56)
STD_WINDOWS = (7, 28)


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["series_id", "date"]).copy()
    grouped = out.groupby("series_id", observed=True)["demand"]

    # shift(1) is mandatory: each rolling value is built only from demand observed
    # strictly before the row being predicted, preventing target leakage.
    for window in MEAN_WINDOWS:
        out[f"rolling_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean()
        )
    for window in STD_WINDOWS:
        out[f"rolling_std_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=2).std()
        )
    return out
