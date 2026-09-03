from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    def __init__(self, seasonal_period: int = 7):
        if seasonal_period < 1:
            raise ValueError("seasonal_period must be positive")
        self.seasonal_period = seasonal_period

    def forecast_values(self, history, horizon: int) -> np.ndarray:
        values = list(np.asarray(history, dtype=float))
        if len(values) < self.seasonal_period:
            raise ValueError("History is shorter than the seasonal period.")
        predicted = []
        for _ in range(horizon):
            source_index = len(values) - self.seasonal_period
            value = values[source_index]
            predicted.append(value)
            values.append(value)
        return np.asarray(predicted, dtype=float)

    def forecast_frame(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for series_id, future_group in future.groupby("series_id", observed=True, sort=False):
            hist = history.loc[history["series_id"] == series_id].sort_values("date")
            target = future_group.sort_values("date")
            values = self.forecast_values(hist["demand"].to_numpy(), len(target))
            out = target.copy()
            out["forecast"] = values
            rows.append(out)
        return pd.concat(rows, ignore_index=True) if rows else future.assign(forecast=np.nan)
