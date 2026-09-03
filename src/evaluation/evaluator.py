from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.bias import forecast_bias
from src.evaluation.metrics import mae
from src.evaluation.rmsse import rmsse
from src.evaluation.wmape import wmape


def evaluate_forecasts(
    forecast_frame: pd.DataFrame,
    history: pd.DataFrame | None = None,
    actual_col: str = "demand",
    forecast_col: str = "forecast",
) -> dict[str, float]:
    actual = forecast_frame[actual_col].to_numpy(dtype=float)
    predicted = forecast_frame[forecast_col].to_numpy(dtype=float)
    metrics = {
        "MAE": mae(actual, predicted),
        "WMAPE": wmape(actual, predicted),
        "Bias": forecast_bias(actual, predicted),
    }
    rmsse_values = []
    if history is not None and "series_id" in forecast_frame.columns:
        for series_id, group in forecast_frame.groupby("series_id", observed=True):
            train_values = history.loc[history["series_id"] == series_id, "demand"]
            value = rmsse(group[actual_col], group[forecast_col], train_values)
            if np.isfinite(value):
                rmsse_values.append(value)
    metrics["RMSSE"] = float(np.mean(rmsse_values)) if rmsse_values else float("nan")
    return metrics
