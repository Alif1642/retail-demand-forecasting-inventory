from __future__ import annotations

import numpy as np
import pandas as pd


def residual_quantiles(residuals, interval: float = 0.90) -> tuple[float, float]:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    alpha = 1.0 - interval
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def add_prediction_intervals(
    frame: pd.DataFrame,
    residuals,
    interval: float = 0.90,
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    lower_q, upper_q = residual_quantiles(residuals, interval)
    out = frame.copy()
    out["lower_bound"] = np.maximum(0.0, out[forecast_col] + lower_q)
    out["upper_bound"] = np.maximum(out["lower_bound"], out[forecast_col] + upper_q)
    return out


def interval_coverage(actual, lower, upper) -> float:
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean((a >= lo) & (a <= hi)))


def average_interval_width(lower, upper) -> float:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean(hi - lo)) if lo.size else float("nan")
