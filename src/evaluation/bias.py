from __future__ import annotations

import numpy as np


def forecast_bias(actual, forecast) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denominator = np.abs(a).sum()
    if denominator == 0:
        return 0.0 if np.abs(f - a).sum() == 0 else float("nan")
    return float((f - a).sum() / denominator)
