from __future__ import annotations

import numpy as np


def wmape(actual, forecast) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denominator = np.abs(a).sum()
    if denominator == 0:
        return 0.0 if np.abs(a - f).sum() == 0 else float("nan")
    return float(np.abs(a - f).sum() / denominator)
