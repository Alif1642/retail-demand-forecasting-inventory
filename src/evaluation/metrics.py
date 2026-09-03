from __future__ import annotations

import numpy as np


def mae(actual, forecast) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - f)))
