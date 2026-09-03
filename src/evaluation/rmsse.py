from __future__ import annotations

import numpy as np


def rmsse(actual, forecast, training_series) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    train = np.asarray(training_series, dtype=float)
    if a.size == 0 or train.size < 2:
        return float("nan")
    scale = np.mean(np.diff(train) ** 2)
    if scale == 0:
        return 0.0 if np.mean((a - f) ** 2) == 0 else float("nan")
    return float(np.sqrt(np.mean((a - f) ** 2) / scale))
