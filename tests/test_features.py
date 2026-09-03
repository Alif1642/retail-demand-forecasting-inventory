import unittest

import numpy as np
import pandas as pd

from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "series_id": ["A"] * 60,
            "date": pd.date_range("2025-01-01", periods=60),
            "demand": np.arange(60, dtype=float),
        })

    def test_lag_7(self):
        out = add_lag_features(self.frame)
        self.assertEqual(out.loc[10, "lag_7"], 3.0)

    def test_rolling_is_leakage_safe(self):
        out = add_rolling_features(self.frame)
        self.assertEqual(out.loc[7, "rolling_mean_7"], 3.0)
        mutated = self.frame.copy()
        mutated.loc[7, "demand"] = 9999
        out2 = add_rolling_features(mutated)
        self.assertEqual(out2.loc[7, "rolling_mean_7"], 3.0)


if __name__ == "__main__":
    unittest.main()
