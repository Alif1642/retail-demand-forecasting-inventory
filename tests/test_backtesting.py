import unittest

import pandas as pd

from src.forecasting.backtesting import rolling_origins


class BacktestingTests(unittest.TestCase):
    def test_rolling_origins_are_chronological(self):
        frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=150)})
        folds = rolling_origins(frame, horizon=28, n_folds=3)
        self.assertEqual(len(folds), 3)
        self.assertLess(folds[0][1], folds[1][0])
        self.assertLess(folds[1][1], folds[2][0])


if __name__ == "__main__":
    unittest.main()
