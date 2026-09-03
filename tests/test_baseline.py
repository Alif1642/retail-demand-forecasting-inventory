import unittest

from src.forecasting.seasonal_naive import SeasonalNaive


class BaselineTests(unittest.TestCase):
    def test_recursive_weekly_pattern(self):
        model = SeasonalNaive(7)
        forecast = model.forecast_values([1, 2, 3, 4, 5, 6, 7], 10)
        self.assertEqual(forecast.tolist(), [1, 2, 3, 4, 5, 6, 7, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
