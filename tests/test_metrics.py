import math
import unittest

from src.evaluation.bias import forecast_bias
from src.evaluation.metrics import mae
from src.evaluation.rmsse import rmsse
from src.evaluation.wmape import wmape


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        actual = [10, 20]
        forecast = [12, 18]
        self.assertEqual(mae(actual, forecast), 2.0)
        self.assertAlmostEqual(wmape(actual, forecast), 4 / 30)
        self.assertAlmostEqual(forecast_bias(actual, forecast), 0.0)
        self.assertTrue(math.isfinite(rmsse(actual, forecast, [1, 2, 3, 4])))

    def test_zero_denominator(self):
        self.assertEqual(wmape([0, 0], [0, 0]), 0.0)
        self.assertEqual(forecast_bias([0, 0], [0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
