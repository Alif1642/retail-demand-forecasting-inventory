import unittest

from api.main import app
from api.routes.health import health


class ApiTests(unittest.TestCase):
    def test_app_exists(self):
        self.assertEqual(app.title, "Retail Demand Forecasting & Inventory Decision Engine")

    def test_health_endpoint_function(self):
        payload = health()
        self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
