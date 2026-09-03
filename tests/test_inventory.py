import unittest

from src.inventory.decision_engine import inventory_decision
from src.inventory.reorder import reorder_point
from src.inventory.safety_stock import safety_stock


class InventoryTests(unittest.TestCase):
    def test_safety_stock_and_reorder(self):
        safety = safety_stock(2.0, 7, 0.95)
        self.assertGreater(safety, 0)
        self.assertAlmostEqual(reorder_point(70, safety), 70 + safety)

    def test_reorder_recommendation(self):
        result = inventory_decision(
            forecast=[10.0] * 28,
            lower_bound=[8.0] * 28,
            upper_bound=[12.0] * 28,
            current_inventory=20,
        )
        self.assertEqual(result["recommendation"], "REORDER")
        self.assertEqual(result["stockout_risk"], "HIGH")


if __name__ == "__main__":
    unittest.main()
