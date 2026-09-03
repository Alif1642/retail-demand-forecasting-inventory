import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import load_sales


class LoaderTests(unittest.TestCase):
    def test_sample_mode_limits_real_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({
                "id": ["a_evaluation", "b_evaluation"],
                "item_id": ["a", "b"], "dept_id": ["d", "d"], "cat_id": ["c", "c"],
                "store_id": ["s", "s"], "state_id": ["CA", "CA"], "d_1": [1, 2],
            })
            frame.to_csv(root / "sales_train_evaluation.csv", index=False)
            loaded = load_sales(root, data_mode="sample", max_series=1)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(int(loaded["d_1"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
