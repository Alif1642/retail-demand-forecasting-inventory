from __future__ import annotations

import pandas as pd


class ReconciliationStrategy:
    """Extensible interface for forecast reconciliation strategies."""

    def reconcile(self, bottom_level: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BottomUpReconciliation(ReconciliationStrategy):
    """Bottom-up placeholder that preserves SKU/store forecasts for aggregation."""

    def reconcile(self, bottom_level: pd.DataFrame) -> pd.DataFrame:
        return bottom_level.copy()
