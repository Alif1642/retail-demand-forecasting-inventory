from __future__ import annotations


def shortage_cost(shortage_units: float, stockout_cost: float) -> float:
    return max(0.0, shortage_units) * max(0.0, stockout_cost)


def excess_holding_cost(excess_units: float, holding_cost: float) -> float:
    return max(0.0, excess_units) * max(0.0, holding_cost)
