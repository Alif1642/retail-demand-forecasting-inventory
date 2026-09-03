from __future__ import annotations


def stockout_risk(current_inventory: float, expected_lead_demand: float, safety_stock_units: float) -> str:
    reorder_point = expected_lead_demand + safety_stock_units
    if current_inventory < expected_lead_demand:
        return "HIGH"
    if current_inventory < reorder_point:
        return "MEDIUM"
    return "LOW"


def overstock_risk(
    current_inventory: float,
    expected_horizon_demand: float,
    upper_horizon_demand: float,
    holding_cost: float,
) -> str:
    expected_horizon_demand = max(0.0, expected_horizon_demand)
    upper_horizon_demand = max(expected_horizon_demand, upper_horizon_demand)
    excess_units = max(0.0, current_inventory - upper_horizon_demand)
    cost_exposure = excess_units * max(0.0, holding_cost)
    baseline_holding_cost = max(1.0, expected_horizon_demand * max(0.0, holding_cost))
    normalized_exposure = cost_exposure / baseline_holding_cost
    if excess_units <= 0:
        return "LOW"
    if normalized_exposure <= 1.0:
        return "MEDIUM"
    return "HIGH"
