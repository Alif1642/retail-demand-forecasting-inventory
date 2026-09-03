from __future__ import annotations

import math

from src.inventory.costs import excess_holding_cost, shortage_cost
from src.inventory.reorder import recommended_order_quantity, reorder_point
from src.inventory.risk import overstock_risk, stockout_risk
from src.inventory.safety_stock import safety_stock

RISK_WEIGHT = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}


def inventory_decision(
    forecast: list[float],
    lower_bound: list[float],
    upper_bound: list[float],
    current_inventory: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    holding_cost: float = 1.0,
    stockout_cost: float = 5.0,
) -> dict:
    if not forecast:
        raise ValueError("forecast cannot be empty")
    lead = min(lead_time_days, len(forecast))
    expected_horizon = float(sum(forecast))
    upper_horizon = float(sum(upper_bound)) if upper_bound else expected_horizon
    expected_lead = float(sum(forecast[:lead]))
    daily_uncertainty = [max(0.0, hi - lo) / 3.29 for lo, hi in zip(lower_bound[:lead], upper_bound[:lead])]
    demand_std = math.sqrt(sum(value * value for value in daily_uncertainty)) / math.sqrt(max(lead, 1)) if daily_uncertainty else 0.0
    safety = safety_stock(demand_std, lead, service_level)
    reorder = reorder_point(expected_lead, safety)
    target_stock = expected_horizon + safety
    order_qty = recommended_order_quantity(current_inventory, reorder, target_stock)
    stock_risk = stockout_risk(current_inventory, expected_lead, safety)
    over_risk = overstock_risk(current_inventory, expected_horizon, upper_horizon, holding_cost)

    shortage_units = max(0.0, reorder - current_inventory)
    excess_units = max(0.0, current_inventory - upper_horizon)
    shortage_exposure = shortage_cost(shortage_units, stockout_cost)
    overstock_exposure = excess_holding_cost(excess_units, holding_cost)
    cost_scale = max(1.0, shortage_exposure + overstock_exposure + expected_horizon)
    priority = 100.0 * min(1.0, (
        0.55 * RISK_WEIGHT[stock_risk]
        + 0.25 * RISK_WEIGHT[over_risk]
        + 0.20 * min(1.0, (shortage_exposure + overstock_exposure) / cost_scale)
    ))

    if over_risk == "HIGH":
        recommendation = "OVERSTOCK"
    elif current_inventory <= reorder:
        recommendation = "REORDER"
    elif stock_risk == "MEDIUM" or over_risk == "MEDIUM":
        recommendation = "MONITOR"
    else:
        recommendation = "HOLD"

    return {
        "expected_demand": round(expected_horizon, 3),
        "expected_lead_time_demand": round(expected_lead, 3),
        "safety_stock": round(safety, 3),
        "reorder_point": round(reorder, 3),
        "recommended_order_quantity": round(order_qty, 3),
        "stockout_risk": stock_risk,
        "overstock_risk": over_risk,
        "priority_score": round(priority, 2),
        "recommendation": recommendation,
    }
