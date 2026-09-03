from __future__ import annotations


def reorder_point(expected_lead_time_demand: float, safety_stock_units: float) -> float:
    return max(0.0, expected_lead_time_demand + safety_stock_units)


def recommended_order_quantity(current_inventory: float, reorder_point_units: float, target_stock: float) -> float:
    if current_inventory > reorder_point_units:
        return 0.0
    return max(0.0, target_stock - current_inventory)
