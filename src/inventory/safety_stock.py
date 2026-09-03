from __future__ import annotations

import math

Z_BY_SERVICE_LEVEL = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}


def safety_stock(demand_std: float, lead_time_days: int, service_level: float = 0.95) -> float:
    if service_level not in Z_BY_SERVICE_LEVEL:
        raise ValueError(f"Supported service levels: {sorted(Z_BY_SERVICE_LEVEL)}")
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be positive")
    return max(0.0, Z_BY_SERVICE_LEVEL[service_level] * max(0.0, demand_std) * math.sqrt(lead_time_days))
