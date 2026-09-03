# Inventory Decision Engine

Forecasts are translated into operating recommendations using expected demand, forecast uncertainty, current inventory, lead time, service level, and cost inputs.

## Safety stock

`Safety Stock = Z × Demand Std × sqrt(Lead Time)`

Supported service levels and Z values:

- 90% → 1.282
- 95% → 1.645
- 99% → 2.326

The implementation derives an approximate daily demand standard deviation from the forecast interval width and then applies the safety-stock formula.

## Reorder point

`Reorder Point = Expected Lead-Time Demand + Safety Stock`

If current inventory is at or below the reorder point, order quantity is the gap to a target stock level equal to forecast-horizon demand plus safety stock.

## Stockout risk

- **HIGH:** current inventory is below expected lead-time demand.
- **MEDIUM:** inventory covers expected lead-time demand but is below the reorder point.
- **LOW:** inventory is at or above the reorder point.

These boundaries correspond directly to expected lead-time coverage and the safety-stock buffer.

## Overstock risk

The engine first calculates excess units above the upper forecast bound for the full horizon. It converts those units to holding-cost exposure and normalizes that exposure by the holding cost of expected horizon demand.

- **LOW:** no units are above the upper forecast bound.
- **MEDIUM:** normalized excess holding-cost exposure is at most one expected-horizon holding-cost equivalent.
- **HIGH:** normalized exposure exceeds one expected-horizon holding-cost equivalent.

## Recommendation

- `REORDER`: inventory is at/below reorder point.
- `MONITOR`: risk is intermediate but an immediate order is not required.
- `HOLD`: inventory is adequately positioned.
- `OVERSTOCK`: overstock risk is high.

The priority score combines stockout risk, overstock risk, and normalized cost exposure. It is a triage score, not a causal estimate of business impact.
