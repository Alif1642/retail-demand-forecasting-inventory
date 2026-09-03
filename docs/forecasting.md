# Forecasting

## Baseline

The seasonal-naive baseline uses a 7-day seasonal period. For multi-step forecasts it repeats the most recent weekly pattern recursively, so validation demand is never reused as future input.

## LightGBM

The primary model uses lag, rolling, calendar, event, SNAP, price, and stable identifier features. Training and validation are chronological. Random shuffled splitting is not used.

## Leakage prevention

Demand lags use group-wise shifts by SKU/store series. Rolling demand features use `shift(1)` before every rolling calculation. During recursive inference, future demand starts as missing; each predicted day is appended before the next day's features are recomputed.

Price variables are treated as available exogenous inputs. The project reports predictive associations only and does not make causal claims about price effects.
