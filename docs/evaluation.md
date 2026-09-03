# Evaluation

The project evaluates forecasts with MAE, WMAPE, RMSSE, signed forecast bias, prediction-interval coverage, and interval width.

## MAE

`mean(abs(actual - forecast))`

## WMAPE

`sum(abs(actual - forecast)) / sum(abs(actual))`

A zero actual-demand denominator returns zero only when the total absolute error is also zero; otherwise the result is undefined.

## Forecast bias

`sum(forecast - actual) / sum(abs(actual))`

Positive values indicate aggregate over-forecasting; negative values indicate aggregate under-forecasting.

## RMSSE

For one series with validation observations `y_t`, forecasts `f_t`, and training history `x_1 ... x_n`:

`RMSSE = sqrt( mean((y_t - f_t)^2) / mean((x_t - x_(t-1))^2) )`

The denominator is computed only from historical training demand. For multiple SKU/store series, the implementation computes series-level RMSSE where the scale is valid and reports their arithmetic mean.

## Backtesting

Rolling-origin folds preserve time order. Each fold trains only on dates before its validation origin and forecasts the next configured horizon recursively. Baseline and LightGBM results are reported side by side.

## Prediction intervals

Intervals use empirical signed residual quantiles from validation history. Coverage and average interval width are reported so uncertainty quality can be checked empirically. The method is intentionally presented as residual-based uncertainty estimation rather than a guaranteed probabilistic calibration method.
