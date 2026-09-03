from _bootstrap import ROOT

import pandas as pd

from src.config import settings
from src.forecasting.backtesting import run_backtest, summarize_backtest
from src.utils.helpers import write_json
from src.utils.paths import ensure_directories


def main():
    path = settings.processed_dir / "m5_prepared.csv"
    if not path.exists():
        raise FileNotFoundError("Prepared data is missing. Run scripts/prepare_data.py first.")
    frame = pd.read_csv(path, parse_dates=["date"])
    results, details = run_backtest(frame, settings.forecast_horizon, settings.n_backtest_folds)
    ensure_directories(settings.metrics_dir)
    output = settings.metrics_dir / "backtest_results.csv"
    results.to_csv(output, index=False)
    details.to_csv(settings.metrics_dir / "backtest_predictions.csv", index=False)
    write_json(settings.metrics_dir / "backtest_summary.json", summarize_backtest(results))
    print(results.to_string(index=False))
    print(f"Saved backtest results to {output}")


if __name__ == "__main__":
    main()
