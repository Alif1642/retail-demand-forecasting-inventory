# M5 Data Setup

Download the real M5 Forecasting Walmart dataset separately. Do not copy generated or synthetic replacements into the project.

Place these files in `data/raw/m5/`:

- `calendar.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `sample_submission.csv`

The repository ignores the CSV files. Sample mode reads the first configured set of real sales series and filters the sell-price table to those item/store pairs. Full mode loads the complete source tables.
