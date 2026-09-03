from _bootstrap import ROOT

from src.config import settings
from src.data.loader import load_m5_data
from src.data.preprocessing import prepare_m5_long
from src.data.validator import validate_m5_data
from src.utils.paths import ensure_directories


def main():
    settings.validate()
    data = load_m5_data(data_mode=settings.data_mode, max_series=settings.max_series)
    validate_m5_data(data["sales"], data["calendar"], data["prices"])
    prepared = prepare_m5_long(data["sales"], data["calendar"], data["prices"])
    ensure_directories(settings.processed_dir)
    output = settings.processed_dir / "m5_prepared.csv"
    prepared.to_csv(output, index=False)
    print(f"Saved {len(prepared):,} prepared rows to {output}")
    print(f"Series: {prepared['series_id'].nunique():,}; dates: {prepared['date'].min()} to {prepared['date'].max()}")
    print(f"Missing sell prices: {prepared['sell_price'].isna().sum():,}")


if __name__ == "__main__":
    main()
