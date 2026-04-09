"""Price data ingestion from pre-raw CSV files."""

import pandas as pd
from pathlib import Path

from src.config.tickers import FUTURES_TICKER_TO_NAME

_PRICE_COT_FILES: dict[str, str] = {
    'CL': 'wti_price_cot.csv',
    'XB': 'rbob_price_cot.csv',
    'HO': 'ho_price_cot.csv',
    'QS': 'gasoil_price_cot.csv',
    'CO': 'br_price_cot.csv',
}

_RAW_COLUMNS = [
    'tradeDate',
    'F1_Price',
    'F2_Price',
    'F3_Price',
    'F1_RolledPrice',
    'F2_RolledPrice',
    'F3_RolledPrice',
    'Commercial_NetPosition',
    'CommercialLongPosition',
    'CommercialShortPosition',
    'ManagedMoney_NetPosition',
    'ManagedMoney_LongPosition',
    'ManagedMoney_ShortPosition',
    'Name',
]

_PRICE_OUTPUT_COLUMNS = [
    'tradeDate',
    'Name',
    'F1_Price',
    'F2_Price',
    'F3_Price',
    'F1_RolledPrice',
    'F2_RolledPrice',
    'F3_RolledPrice',
]


def ingest_price_data(pre_raw_data_directory: Path) -> pd.DataFrame:
    """Ingest price data from pre-raw CSV files for all tickers.

    Reads {ticker}_price_cot.csv files, standardizes column names,
    and returns a unified price database.

    Parameters
    ----------
    pre_raw_data_directory : Path
        Directory containing the pre-raw CSV files.

    Returns
    -------
    pd.DataFrame
        Unified price data with columns: tradeDate, Name, F1-F3 prices
        (raw and rolled).
    """
    expected_column_count = len(_RAW_COLUMNS)
    frames = []
    for ticker, filename in _PRICE_COT_FILES.items():
        ticker_data = pd.read_csv(pre_raw_data_directory / filename)
        ticker_data['Name'] = ticker
        if len(ticker_data.columns) != expected_column_count:
            print(f"  WARNING: skipping {filename} — expected {expected_column_count} "
                  f"columns, got {len(ticker_data.columns)}")
            continue
        ticker_data.columns = _RAW_COLUMNS
        frames.append(ticker_data)

    if not frames:
        return pd.DataFrame(columns=_PRICE_OUTPUT_COLUMNS)
    combined_prices = pd.concat(frames, ignore_index=True)
    return combined_prices[_PRICE_OUTPUT_COLUMNS]
