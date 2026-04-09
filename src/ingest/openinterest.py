"""Open interest data ingestion from pre-raw CSV files."""

import pandas as pd
from pathlib import Path

_OPEN_INTEREST_FILES: dict[str, str] = {
    'CL': 'wti_oi.csv',
    'XB': 'rbob_oi.csv',
    'HO': 'ho_oi.csv',
    'QS': 'gasoil_oi.csv',
    'CO': 'br_oi.csv',
}

_RAW_COLUMNS = [
    'tradeDate',
    'F1_OI',
    'F1_AGG',
    'F2_OI',
    'F2_AGG',
    'F3_OI',
    'AGG_OI',
    'Name',
]

_OUTPUT_COLUMNS = [
    'tradeDate',
    'Name',
    'F1_OI',
    'F2_OI',
    'F3_OI',
    'AGG_OI',
]


def ingest_openinterest_data(pre_raw_data_directory: Path) -> pd.DataFrame:
    """Ingest open interest data from pre-raw CSV files for all tickers.

    Reads {ticker}_oi.csv files, standardizes column names,
    drops the header row artifact, and returns a unified OI database.

    Parameters
    ----------
    pre_raw_data_directory : Path
        Directory containing the pre-raw CSV files.

    Returns
    -------
    pd.DataFrame
        Unified open interest data with columns: tradeDate, Name,
        F1-F3 OI, and aggregate OI.
    """
    expected_column_count = len(_RAW_COLUMNS)
    frames = []
    for ticker, filename in _OPEN_INTEREST_FILES.items():
        ticker_data = pd.read_csv(pre_raw_data_directory / filename)
        ticker_data['Name'] = ticker
        if len(ticker_data.columns) != expected_column_count:
            print(f"  WARNING: skipping {filename} — expected {expected_column_count} "
                  f"columns, got {len(ticker_data.columns)}")
            continue
        ticker_data.columns = _RAW_COLUMNS
        frames.append(ticker_data)

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    combined_open_interest = pd.concat(frames, ignore_index=True)
    # Drop the first row which contains a header artifact from the raw CSVs
    combined_open_interest = combined_open_interest.iloc[1:].reset_index(drop=True)
    return combined_open_interest[_OUTPUT_COLUMNS]
