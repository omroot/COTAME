"""Volume data ingestion from pre-raw CSV files."""

import numpy as np
import pandas as pd
from pathlib import Path

_VOLUME_FILES: dict[str, str] = {
    'CL': 'wti_vol.csv',
    'XB': 'rbob_vol.csv',
    'HO': 'ho_vol.csv',
    'QS': 'gasoil_vol.csv',
    'CO': 'br_vol.csv',
}

_VOLUME_COLUMNS_WITH_SPREAD = [
    'tradeDate',
    'F1_Volume',
    'F2_Volume',
    'F3_Volume',
    'F1MinusF2_Volume',
    'Name',
]

_VOLUME_COLUMNS_WITHOUT_SPREAD = [
    'tradeDate',
    'F1_Volume',
    'F2_Volume',
    'F3_Volume',
    'Name',
]

_VOLUME_OUTPUT_COLUMNS = [
    'tradeDate',
    'Name',
    'F1_Volume',
    'F2_Volume',
    'F3_Volume',
    'F1MinusF2_Volume',
]


def ingest_volume_data(pre_raw_data_directory: Path) -> pd.DataFrame:
    """Ingest volume data from pre-raw CSV files for all tickers.

    Reads {ticker}_vol.csv files, standardizes column names,
    and returns a unified volume database. Handles files with or
    without the F1-F2 spread volume column.

    Parameters
    ----------
    pre_raw_data_directory : Path
        Directory containing the pre-raw CSV files.

    Returns
    -------
    pd.DataFrame
        Unified volume data with columns: tradeDate, Name, F1-F3 volumes,
        and F1-F2 spread volume.
    """
    frames = []
    for ticker, filename in _VOLUME_FILES.items():
        ticker_data = pd.read_csv(pre_raw_data_directory / filename)
        ticker_data['Name'] = ticker
        column_count = len(ticker_data.columns)

        if column_count == len(_VOLUME_COLUMNS_WITH_SPREAD):
            ticker_data.columns = _VOLUME_COLUMNS_WITH_SPREAD
        elif column_count == len(_VOLUME_COLUMNS_WITHOUT_SPREAD):
            ticker_data.columns = _VOLUME_COLUMNS_WITHOUT_SPREAD
            ticker_data['F1MinusF2_Volume'] = np.nan
        else:
            print(f"  WARNING: skipping {filename} — unexpected column count: {column_count}")
            continue

        frames.append(ticker_data)

    if not frames:
        return pd.DataFrame(columns=_VOLUME_OUTPUT_COLUMNS)
    combined_volume = pd.concat(frames, ignore_index=True)
    return combined_volume[_VOLUME_OUTPUT_COLUMNS]
