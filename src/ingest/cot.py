"""
COT data ingestion — two modes:

1. **Local pre_raw** (default): reads the pre-joined {ticker}_price_cot.csv
   files from the pre_raw_data directory.  Fast, no external dependency.

2. **cot-ingest**: reads the raw CFTC disaggregated + ICE COT reports from
   the cot-ingest repo, filters for flat price contracts, and sums positions
   across all contracts per ticker per report date.
"""

import pandas as pd
from pathlib import Path

from src.config.contracts import get_tickers, get_cftc_codes, get_cftc_multipliers, get_ice_names, has_ice


# ---------------------------------------------------------------------------
# Mode 1 — Local pre_raw_data
# ---------------------------------------------------------------------------

_PRE_RAW_FILES = {
    'CL': 'wti_price_cot.csv',
    'XB': 'rbob_price_cot.csv',
    'HO': 'ho_price_cot.csv',
    'QS': 'gasoil_price_cot.csv',
    'CO': 'br_price_cot.csv',
}

_PRE_RAW_COLUMNS = [
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

_COT_OUTPUT_COLS = [
    'tradeDate',
    'Name',
    'Commercial_NetPosition',
    'CommercialLongPosition',
    'CommercialShortPosition',
    'ManagedMoney_NetPosition',
    'ManagedMoney_LongPosition',
    'ManagedMoney_ShortPosition',
]


def ingest_cot_data(pre_raw_data_directory: Path) -> pd.DataFrame:
    """Ingest COT data from local pre_raw_data price_cot files.

    Parameters
    ----------
    pre_raw_data_directory : Path
        Directory containing {ticker}_price_cot.csv files.

    Returns
    -------
    DataFrame with columns: tradeDate, Name, Commercial and ManagedMoney
    long/short/net positions.
    """
    expected_column_count = len(_PRE_RAW_COLUMNS)
    frames = []
    for ticker, filename in _PRE_RAW_FILES.items():
        ticker_data = pd.read_csv(pre_raw_data_directory / filename)
        ticker_data['Name'] = ticker
        if len(ticker_data.columns) != expected_column_count:
            print(f"  WARNING: skipping {filename} — expected {expected_column_count} "
                  f"columns, got {len(ticker_data.columns)}")
            continue
        ticker_data.columns = _PRE_RAW_COLUMNS
        frames.append(ticker_data)

    if not frames:
        return pd.DataFrame(columns=_COT_OUTPUT_COLS)
    dataset = pd.concat(frames, ignore_index=True)
    cot_db = dataset[_COT_OUTPUT_COLS].copy()
    return cot_db


# ---------------------------------------------------------------------------
# Mode 2 — Raw CFTC + ICE from cot-ingest
# ---------------------------------------------------------------------------

_CFTC_POS_COLS = {
    'prod_merc_positions_long': 'CommercialLongPosition',
    'prod_merc_positions_short': 'CommercialShortPosition',
    'm_money_positions_long_all': 'ManagedMoney_LongPosition',
    'm_money_positions_short_all': 'ManagedMoney_ShortPosition',
    'swap_positions_long_all': 'SwapDealers_LongPosition',
    'swap__positions_short_all': 'SwapDealers_ShortPosition',
    'other_rept_positions_long': 'OtherReportables_LongPosition',
    'other_rept_positions_short': 'OtherReportables_ShortPosition',
    'nonrept_positions_long_all': 'NonReportables_LongPosition',
    'nonrept_positions_short_all': 'NonReportables_ShortPosition',
    'open_interest_all': 'OpenInterest',
}

_ICE_POS_COLS = {
    'Prod_Merc_Positions_Long_All': 'CommercialLongPosition',
    'Prod_Merc_Positions_Short_All': 'CommercialShortPosition',
    'M_Money_Positions_Long_All': 'ManagedMoney_LongPosition',
    'M_Money_Positions_Short_All': 'ManagedMoney_ShortPosition',
    'Swap_Positions_Long_All': 'SwapDealers_LongPosition',
    'Swap__Positions_Short_All': 'SwapDealers_ShortPosition',
    'Other_Rept_Positions_Long_All': 'OtherReportables_LongPosition',
    'Other_Rept_Positions_Short_All': 'OtherReportables_ShortPosition',
    'NonRept_Positions_Long_All': 'NonReportables_LongPosition',
    'NonRept_Positions_Short_All': 'NonReportables_ShortPosition',
    'Open_Interest_All': 'OpenInterest',
}

_OUTPUT_POS_COLS = list(set(_CFTC_POS_COLS.values()))


def _load_cftc_for_ticker(cftc_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter CFTC data for a ticker's flat price codes, apply contract-size
    multipliers, and sum positions per date."""
    codes = get_cftc_codes(ticker)
    if not codes:
        return pd.DataFrame()

    sub = cftc_df[cftc_df['cftc_contract_market_code'].isin(codes)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub['tradeDate'] = pd.to_datetime(sub['report_date_as_yyyy_mm_dd'])

    rename = {}
    for raw_col, out_col in _CFTC_POS_COLS.items():
        sub[raw_col] = pd.to_numeric(sub[raw_col], errors='coerce')
        rename[raw_col] = out_col

    sub = sub.rename(columns=rename)

    # Apply contract-size multipliers (e.g. E-Mini WTI 0.5, Micro WTI 0.1)
    multipliers = get_cftc_multipliers(ticker)
    for code, multiplier in multipliers.items():
        if multiplier != 1.0:
            mask = sub['cftc_contract_market_code'] == code
            sub.loc[mask, _OUTPUT_POS_COLS] *= multiplier

    result = sub.groupby('tradeDate')[_OUTPUT_POS_COLS].sum().reset_index()
    return result


def _load_ice_for_ticker(ice_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter ICE data for a ticker and return positions per date."""
    combined_name, futures_name = get_ice_names(ticker)
    if combined_name is None:
        return pd.DataFrame()

    br_comb = ice_df[ice_df['Market_and_Exchange_Names'] == combined_name].copy()
    br_fut = ice_df[ice_df['Market_and_Exchange_Names'] == futures_name].copy()

    for df in [br_comb, br_fut]:
        df['tradeDate'] = pd.to_datetime(df['As_of_Date_Form_MM/DD/YYYY'])

    if not br_comb.empty:
        combined_start = br_comb['tradeDate'].min()
        br_early = br_fut[br_fut['tradeDate'] < combined_start]
        ice_data = pd.concat([br_early, br_comb], ignore_index=True)
    else:
        ice_data = br_fut

    if ice_data.empty:
        return pd.DataFrame()

    rename = {}
    for raw_col, out_col in _ICE_POS_COLS.items():
        if raw_col in ice_data.columns:
            ice_data[raw_col] = pd.to_numeric(ice_data[raw_col], errors='coerce')
            rename[raw_col] = out_col

    ice_data = ice_data.rename(columns=rename)
    result = ice_data[['tradeDate'] + [c for c in _OUTPUT_POS_COLS if c in ice_data.columns]].copy()
    result = result.groupby('tradeDate')[_OUTPUT_POS_COLS].sum().reset_index()
    return result


def ingest_cot_data_from_cot_ingest(
    cftc_path: Path,
    ice_path: Path = None,
) -> pd.DataFrame:
    """Ingest COT data from raw CFTC disaggregated + ICE reports.

    Parameters
    ----------
    cftc_path : Path
        Path to disaggregated_combined.csv
    ice_path : Path, optional
        Path to ice_cot.csv. Required for Brent and Gasoil.

    Returns
    -------
    DataFrame with all 5 disaggregated categories.
    """
    cftc_df = pd.read_csv(cftc_path, low_memory=False)
    ice_df = pd.read_csv(ice_path, low_memory=False) if ice_path else None

    all_tickers = []

    for ticker in get_tickers():
        cftc_part = _load_cftc_for_ticker(cftc_df, ticker)

        ice_part = pd.DataFrame()
        if has_ice(ticker) and ice_df is not None:
            ice_part = _load_ice_for_ticker(ice_df, ticker)

        if not cftc_part.empty and not ice_part.empty:
            combined = pd.concat([cftc_part, ice_part], ignore_index=True)
            combined = combined.groupby('tradeDate')[_OUTPUT_POS_COLS].sum().reset_index()
        elif not cftc_part.empty:
            combined = cftc_part
        elif not ice_part.empty:
            combined = ice_part
        else:
            continue

        combined['Name'] = ticker
        all_tickers.append(combined)

    if not all_tickers:
        return pd.DataFrame()

    result = pd.concat(all_tickers, ignore_index=True)

    for prefix in ['Commercial', 'ManagedMoney', 'SwapDealers',
                    'OtherReportables', 'NonReportables']:
        long_col = f'{prefix}LongPosition' if prefix == 'Commercial' else f'{prefix}_LongPosition'
        short_col = f'{prefix}ShortPosition' if prefix == 'Commercial' else f'{prefix}_ShortPosition'
        net_col = f'{prefix}_NetPosition'

        if long_col in result.columns and short_col in result.columns:
            result[net_col] = result[long_col] - result[short_col]

    result.sort_values(['Name', 'tradeDate'], inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result
