"""Orchestrates preprocessing: builds feature panels and datasets for a given ticker.

Produces two versions of COT-derived outputs:
- Legacy (from pre_raw): Commercial + ManagedMoney only
- Full (from cot-ingest): all 5 disaggregated categories
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from src.preprocessing.base import FutureTicker
from src.preprocessing.prices import PricePanel
from src.preprocessing.volume import VolumePanel
from src.preprocessing.openinterest import OpenInterestPanel
from src.preprocessing.synthetic_spread import SyntheticSpreadBuilder, HedgeMethod
from src.preprocessing.cot import COTPanel
from src.preprocessing.dataset_builder import WeeklyDataSetBuilder
from src.preprocessing.daily_dataset_builder import DailyDataSetBuilder

from src.utils.io.read import RawDataReader
from src.utils.dates import get_nyse_business_dates
from src.settings import Settings


_LEGACY_COT_COLUMNS = [
    'tradeDate',
    'Name',
    'Commercial_NetPosition',
    'CommercialLongPosition',
    'CommercialShortPosition',
    'ManagedMoney_NetPosition',
    'ManagedMoney_LongPosition',
    'ManagedMoney_ShortPosition',
]

_NUMERIC_PRICE_COLUMNS = ['F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice']
_NUMERIC_VOLUME_COLUMNS = ['F1_Volume', 'F2_Volume', 'F3_Volume', 'F1MinusF2_Volume']
_NUMERIC_OI_COLUMNS = ['F1_OI', 'F2_OI', 'F3_OI', 'AGG_OI']


def preprocess_all(ticker: FutureTicker) -> None:
    """Run the full preprocessing pipeline for a single ticker.

    Builds price, volume, open interest, and COT feature panels,
    computes synthetic spreads, and assembles weekly and daily datasets.

    Produces two sets of COT outputs:
    - Legacy: {ticker}_cot_legacy_panel.csv, {ticker}_legacy_dataset.csv,
              {ticker}_legacy_daily_dataset.csv
    - Full:   {ticker}_cot_panel.csv, {ticker}_dataset.csv,
              {ticker}_daily_dataset.csv

    Parameters
    ----------
    ticker : FutureTicker
        The futures ticker to preprocess.
    """
    raw_data_path = Settings.historical.paths.RAW_DATA_PATH
    preprocessed_data_path = Settings.historical.paths.PREPROCESSED_DATA_PATH
    reader = RawDataReader(raw_data_directory=raw_data_path)

    # --- Prices ---
    all_prices = reader.read_prices()
    ticker_prices = all_prices[all_prices['Name'] == ticker.value].copy()
    if ticker_prices.empty:
        print(f"    WARNING: no price data for {ticker.name} ({ticker.value}), skipping.")
        return
    ticker_prices['tradeDate'].dropna(inplace=True)
    ticker_prices.loc[:, 'tradeDate'] = pd.to_datetime(ticker_prices['tradeDate']).dt.date
    for column in _NUMERIC_PRICE_COLUMNS:
        ticker_prices.loc[:, column] = pd.to_numeric(ticker_prices[column], errors='coerce')

    business_dates = get_nyse_business_dates(
        ticker_prices['tradeDate'].min(),
        ticker_prices['tradeDate'].max(),
    )
    ticker_prices = ticker_prices[ticker_prices['tradeDate'].isin(business_dates)]

    price_panel_builder = PricePanel()
    price_panel_builder.fit(dataset=ticker_prices)
    price_panel_builder.panel.to_csv(
        preprocessed_data_path / f'{ticker.name}_prices_panel.csv', index=False,
    )

    # --- Synthetic Spread ---
    synthetic_spread_builder = SyntheticSpreadBuilder(method=HedgeMethod.OLS, windows=[250])
    synthetic_spread_data = synthetic_spread_builder.compute(price_panel_builder.panel)
    synthetic_spread_data.to_csv(
        preprocessed_data_path / f'{ticker.name}_synthetic_spread_db.csv', index=False,
    )

    # --- Volume ---
    all_volume = reader.read_volume()
    for column in _NUMERIC_VOLUME_COLUMNS:
        all_volume[column] = pd.to_numeric(all_volume[column], errors='coerce')
    ticker_volume = all_volume[all_volume['Name'] == ticker.value].copy()
    ticker_volume['tradeDate'] = pd.to_datetime(ticker_volume['tradeDate']).dt.date
    ticker_volume = ticker_volume[ticker_volume['tradeDate'].isin(business_dates)]

    volume_panel_builder = VolumePanel()
    volume_panel_builder.fit(dataset=ticker_volume)
    volume_panel_builder.panel.to_csv(
        preprocessed_data_path / f'{ticker.name}_volume_panel.csv', index=False,
    )

    # --- Open Interest ---
    all_open_interest = reader.read_openinterest()
    for column in _NUMERIC_OI_COLUMNS:
        all_open_interest[column] = pd.to_numeric(all_open_interest[column], errors='coerce')
    all_open_interest['F1_OI_Minus_F2_OI'] = all_open_interest['F1_OI'] - all_open_interest['F2_OI']
    ticker_open_interest = all_open_interest[all_open_interest['Name'] == ticker.value].copy()
    ticker_open_interest['tradeDate'] = pd.to_datetime(ticker_open_interest['tradeDate']).dt.date
    ticker_open_interest = ticker_open_interest[ticker_open_interest['tradeDate'].isin(business_dates)]

    open_interest_panel_builder = OpenInterestPanel()
    open_interest_panel_builder.fit(dataset=ticker_open_interest)
    open_interest_panel_builder.panel.to_csv(
        preprocessed_data_path / f'{ticker.name}_openinterest_panel.csv', index=False,
    )

    # --- COT Legacy (Commercial + ManagedMoney from pre_raw) ---
    all_cot_legacy = reader.read_cot_legacy()
    all_cot_legacy.dropna(inplace=True)
    all_cot_legacy = all_cot_legacy[_LEGACY_COT_COLUMNS]
    ticker_cot_legacy = all_cot_legacy[all_cot_legacy['Name'] == ticker.value].copy()

    cot_legacy_panel_builder = COTPanel()
    cot_legacy_panel_builder.fit(dataset=ticker_cot_legacy)
    cot_legacy_panel_builder.panel.to_csv(
        preprocessed_data_path / f'{ticker.name}_cot_legacy_panel.csv', index=False,
    )

    legacy_dataset_builder = WeeklyDataSetBuilder()
    legacy_dataset_builder.fit(
        cot_db=cot_legacy_panel_builder.panel,
        synthetic_spread_db=synthetic_spread_data,
        volume_db=volume_panel_builder.panel,
        openinterest_db=open_interest_panel_builder.panel,
    )
    legacy_dataset_builder.data.to_csv(
        preprocessed_data_path / f'{ticker.name}_legacy_dataset.csv', index=False,
    )

    legacy_daily_builder = DailyDataSetBuilder()
    legacy_daily_builder.fit(
        cot_db=cot_legacy_panel_builder.panel,
        price_panel=price_panel_builder.panel,
        volume_panel=volume_panel_builder.panel,
        openinterest_panel=open_interest_panel_builder.panel,
        synthetic_spread_db=synthetic_spread_data,
    )
    legacy_daily_builder.data.to_csv(
        preprocessed_data_path / f'{ticker.name}_legacy_daily_dataset.csv', index=False,
    )

    # --- COT Full (all 5 categories from cot-ingest) ---
    all_cot_full = reader.read_cot()
    all_cot_full.dropna(inplace=True)
    ticker_cot_full = all_cot_full[all_cot_full['Name'] == ticker.value].copy()

    cot_full_panel_builder = COTPanel()
    cot_full_panel_builder.fit(dataset=ticker_cot_full)
    cot_full_panel_builder.panel.to_csv(
        preprocessed_data_path / f'{ticker.name}_cot_panel.csv', index=False,
    )

    dataset_builder = WeeklyDataSetBuilder()
    dataset_builder.fit(
        cot_db=cot_full_panel_builder.panel,
        synthetic_spread_db=synthetic_spread_data,
        volume_db=volume_panel_builder.panel,
        openinterest_db=open_interest_panel_builder.panel,
    )
    dataset_builder.data.to_csv(
        preprocessed_data_path / f'{ticker.name}_dataset.csv', index=False,
    )

    daily_builder = DailyDataSetBuilder()
    daily_builder.fit(
        cot_db=cot_full_panel_builder.panel,
        price_panel=price_panel_builder.panel,
        volume_panel=volume_panel_builder.panel,
        openinterest_panel=open_interest_panel_builder.panel,
        synthetic_spread_db=synthetic_spread_data,
    )
    daily_builder.data.to_csv(
        preprocessed_data_path / f'{ticker.name}_daily_dataset.csv', index=False,
    )


if __name__ == "__main__":
    preprocess_all(ticker=FutureTicker.WTI)
    preprocess_all(ticker=FutureTicker.BRENT)
    preprocess_all(ticker=FutureTicker.RBOB)
    preprocess_all(ticker=FutureTicker.HEATING_OIL)
    preprocess_all(ticker=FutureTicker.GASOIL)
