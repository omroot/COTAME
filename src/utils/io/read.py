"""Readers for raw and preprocessed data directories."""

from pathlib import Path

import pandas as pd

from src.preprocessing.base import FutureTicker


class RawDataReader:
    """Reads unified raw data CSVs from the raw_data directory."""

    def __init__(self, raw_data_directory: Path) -> None:
        self.raw_data_directory = raw_data_directory

    def _read(self, filename: str) -> pd.DataFrame:
        """Read a CSV file from the raw data directory."""
        return pd.read_csv(self.raw_data_directory / filename)

    def read_prices(self) -> pd.DataFrame:
        """Read the unified prices database."""
        return self._read("prices_db.csv")

    def read_volume(self) -> pd.DataFrame:
        """Read the unified volume database."""
        return self._read("volume_db.csv")

    def read_openinterest(self) -> pd.DataFrame:
        """Read the unified open interest database."""
        return self._read("openinterest_db.csv")

    def read_cot(self) -> pd.DataFrame:
        """Read the unified COT database (full disaggregated, from cot-ingest)."""
        return self._read("cot_db.csv")

    def read_cot_legacy(self) -> pd.DataFrame:
        """Read the legacy COT database (Commercial + ManagedMoney only, from pre_raw)."""
        return self._read("cot_legacy_db.csv")


class PreprocessedDataReader:
    """Reads preprocessed feature panels and datasets."""

    def __init__(self, preprocessed_data_directory: Path) -> None:
        self.preprocessed_data_directory = preprocessed_data_directory

    def _read(self, filename: str) -> pd.DataFrame:
        """Read a CSV file from the preprocessed data directory."""
        return pd.read_csv(self.preprocessed_data_directory / filename)

    def read_prices(self, ticker: FutureTicker) -> pd.DataFrame:
        """Read the price feature panel for a ticker."""
        return self._read(f"{ticker.name}_prices_panel.csv")

    def read_volume(self, ticker: FutureTicker) -> pd.DataFrame:
        """Read the volume feature panel for a ticker."""
        return self._read(f"{ticker.name}_volume_panel.csv")

    def read_openinterest(self, ticker: FutureTicker) -> pd.DataFrame:
        """Read the open interest feature panel for a ticker."""
        return self._read(f"{ticker.name}_openinterest_panel.csv")

    def read_cot(self, ticker: FutureTicker) -> pd.DataFrame:
        """Read the COT feature panel for a ticker."""
        return self._read(f"{ticker.name}_cot_panel.csv")

    def read_dataset(self, ticker: FutureTicker) -> pd.DataFrame:
        """Read the weekly dataset for a ticker."""
        return self._read(f"{ticker.name}_dataset.csv")

    def read_daily_dataset(self, ticker: FutureTicker) -> pd.DataFrame:
        """Read the daily dataset for a ticker."""
        return self._read(f"{ticker.name}_daily_dataset.csv")
