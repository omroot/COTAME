"""Saver for raw data outputs."""

from pathlib import Path

import pandas as pd


class RawDataSaver:
    """Saves DataFrames as CSVs to the raw_data directory."""

    def __init__(self, raw_data_directory: Path) -> None:
        self.raw_data_directory = raw_data_directory

    def save(self, dataframe: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame to a CSV file in the raw data directory."""
        dataframe.to_csv(self.raw_data_directory / filename, index=False)
