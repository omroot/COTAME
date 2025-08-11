from pathlib import Path
import glob
import pickle
import pandas as pd

from src.preprocessing.base import FutureTicker

class RawDataReader():
    def __init__(self, raw_data_directory: Path):
        self.raw_data_directory = raw_data_directory

    def _read(self, fname:str)->pd.DataFrame:
        return pd.read_csv(fname)

    def read_prices(self) -> pd.DataFrame:
        file_name = str(self.raw_data_directory   ) + "/prices_db.csv"
        return  self._read(file_name)

    def read_volume(self) -> pd.DataFrame:
        file_name = str(self.raw_data_directory ) + "/volume_db.csv"
        return self._read(file_name)

    def read_openinterest(self) -> pd.DataFrame:
        file_name = str(self.raw_data_directory ) + "/openinterest_db.csv"
        return self._read(file_name)
    def read_cot(self) -> pd.DataFrame:
        file_name = str(self.raw_data_directory ) + "/cot_db.csv"
        return self._read(file_name)

class PreprocessedDataReader():
    def __init__(self, preprocessed_data_directory: Path):
        self.preprocessed_data_directory = preprocessed_data_directory

    def _read(self, fname:str)->pd.DataFrame:
        return pd.read_csv(fname)

    def read_prices(self, ticker: FutureTicker) -> pd.DataFrame:
        file_name = str(self.preprocessed_data_directory   ) + f"/{ticker.name}_prices_panel.csv"
        return  self._read(file_name)

    def read_volume(self, ticker: FutureTicker) -> pd.DataFrame:
        file_name = str(self.preprocessed_data_directory ) + f"/{ticker.name}_volume_panel.csv"
        return self._read(file_name)

    def read_openinterest(self, ticker: FutureTicker) -> pd.DataFrame:
        file_name = str(self.preprocessed_data_directory ) + f"/{ticker.name}_openinterest_panel.csv"
        return self._read(file_name)
    def read_cot(self, ticker: FutureTicker) -> pd.DataFrame:
        file_name = str(self.preprocessed_data_directory ) + f"/{ticker.name}_cot_panel.csv"
        return self._read(file_name)
    def read_dataset(self, ticker: FutureTicker) -> pd.DataFrame:
            file_name = str(self.preprocessed_data_directory ) + f"/{ticker.name}_dataset.csv"
            return self._read(file_name)