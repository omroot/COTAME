"""Volume feature panel: computes cumulative volumes and volume changes."""

from typing import Optional

import pandas as pd


class VolumePanel:
    """Builds a volume feature panel with cumulative sums and lagged changes.

    Parameters
    ----------
    lookback_windows : list[int]
        Rolling windows (in days) for backward features.
    lookforward_windows : list[int]
        Rolling windows (in days) for forward features.
    """

    def __init__(
        self,
        lookback_windows: list[int] = [1, 5, 10, 15, 20],
        lookforward_windows: list[int] = [1, 5, 10, 15, 20],
    ) -> None:
        self.lookback_windows = lookback_windows
        self.lookforward_windows = lookforward_windows
        self.panel: Optional[pd.DataFrame] = None

    def compute_backward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative volumes and volume changes over lookback windows."""
        dataset.sort_values(by='tradeDate', ascending=True, inplace=True)
        volume_columns = ['F1_Volume', 'F2_Volume', 'F3_Volume']
        for column_name in volume_columns:
            for window in self.lookback_windows:
                dataset[f'prior_cumulative_{window}D_{column_name}'] = (
                    dataset[column_name].rolling(window=window).sum()
                )
                dataset[f'prior_{window}D_{column_name}_change'] = (
                    dataset[column_name] - dataset[column_name].shift(window)
                )
        return dataset

    def compute_forward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute forward cumulative volumes over lookforward windows."""
        dataset.sort_values(by='tradeDate', ascending=False, inplace=True)
        volume_columns = ['F1_Volume', 'F2_Volume', 'F3_Volume']
        for column_name in volume_columns:
            for window in self.lookforward_windows:
                dataset[f'forward_cumulative_{window}D_{column_name}'] = (
                    dataset[column_name].rolling(window=window).sum()
                )
        return dataset

    def fit(self, dataset: pd.DataFrame) -> None:
        """Compute all volume features and store the resulting panel."""
        dataset = self.compute_backward_features(dataset)
        dataset = self.compute_forward_features(dataset)
        self.panel = dataset
