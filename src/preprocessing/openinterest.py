"""Open interest feature panel: computes OI changes over rolling windows."""

from typing import Optional

import pandas as pd


class OpenInterestPanel:
    """Builds an open interest feature panel with lagged and forward OI changes.

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
        """Compute OI changes over lookback windows."""
        dataset.sort_values(by='tradeDate', ascending=True, inplace=True)
        open_interest_columns = ['F1_OI', 'F2_OI', 'F3_OI', 'AGG_OI']
        for column_name in open_interest_columns:
            for window in self.lookback_windows:
                dataset[f'prior_{window}D_{column_name}_change'] = (
                    dataset[column_name] - dataset[column_name].shift(window)
                )
        return dataset

    def compute_forward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute forward OI changes over lookforward windows."""
        dataset.sort_values(by='tradeDate', ascending=False, inplace=True)
        open_interest_columns = ['F1_OI', 'F2_OI', 'F3_OI', 'AGG_OI']
        for column_name in open_interest_columns:
            for window in self.lookforward_windows:
                dataset[f'forward_{window}D_{column_name}_change'] = (
                    dataset[column_name].shift(-window) - dataset[column_name]
                )
        return dataset

    def fit(self, dataset: pd.DataFrame) -> None:
        """Compute all open interest features and store the resulting panel."""
        dataset = self.compute_backward_features(dataset)
        dataset = self.compute_forward_features(dataset)
        self.panel = dataset
