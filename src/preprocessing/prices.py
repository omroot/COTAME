"""Price feature panel: computes backward/forward price changes and rolling volatility."""

from typing import Optional

import pandas as pd


class PricePanel:
    """Builds a price feature panel with lagged changes and rolling volatility.

    Parameters
    ----------
    lookback_windows : list[int]
        Lag windows (in days) for backward features.
    lookforward_windows : list[int]
        Lead windows (in days) for forward features.
    """

    def __init__(
        self,
        lookback_windows: list[int] = list(range(1, 20)),
        lookforward_windows: list[int] = list(range(1, 20)),
    ) -> None:
        self.lookback_windows = lookback_windows
        self.lookforward_windows = lookforward_windows
        self.panel: Optional[pd.DataFrame] = None

    def compute_backward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute lagged price changes and 20-day rolling volatility."""
        dataset['F1MinusF2_RolledPrice'] = dataset['F1_RolledPrice'] - dataset['F2_RolledPrice']
        price_columns = ['F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice', 'F1MinusF2_RolledPrice']
        for column_name in price_columns:
            for window in self.lookback_windows:
                dataset[f'prior_{window}D_{column_name}_change'] = (
                    dataset[column_name] - dataset[column_name].shift(window)
                )
            dataset[f'{column_name}_rolling_20D_volatility'] = (
                dataset[f'prior_1D_{column_name}_change'].rolling(window=20).std()
            )
        return dataset

    def compute_forward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute forward price changes for target variable construction."""
        price_columns = ['F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice', 'F1MinusF2_RolledPrice']
        for column_name in price_columns:
            for window in self.lookforward_windows:
                dataset[f'forward_{window}D_{column_name}_change'] = (
                    dataset[column_name].shift(-window) - dataset[column_name]
                )
        return dataset

    def fit(self, dataset: pd.DataFrame) -> None:
        """Compute all price features and store the resulting panel."""
        dataset['month'] = [date.strftime('%Y-%m') for date in dataset['tradeDate']]
        dataset['F1MinusF2_RolledPrice'] = dataset['F1_RolledPrice'] - dataset['F2_RolledPrice']
        dataset = self.compute_backward_features(dataset)
        dataset = self.compute_forward_features(dataset)
        self.panel = dataset
