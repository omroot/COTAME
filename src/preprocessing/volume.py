
import pandas as pd

class VolumePanel():

    def __init__(self,
                 lookback_windows: int = [1, 5, 10, 15, 20],
                 lookforward_windows: int = [1, 5, 10, 15, 20]) -> None:
        self.lookback_windows = lookback_windows
        self.lookforward_windows = lookforward_windows
        self.panel = None

    def compute_backward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.sort_values(by='tradeDate', ascending=True, inplace=True)
        for name in ['F1_Volume',
                     'F2_Volume',
                     'F3_Volume']:
            for w in self.lookback_windows:
                dataset[f'prior_cumulative_{w}D_{name}'] = dataset[f'{name}'].rolling(window=w).sum()
                dataset[f'prior_{w}D_{name}_change'] = dataset[f'{name}'] - dataset[f'{name}'].shift(w)

        return dataset

    def compute_forward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.sort_values(by='tradeDate', ascending=False, inplace=True)
        for name in ['F1_Volume',
                     'F2_Volume',
                     'F3_Volume']:
            for w in self.lookforward_windows:
                dataset[f'forward_cumulative_{w}D_{name}'] = dataset[f'{name}'].rolling(window=w).sum()
        return dataset

    def fit(self, dataset: pd.DataFrame) -> None:

        dataset = self.compute_backward_features(dataset)
        dataset = self.compute_forward_features(dataset)
        self.panel = dataset

