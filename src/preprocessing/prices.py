import pandas as pd


class PricePanel():

    def __init__(self,
                 lookback_windows: int = list(range(1, 20)),
                 lookforward_windows: int = list(range(1, 20))) -> None:
        self.lookback_windows = lookback_windows
        self.lookforward_windows = lookforward_windows
        self.panel = None

    def compute_backward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['F1MinusF2_RolledPrice'] = dataset['F1_RolledPrice'] - dataset['F2_RolledPrice']
        for name in ['F1_RolledPrice',
                     'F2_RolledPrice',
                     'F3_RolledPrice',
                     'F1MinusF2_RolledPrice'
                     ]:
            for w in self.lookback_windows:
                dataset[f'prior_{w}D_{name}_change'] = dataset[f'{name}'] - dataset[f'{name}'].shift(w)
            # computing rolling volatility
            dataset[f'{name}_rolling_20D_volatility'] = dataset[f'prior_1D_{name}_change'].rolling(window=20).std()
        return dataset

    def compute_forward_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        for name in ['F1_RolledPrice',
                     'F2_RolledPrice',
                     'F3_RolledPrice',
                     'F1MinusF2_RolledPrice'
                     ]:
            for w in self.lookforward_windows:
                dataset[f'forward_{w}D_{name}_change'] = dataset[f'{name}'].shift(-w) - dataset[f'{name}']
        return dataset

    def fit(self, dataset: pd.DataFrame) -> None:
        dataset['month'] = [d.strftime('%Y-%m') for d in dataset['tradeDate']]
        dataset['F1MinusF2_RolledPrice'] = dataset['F1_RolledPrice'] - dataset['F2_RolledPrice']
        dataset = self.compute_backward_features(dataset)
        dataset = self.compute_forward_features(dataset)
        self.panel = dataset
