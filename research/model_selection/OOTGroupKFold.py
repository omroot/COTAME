
import itertools
import math
import random

from typing import Generator, Iterator, Tuple, Union


import numpy as np
import pandas as pd
from pandas import Index
from sklearn.model_selection._split import BaseCrossValidator, _BaseKFold

from research.model_selection.CombinatorialPurgedCV import CombinatorialPurgedCV
class OOTGroupKFold:

    def __init__(self,
                 n_splits: int,
                 tickers: pd.Series,
                 k:int = 2,
                 purge_amount: Union[int, float]=0,
                 verbose: bool=False
                 ):
        self.n_splits = n_splits
        self.tickers = tickers.unique()
        self.ticker_column_name = tickers.name
        random.shuffle(self.tickers)
        self.k=k
        self.purge_amount = purge_amount
        self.purge_amount_rows = self.purge_amount
        self.verbose = verbose

    def _ticker_groups(self) -> Tuple[np.ndarray, np.ndarray]:

        half = len(self.tickers) // 2
        return np.arange(half), np.arange(half, len(self.tickers))

    def  split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        assert np.array_equal(X.index, np.arange(len(X))), (
            "The index of X should be contiguous integers. "
            "Please reset the index of X using 'reset_index(drop=True)'."
        )
        if self.purge_amount < 1:
            self.purge_amount_rows = int(self.purge_amount * len(X))
        cpcv =CombinatorialPurgedCV(
            n_partitions=self.n_splits,
            k=self.k,
            purge_amount=self.purge_amount_rows



        )
        group_a, group_b = self._ticker_groups()
        for train_index, test_index in cpcv.split(X):
            # determine the tickers in the training set
            train_tickers = self.tickers[
                np.isin(
                    self.tickers, X.iloc[train_index][self.ticker_column_name].unique()
                )
            ]
            # Split the train tickers into two groups
            train_group_a = train_tickers[np.isin(train_tickers, self.tickers[group_a])]
            train_group_b = train_tickers[np.isin(train_tickers, self.tickers[group_b])]

            # Generate indices for train and test sets for each ticker group
            train_indices_a = X.index[
                X[self.ticker_column_name].isin(train_group_a) & X.index.isin(train_index)
            ].values
            test_indices_a = X.index[
                X[self.ticker_column_name].isin(train_group_b) & X.index.isin(test_index)
                ].values


            train_indices_b = X.index[
                X[self.ticker_column_name].isin(train_group_b) & X.index.isin(train_index)
            ].values
            test_indices_b = X.index[
                X[self.ticker_column_name].isin(train_group_a) & X.index.isin(test_index)
                ].values

            if self.verbose:
                print(
                    f"Train Group A length: {len(train_indices_a) }, "
                    f"Test Group A length: {len(test_indices_a)}, "
                )
                print(
                    f"Train Group B length: {len(train_indices_b) }, "
                    f"Test Group B length: {len(test_indices_b)}, "
                )

            yield train_indices_a, test_indices_a
            yield train_indices_b, test_indices_b
