import itertools

import numpy as np
import pandas as pd
from pandas import Index

from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):

    def __init__(self,
                 n_splits:int,
                 data: pd.DataFrame,
                 n_purge: str=10,
                 n_embargo: int=0,
                 date_column: str = "tradeDate",
                 random_state: int = None ,
                 ):
        """ """
        super().__init__(n_splits, shuffle=False, random_state=random_state)
        self.data = data
        self.n_purge = n_purge
        self.n_embargo = n_embargo
        self.n_splits = n_splits
        self.date_column = date_column
        # Get unique dates
        self.unq_dates = data[self.date_column].unique()
        self.unq_dates.sort()

    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups: list = None) -> iter:
        """"""
        data_index = self.data.index
        if (X.index != data_index).any():
            raise ValueError("Index of X and data must be identical")
        date_splits = np.array_split(self.unq_dates, self.n_splits)
        for i in range(self.n_splits):
            if i ==0:
                pre_dates = np.array([], dtype = "datetime64[ns]")
            else:
                pre_dates = np.concatenate(date_splits[:i])
                pre_dates = pre_dates[: -self.n_purge]
            if i == self.n_splits-1:
                post_dates = np.concatenate(date_splits[i+1 :])
                post_dates = post_dates[self.n_purge+self.n_embargo : ]
            test_dates = date_splits[i]
            train_dates = np.concatenate([pre_dates, post_dates])
            train_mask = self.data[self.date_column].isin(train_dates)
            test_mask = self.data[self.date_column].isin(test_dates)
            train_index = data_index[train_mask]
            test_index = data_index[test_mask]
            yield train_index, test_index
    def get_n_splits(self, X: pd.DataFrame = None,
                     y: pd.DataFrame = None,
                     groups: list = None
                     ):
        """" Returns the number of splits in the cross-validator """
        return self.n_splits
    def visualize(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Returns the representation of train/test dates formed by the CV."""
        unique_dates = self.data[self.date_column].unique().tolist()
        split_counter = 1
        df_list = []
        for train_idx, test_idx in self.split(X):
            d = {"split": split_counter}
            test = self.data.loc[test_idx]
            train = self.data.loc[train_idx]
            train_dates = set(train[self.date_column].unique().tolist())
            test_dates = set(test[self.date_column].unique().tolist())
            for date in unique_dates:
                if date in train_dates:
                    d[date] = "train"
                elif date in test_dates:
                    d[date] = "test"
                else:
                    d[date] = ""
            df_list.append(d)
            split_counter +=1
        date_df = pd.DataFrame(df_list)
        date_df = date_df.set_index("split")
        date_df = date_df.T
        return date_df

    def __repr__(self)-> str:
        """ Represents the class instance in a debug context """
        ret = "PurgedKFold\n"
        ret += f"\tdata points: {len(self.data):, .0f}\n"
        ret +=f"\tdate column: {self.date_column }\n"
        ret += f"\tunique dates: {len(self.unq_dates)}\n"
        ret += f"\tpurge dates: {len(self.n_purge)}\n"
        ret += f"\tembargo dates: {len(self.n_embargo)}\n"
        ret += f"\tn_splits: {len(self.n_splits)}\n"
        return ret

    def __str__(self)->str:
        """Returns the class instances as a string"""
        ret = "PurgedKFold CV"
        return ret







