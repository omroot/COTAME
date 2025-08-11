from joblib import Parallel, delayed


from sklearn.base import BaseEstimator

from math import comb
from typing import Generator, Union
import pandas as pd
import numpy as np
from itertools import combinations


class CombinatorialPurgedCV():

    """Combinatorial Purged Cross-Validatpr with Purging

    This class identifies training and testing indicies that split the data into train/test sets. It also implements purging to avoid lookahead bias.


    """

    def __init__(self,
                 n_partitions: int,
                 k:int,
                 purge_amount: int):
        """

        Args:
        :param n_partitions: the total number of partitions
        :param k: the number of partitions to include in the test set
        :param purge_amount: the number of indices to be purged between train/test sets
        """
        self.n_splits =  comb(n_partitions, k)
        self.n_partitions = n_partitions
        self.k = k
        self.purge_amount = purge_amount

    def split(self,
              X: Union[np.ndarray, pd.Series, pd.DataFrame],
              y: Union[np.ndarray, pd.Series] = None,
              groups: np.ndarray = None)->Generator:
        """Generate indices to split data into training and test set with purging. Note that the indicies of X need to be in ascending order for the purging to work correctly, and an assertion error
        will be raised if X is a pd.Series or pd.DataFrame and X.index.is_monotonic_increasing is False

        Args:
             X: training data, where n_samples is the number of samples and n_features is the number of features.
             y: the target variable for supervised learning problems.
             groups: Group labels for the samples used while splitting the dataset into train/test set.
        Yields:
            train (np.ndarray): The training set indices for that split
            test (np.ndarray): the testing set indices for that split
        """
        if isinstance(X, pd.Series) | isinstance(X, pd.DataFrame):
            assert X.index.is_monotonic_increasing, "The indices of  X need to be in ascending order"
        n_samples = len(X)
        indices = np.arange(n_samples)
        partition_size = n_samples // self.n_partitions

        partitions = [indices[i* partition_size: (i+1) * partition_size ]  for i in range(self.n_partitions -1) ]
        partitions.append(indices[(self.n_partitions -1) * partition_size: ])

        test_combinations = combinations (range(self.n_partitions) , self.k )
        for test_indices in test_combinations:
            test_indices = list(test_indices)
            train_indices = list(set(range(self.n_partitions)) - set(test_indices))
            test_mask = np.zeros(n_samples, dtype=bool)
            train_mask = np.zeros(n_samples, dtype=bool)

            for i in test_indices:
                test_mask[partitions[i][0]:partitions[i][-1]+1 ] = True

            for i in train_indices:
                train_mask[partitions[i][0]:partitions[i][-1] + 1] = True

            # Apply purging by shifting the train_mask to exclude purge_amount indices before each test partition
            for test_index in test_indices:
                start_index = max(partitions[test_index][0] - self.purge_amount, 0 )
                train_mask[start_index: partitions[test_index][0] ] = False
            yield indices[train_mask], indices[test_mask]
    def get_n_splits(self,
                     X: np.ndarray = None,
                     y:np.ndarray=None,
                     groups: np.ndarray=None)->int:
        """Returns the number of splitting iterations in the cross-validator
        Args:
        :param X: Training data
        :param y: The target variable
        :param groups:
        :return:
        """
        return self.n_splits



def _fit_and_predict(estimator: BaseEstimator,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     train: np.ndarray,
                     test: np.ndarray,
                     fit_params: dict,
                     method: str)->pd.DataFrame:


    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train,:]
        y_train = y.iloc[train]
        X_test = X.iloc[test,:]
    elif isinstance(X, np.ndarray):
        X_train = X[train,:]
        y_train = y[train]
        X_test = X[test,:]
    else:
        raise ValueError("X must be either a pd.DataFrame or np.ndarray")
    fit_params = fit_params if fit_params is not None else {}
    estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method, None)
    if func is None:
        raise ValueError(f"Estimator must have a {method} method")
    predictions = func(X_test)
    # print(predictions.shape)
    if method == "predict_proba":
        predictions_df = pd.DataFrame(predictions, columns=[f"posterior_{i}" for i in range(predictions.shape[1])])
        predictions_df["index"] = test
    else:
        predictions_df  = pd.DataFrame({"yhat": predictions, "index": test})

    return  predictions_df

def cpcv_predict(
        estimator: BaseEstimator,
        X:pd.DataFrame,
        y: Union[pd.Series, np.ndarray]=None,
        cv: CombinatorialPurgedCV = None,
        n_jobs: int = None,
        verbose: bool = False,
        fit_params: dict = None,
        method: str = "predict" ) -> pd.Series:


    cpcv_splits = list(cv.split(X))
    parallel  = Parallel(n_jobs=n_jobs, verbose=verbose)
    predictions = parallel(
                    delayed(_fit_and_predict)(
                            estimator=estimator,
                             X=X,
                            y=y,
                            train=train,
                            test=test,
                            fit_params=fit_params,
                           method=method)
                           for train, test in cpcv_splits

                           )
    # aggregate multiple predictions

    if method == "predict_proba":
        predictions = pd.concat(predictions).reset_index(drop=True)
        predictions=predictions.groupby("index").mean()#.reset_index()
    elif method == "predict":
        # predictions = pd.concat(predictions).groupby("index").reset_index()["yhat"]
        predictions = pd.concat(predictions).groupby("index").mean().reset_index()["yhat"]

        # predictions = pd.concat(predictions).groupby("index").mean(axis=1).reset_index()["yhat"]

    return predictions


