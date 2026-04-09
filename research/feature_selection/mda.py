import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator

from sklearn.metrics import confusion_matrix

class MDA:
    """
        A class to compute Mean Decrease Accuracy (MDA) for feature importance.

        Parameters
        ----------
        model : BaseEstimator
            The  model for evaluation.
        cv : BaseCrossValidator
            The cross-validation strategy to use.
        clusters : dict, optional
            Dictionary where keys represent cluster names and values represent lists of feature indices
            to shuffle together. If None, shuffling is performed on individual features.
        is_classification : bool, default=True
            Indicates whether the task is classification (True) or regression (False).

        Methods
        ----------
        model : BaseEstimator
            The machine learning model passed during initialization.
        cv : BaseCrossValidator
            The cross-validation strategy passed during initialization.
        clusters : dict or None
            The clustering of features for grouped shuffling.
        is_classification : bool
            The task type indicator.
        mda : pd.DataFrame or None
            DataFrame containing the MDA results after fitting, including the mean and standard error.
        """



    def __init__(self,
                 model: BaseEstimator,
                 cv: BaseCrossValidator,
                 clusters: dict=None,
                 is_classification: bool = True):
        self.model = model
        self.cv = cv
        self.clusters = clusters
        self.is_classification = is_classification
        self.mda = None
    def fit(self, X: pd.DataFrame, y:pd.Series)->None:
        baseline_performance = pd.Series()
        if self.clusters is None:
            shuffled_performance = pd.DataFrame(columns=X.columns)
        else:
            shuffled_performance = pd.DataFrame(columns=self.clusters.keys())
        for i, (train, test) in enumerate(self.cv.split(X)):
            X_train, y_train = X.iloc[train], y.iloc[train]
            X_test, y_test = X.iloc[test], y.iloc[test]
            self.model.fit(X_train, y_train)
            # Predictions before shuffling
            y_test_hat = self.model.predict(X_test)
            if self.is_classification:
                conf_mat = confusion_matrix(y_test, y_test_hat)
                baseline_performance.loc[i] =  np.trace(conf_mat) / np.sum(conf_mat)
            else:
                baseline_performance.loc[i] = pd.DataFrame({'y': y_test, 'yhat': y_test_hat}).corr().iloc[0,1]

            if self.clusters is None:
                for j in X.columns:
                    X_test_ = X_test.copy(deep=True)
                    # Shuffle one column
                    np.random.shuffle(X_test_[j].values)
                    # Predictions after shuffling
                    y_test_hat_ = self.model.predict(X_test_)

                    if self.is_classification:
                        conf_mat = confusion_matrix(y_test, y_test_hat_)
                        shuffled_performance.loc[i, j] =  np.trace(conf_mat) / np.sum(conf_mat)
                    else:
                        shuffled_performance.loc[i, j] = pd.DataFrame({'y': y_test, 'yhat': y_test_hat_}).corr().iloc[0,1]


            else:
                for cluster_name , cluster_indices in self.clusters.items():
                    X_test_ = X_test.copy(deep=True)
                    for k in cluster_indices:
                        np.random.shuffle(X_test_[k].values)
                    y_test_hat_ = self.model.predict(X_test_)
                    if self.is_classification:
                        conf_mat = confusion_matrix(y_test, y_test_hat_)
                        shuffled_performance.loc[i, cluster_name] =  np.trace(conf_mat) / np.sum(conf_mat)
                    else:
                        shuffled_performance.loc[i, cluster_name] = pd.DataFrame({'y': y_test, 'yhat': y_test_hat_}).corr().iloc[0,1]

            mda_folds = (-1 * shuffled_performance).add(baseline_performance, axis=0)
        mda = pd.concat({'mean': mda_folds.mean(), 'std': mda_folds.std() * mda_folds.shape[0] ** -0.5}, axis=1)
        mda.sort_values(by='mean', inplace=True, ascending=True)
        self.mda = mda

