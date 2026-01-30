import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.metrics import confusion_matrix


class SFI:
    """
    Single Feature Importance (SFI).

    Evaluates each feature individually by fitting the model on that single feature
    and scoring via cross-validation. This provides a relevance ranking that is
    independent of other features.

    Parameters
    ----------
    model : BaseEstimator
        The model used for evaluation.
    cv : BaseCrossValidator
        The cross-validation strategy.
    is_classification : bool, default=True
        Whether the task is classification (True) or regression (False).

    Attributes
    ----------
    importances : pd.DataFrame or None
        DataFrame with columns ['mean', 'std'] indexed by feature name,
        sorted descending by mean importance after fitting.
    """

    def __init__(self,
                 model: BaseEstimator,
                 cv: BaseCrossValidator,
                 is_classification: bool = True):
        self.model = model
        self.cv = cv
        self.is_classification = is_classification
        self.importances: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        scores = {}
        for feature in X.columns:
            fold_scores = []
            X_single = X[[feature]]
            for train, test in self.cv.split(X_single):
                X_train, y_train = X_single.iloc[train], y.iloc[train]
                X_test, y_test = X_single.iloc[test], y.iloc[test]
                self.model.fit(X_train, y_train)
                y_hat = self.model.predict(X_test)
                if self.is_classification:
                    conf_mat = confusion_matrix(y_test, y_hat)
                    fold_scores.append(np.trace(conf_mat) / np.sum(conf_mat))
                else:
                    corr = pd.DataFrame({'y': y_test, 'yhat': y_hat}).corr().iloc[0, 1]
                    fold_scores.append(corr)
            scores[feature] = fold_scores

        scores_df = pd.DataFrame(scores)
        self.importances = pd.concat({
            'mean': scores_df.mean(),
            'std': scores_df.std() * scores_df.shape[0] ** -0.5
        }, axis=1)
        self.importances.sort_values(by='mean', ascending=False, inplace=True)
