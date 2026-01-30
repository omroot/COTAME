import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator

from research.feature_selection.sfi import SFI


class MRMR:
    """
    Maximum Relevance Minimum Redundancy (MRMR) feature selector.

    Uses Single Feature Importance (SFI) to rank features by relevance,
    then greedily selects features while removing those that are too
    correlated with already-selected ones.

    Parameters
    ----------
    model : BaseEstimator
        The model used for SFI evaluation.
    cv : BaseCrossValidator
        The cross-validation strategy.
    correlation_threshold : float, default=0.5
        Maximum absolute correlation allowed between a candidate feature
        and any already-selected feature. Features exceeding this threshold
        are removed from consideration.
    similarity_method : str, default='pearson'
        Method used to compute the feature correlation matrix.
    is_classification : bool, default=True
        Whether the task is classification (True) or regression (False).

    Attributes
    ----------
    selected_features : list[str] or None
        List of selected feature names after fitting.
    sfi : SFI or None
        The fitted SFI instance, available after fitting.
    correlation_matrix : pd.DataFrame or None
        The feature correlation matrix used during selection.
    """

    def __init__(self,
                 model: BaseEstimator,
                 cv: BaseCrossValidator,
                 correlation_threshold: float = 0.5,
                 similarity_method: str = 'pearson',
                 is_classification: bool = True):
        self.model = model
        self.cv = cv
        self.correlation_threshold = correlation_threshold
        self.similarity_method = similarity_method
        self.is_classification = is_classification
        self.selected_features: list = None
        self.sfi: SFI = None
        self.correlation_matrix: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.sfi = SFI(model=self.model, cv=self.cv, is_classification=self.is_classification)
        self.sfi.fit(X, y)

        self.correlation_matrix = X.corr(method=self.similarity_method).abs()

        ranked_features = list(self.sfi.importances.index)
        selected = []
        remaining = set(ranked_features)

        for feature in ranked_features:
            if feature not in remaining:
                continue
            selected.append(feature)
            remaining.discard(feature)
            to_remove = set()
            for candidate in remaining:
                if self.correlation_matrix.loc[feature, candidate] > self.correlation_threshold:
                    to_remove.add(candidate)
            remaining -= to_remove

        self.selected_features = selected
