
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from enum import Enum

class FutureTicker(Enum):
    """Enumeration of common futures tickers with display names."""
    WTI = ("CL", "wti")       # West Texas Intermediate Crude Oil
    RBOB = ("XB", "rbob")     # Reformulated Gasoline Blendstock
    HEATING_OIL = ("HO", "ho") 
    GASOIL = ("QS", "gasoil")
    BRENT = ("CO", "br")

    def __init__(self, ticker_symbol: str, name: str):
        self._ticker_symbol = ticker_symbol
        self._name = name

    @property
    def value(self) -> str:
        """Exchange ticker symbol."""
        return self._ticker_symbol

    @property
    def name(self) -> str:
        """Short standardized name."""
        return self._name


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for column in X.columns:
                X[column] = winsorize(X[column], limits=self.limits)
        else:
            X = winsorize(X, limits=self.limits)
        return X
