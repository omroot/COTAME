import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from typing import List
from enum import Enum


class HedgeMethod(Enum):
    OLS = "ols"
    PCA = "pca"


class SyntheticSpreadBuilder:
    """
    Class to compute rolling hedge ratios (betas) between F1 and F2 using OLS or PCA.

    Attributes:
        method (HedgeMethod): Method to use ('OLS' or 'PCA').
        windows (list[int]): List of rolling window sizes to compute hedge ratios.
    """

    def __init__(self, method: HedgeMethod = HedgeMethod.OLS, windows: List[int] = [10, 20]):
        if not isinstance(method, HedgeMethod):
            raise ValueError("method must be an instance of HedgeMethod Enum")
        self.method = method
        self.windows = windows

    def _rolling_beta_ols(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        betas = []
        for i in range(len(x)):
            if i < window:
                betas.append(np.nan)
            else:
                x_window = x[i - window:i]
                y_window = y[i - window:i]
                X = sm.add_constant(x_window)
                model = sm.OLS(y_window, X).fit()
                betas.append(model.params[1])
        return pd.Series(betas, index=x.index)

    def _rolling_beta_pca(self, f1: pd.Series, f2: pd.Series, window: int) -> pd.Series:
        betas = []
        for i in range(len(f1)):
            if i < window:
                betas.append(np.nan)
            else:
                window_data = np.column_stack((f1[i - window:i], f2[i - window:i]))
                pca = PCA(n_components=1)
                pca.fit(window_data)
                pc1 = pca.components_[0]
                if np.abs(pc1[1]) > 1e-6:
                    beta = pc1[0] / pc1[1]
                else:
                    beta = np.nan
                betas.append(beta)
        return pd.Series(betas, index=f1.index)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling hedge ratios for specified windows using selected method.

        Parameters:
            df (pd.DataFrame): Must contain ['tradeDate', 'F1_RolledPrice', 'F2_RolledPrice']

        Returns:
            pd.DataFrame: Original DataFrame with added beta columns.
        """
        df = df.copy()
        df = df.sort_values('tradeDate').reset_index(drop=True)

        for window in self.windows:
            if self.method == HedgeMethod.OLS:
                beta_series = self._rolling_beta_ols(
                    df["F2_RolledPrice"], df["F1_RolledPrice"], window
                )
                df[f"beta_ols_{window}"] = beta_series
            elif self.method == HedgeMethod.PCA:
                beta_series = self._rolling_beta_pca(
                    df["F1_RolledPrice"], df["F2_RolledPrice"], window
                )
                df[f"beta_pca_{window}"] = beta_series

        return df
