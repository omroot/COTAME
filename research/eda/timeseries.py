
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import grangercausalitytests

def test_adf_stationarity(data: pd.Series,
                          alpha: float = 0.05) -> tuple[float, float, str]:
    """ Run an ADF stationarity test. """

    adf_result = adfuller(data.values)
    t_stat, p_value, _, _, critical_values, _ = adf_result
    print(f"ADF statistic: {t_stat: .2f}")
    print(f"ADF p-value: {p_value: .2f}")
    # ADF null hypothesis: there is a unit root and the series is non-stationary
    print(f"ADF: non-stationary - unit root"
          if p_value >= alpha else
          f"ADF: stationary or only difference-stationary"

          )

    if p_value < alpha:
        conclusion = "The series is stationary"
    else:
        conclusion = "The series is not stationary"
    return t_stat, p_value, conclusion


def test_kpss_stationarity(data: pd.Series,
                              alpha: float = 0.05) -> tuple[float, float, str]:
    """ Run an KPSS stationarity test. """

    kpss_result = kpss(data.values)
    statistic, p_value, n_lags, critical_values, _ = kpss_result
    print(f"KPSS statistic: {statistic: .2f}")
    print(f"KPSS p-value: {p_value: .2f}")
    # KPSS null hypothesis:  is trend stationary or has no unit root
    print(f"KPSS: non-stationary - unit root"
          if p_value < alpha else
          f"KPSS: stationary or only difference-stationary"

          )

    if p_value >= alpha:
        conclusion = "The series is stationary"
    else:
        conclusion = "The series is not stationary"
    return statistic, p_value, conclusion


def test_xy_cointegration(dataset: pd.DataFrame,
                           x_name: str,
                           y_name: str,
                           maxlag: int =10,
                          alpha: float = 0.05) -> tuple[float, float, str]:
    """ Test the cointegration between two variables x and y """

    score , p_value, _ = coint(dataset[x_name],
                              dataset[y_name],
                              maxlag=10)

    if p_value < alpha:
        conclusion = f" {x_name} and {y_name} are co-integrated and there must be Granger causality between both."
    else:
        conclusion = f" {x_name} and {y_name} are NOT co-integrated and  Granger causality is uncertain."
    return score, p_value, conclusion

def test_xy_grangercausality(dataset: pd.DataFrame,
                             cause_name: str,
                             effect_name: str,
                             maxlag: int = 10):

    """ Run Granger causality test between cause_name and effect_name."""
    result = grangercausalitytests(x=dataset[[effect_name, cause_name]],
                                   verbose=True,
                                   maxlag=maxlag
                                   )
    return result
