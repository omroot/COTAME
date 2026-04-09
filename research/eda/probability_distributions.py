import scipy.stats as stats
import pandas as pd
from typing import Tuple, Dict, Any

def test_normality_shapiro(data: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Test for normality using the Shapiro-Wilk test.

    Args:
        data (pd.Series): Input data (NaNs will be dropped).
        alpha (float): Significance level for the test.

    Returns:
        Dict[str, Any]: Dictionary containing statistic, p-value, and verdict.
    """
    # Remove NaN values
    clean_data = data.dropna()

    # Perform Shapiro-Wilk test
    shapiro_test = stats.shapiro(clean_data)

    # Determine verdict
    if shapiro_test.pvalue > alpha:
        verdict = "Likely normal"
    else:
        verdict = "Not normal"

    return {
        "statistic": shapiro_test.statistic,
        "p_value": shapiro_test.pvalue,
        "alpha": alpha,
        "verdict": verdict
    }

