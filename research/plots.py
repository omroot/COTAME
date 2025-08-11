

from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from research.metrics import bin_summary_of_xy



def plot_bin_summary_of_xy(

        x: pd.Series,
        y: pd.Series,
        k:int,
        unique_flag : Optional[bool] = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,)->None:
    """ Calls the bin_summary_of_xy function and plots the returned bins."""
    bin_analytics = bin_summary_of_xy(x,y,k, unique_flag)
    plt.errorbar(

        x= bin_analytics["x_mean"],
        y=bin_analytics["y_mean"],
        yerr= bin_analytics["y_se"],
    )

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    return bin_analytics
def plot_heatmap(df: pd.DataFrame,
                 rows: List[str],
                 columns: List[str],
                 title: str,
                 fig_size: tuple)-> None:
    """Plot a heatmap on the specified columns and rows """
    plt.rcParams["figure.figsize"] = fig_size
    display_df = df[columns]
    display_df.index = rows
    sns.heatmap(display_df)
    plt.title(title)
    plt.show()
    plt.close()




