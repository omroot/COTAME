import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional


def zeta_significance(
        data: pd.DataFrame,
        feature_name: str,
        response_name: str,
        class_of_interest: Optional[Union[int, str]] = None,
        k: Optional[int] = None,
        plot: bool = False
) -> pd.DataFrame:
    """
    Calculate the zeta significance score for each category or bin of a feature.

    Parameters:
    - data (pd.DataFrame): The dataset containing the feature and response columns.
    - feature_name (str): Name of the feature column, which can be categorical or numerical.
                          If numerical, it will be discretized into bins if `k` is provided.
    - response_name (str): Name of the response column (target variable).
    - class_of_interest (Optional[Union[int, str]]): The specific class to test against.
                          If None, zeta profiles will be computed for all unique classes.
    - k (Optional[int]): Number of bins for discretizing a numerical feature. Ignored if the feature is categorical.
    - plot (bool): If True, plot the zeta-profile for each class in the response column, with significance thresholds
                   indicated at ±2.

    Returns:
    - pd.DataFrame: A DataFrame showing each category/bin, its zeta score, and interpretation for each class.
    """
    # Discretize if the feature is numerical
    if pd.api.types.is_numeric_dtype(data[feature_name]) and k is not None:
        data['binned_feature'] = pd.cut(data[feature_name], bins=k, labels=False)
        categorical_feature_name = 'binned_feature'
    else:
        categorical_feature_name = feature_name

    # Determine classes to evaluate
    classes_to_evaluate = [class_of_interest] if class_of_interest is not None else data[response_name].unique()

    # Create an empty list to store results
    results = []

    # For each class, calculate zeta significance scores
    for target_class in classes_to_evaluate:
        # Calculate p(c), the prior probability of the current class of interest
        p_c = np.mean(data[response_name] == target_class)

        class_results = []  # Store results for the current class

        # For each unique value/bin of the feature, calculate zeta
        for categorical_feature_value in data[categorical_feature_name].unique():
            # Get the subset of data where the feature equals v_X
            subset = data[data[categorical_feature_name] == categorical_feature_value]
            N_X_v = len(subset)  # N_{X_v}, the number of instances with feature value/bin categorical_feature_value

            # Calculate p(c | X=v) for the current feature value/bin
            p_c_given_X_v = np.mean(subset[response_name] == target_class)

            # Calculate zeta score
            if N_X_v > 0:
                zeta_score = np.sqrt(N_X_v) * (p_c_given_X_v - p_c) / np.sqrt(p_c * (1 - p_c))
            else:
                zeta_score = 0  # Avoid division by zero if N_X_v is 0

            # Interpret the zeta score
            if zeta_score > 2:
                interpretation = f"{feature_name}={categorical_feature_value} likely predicts {target_class}"
            elif zeta_score < -2:
                interpretation = f"{feature_name}={categorical_feature_value} likely predicts not {target_class}"
            else:
                interpretation = f"{feature_name}={categorical_feature_value} does not significantly predict {target_class}"

            # Append the result for the current class and feature value
            class_results.append({
                "Class": target_class,
                "Feature_Bin": categorical_feature_value,
                "Zeta_Score": zeta_score,
                "Interpretation": interpretation
            })

        # Append class results to main results list
        results.extend(class_results)

    # Convert results into a DataFrame for easy reading
    result_df = pd.DataFrame(results)

    # Plot the zeta-profile if requested
    if plot:
        plt.figure(figsize=(10, 6))

        # Plot each class's zeta-profile separately
        for target_class in classes_to_evaluate:
            class_df = result_df[result_df["Class"] == target_class]
            plt.plot(
                class_df['Feature_Bin'],
                class_df['Zeta_Score'],
                marker='o',
                linestyle='-',
                label=f'Class {target_class}'
            )

        # Threshold lines for significance
        plt.axhline(2, color='r', linestyle='--', label='Significant threshold (positive)')
        plt.axhline(-2, color='r', linestyle='--', label='Significant threshold (negative)')

        plt.xlabel('Feature Bins/Values')
        plt.ylabel('Zeta Score')
        plt.title(f'Zeta Significance Profile of {feature_name}')
        plt.legend()
        plt.show()
        plt.close()

    return result_df

# Example usage
if __name__ == '__main__':
    # Assume `df` is a DataFrame containing your data, with 'feature_name' as a numerical or categorical feature
    # and 'class' as the class column.
    df = pd.DataFrame({
        'feature_name': [1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7],
        'class': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    })
    zeta_result = zeta_significance(df, 'feature_name', 'class', k=4, plot=True)
    print(zeta_result)
