import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict

from research.model_selection.PurgedKFold import PurgedKFold
from research.plots import plot_heatmap
def get_single_feature_importance(dataset: pd.DataFrame,
                                  model:BaseEstimator,
                                  features: list[str],
                                  responses: list[str],
                                  undersampling_rate: float = 0.25,
                                  n_splits: int = 5,
                                  n_purge: int = 30,
                                  plot: bool=False,
                                  verbose: bool = False,
                                  description: str = 'BWIC',
                                  figsize = (10,12)
                                  ) -> pd.DataFrame:

    dataset['tradeDate'] = pd.to_datetime(dataset['tradeDate']).dt.date
    feature_significance = {}
    for target in responses:
        if verbose:
            print(target)
        oos_predicted_actual_correlation_net = []
        for feat in features:
            if verbose:
                print(feat)
            Xy = dataset.dropna(subset=[target])
            Xy = Xy.sample(frac=undersampling_rate)
            Xy.reset_index(drop=True, inplace = True)
            cv = PurgedKFold(
                n_splits = n_splits,
                data = Xy,
                n_purge = n_purge

            )
            yhat = cross_val_predict(
                estimator = model,
                X=Xy[[feat]],
                y=Xy[target],
                method = 'predict',
                cv=cv,
                n_jobs = 8,
                verbose = 0

            )
            oos_predicted_actual_correlation_net.append(
                pd.DataFrame(
                    {'y': Xy[target],
                     'yhat': yhat

                    }

                ).corr().iloc[0,1])
    feature_significance[target]=oos_predicted_actual_correlation_net
    sfi_db = pd.DataFrame(feature_significance)
    sfi_db['Feature']=features
    sfi_db.sort_values(by=responses[-1], inplace=True)
    if plot:
        plot_heatmap(
            sfi_db,
            sfi_db['Feature'].tolist(),
            responses,
            title = f'SFI heatmap for {description} for target {target}',
            fig_size=figsize


        )







def get_feature_permutation_importance(

        dataset: pd.DataFrame,
        model: BaseEstimator,
        responses: list[str],
        responses_features_map: list[str],
        undersampling_rate: float = 0.25,
        n_splits: int = 5,
        n_purge: int = 30,
        plot: bool = False,
        verbose: bool = False,
        description: str = 'BWIC',
        figsize: tuple[int]=(10,12)
) -> dict:

    " Computes each feature's permutation importance "

    dataset['tradeDare'] = pd.to_datetime(dataset['tradeDare']).dt.date
    responses_pi_data = {}
    for target in responses:
        if verbose:
            print(target)
        features = responses_features_map[target]
        Xy = dataset.dropna(subset=[target])
        Xy = Xy.sample(frac=undersampling_rate)
        Xy.reset_index(drop = True, inplace = True)
        cv = PurgedKFold(
            n_splits=n_splits,
            data = Xy,
            n_purge=n_purge

        )
        pi_df_folds = []
        for i, (train_index, test_index) in enumerate(cv.split(Xy)):
            if verbose:
                print(f'Fold: {i}')
            train_data = Xy.iloc[train_index,:]
            train_data.reset_index(drop=True, inplace = True)
            model.fit(train_data[features], train_data[target])

            pi_result = permutation_importance(
                model,
                X=train_data[features],
                y=train_data[target],
                scoring='r2',
                n_jobs=8,
                n_repeats=4

            )
            # TO BE RE CHECKED
            pi_df = pd.DataFrame({i: pi_result['importances_mean']})
            pi_df.index = features
            pi_df = pi_df.T
            pi_df_folds.append(pi_df)

        pi_data = pd.concat(pi_df_folds)
        pi_data_means = pi_data.mean()
        sorted_columns = sorted(pi_data, key=lambda x: pi_data_means[x])
        pi_data = pi_data[sorted_columns]
        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.boxplot(pi_data, vert=False, labels=pi_data.columns)
            ax.axvline(x=0, alpha=0.5, linestyle=':')
            plt.title(
                f'{description} Permutation importance for modeling {target} '

            )
            plt.show()
            plt.close()
        responses_pi_data[target] = pi_data
    return  responses_pi_data




