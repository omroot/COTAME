import pandas as pd

import numpy as np

from typing import Tuple, List, Dict
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from sklearn.model_selection import cross_val_predict


from research.model_selection.PurgedKFold import PurgedKFold



def run_hpo(dataset: pd.DataFrame,
            target: str,
            features: List[str],
            model,
            parameter_space: Dict) -> Dict:
    """ Run hyperparameter fine tuning using Bayesian optimization on one response, one model """

    Xy = dataset.copy(deep=True).replace(to_replace=np.Infinity, value=np.nan)
    Xy = Xy.replace(to_replace=-np.Infinity, value=np.nan)
    Xy = Xy.dropna(subset=[target])
    Xy.reset_index(drop = True, inplace = True)

    cv = PurgedKFold(
        n_splits=2,
        data=Xy,
        n_purge=10,

    )
    opt = BayesSearchCV(model,
                        parameter_space,
                        n_iter=10,
                        n_jobs=10,
                        n_points=4,
                        iid=False,
                        cv=cv,
                        verbose=1
                        )
    opt.fit(X=Xy[features], y=Xy[target])
    print(f"{target} best parameters: {opt.best_params_} ")

    yhat = cross_val_predict(estimator=opt.best_estimator_,
                             X=Xy[features],
                             y=Xy[target],
                             method = 'predict',
                             cv = cv,
                             n_jobs=8,
                             verbose = 0 )

    optimal_predicted_actual_correlation = pd.DataFrame(
        {'y': Xy[target],
         'yhat': yhat
        }
    ).corr().iloc[0,1]
    result = {'best_params': opt.best_params_,
              'best_performance':     optimal_predicted_actual_correlation
    }
    return result