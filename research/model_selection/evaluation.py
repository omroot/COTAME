

import numpy as np
import pandas as pd

import optuna
from optuna import pruners

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection._split import BaseCrossValidator
from research.model_selection.CombinatorialPurgedCV import cpcv_predict  

RANDOM_STATE = 42

def build_estimator_from_trial(trial: optuna.Trial):
    """Choose a model family and its hyperparameters conditionally."""
    model_name = trial.suggest_categorical(
        "model",
        ["enet", "lasso", "ols", "ridge", "rf", "extra", "xgb"]
    )

    if model_name == "enet":
        alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        base = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "lasso":
        alpha = trial.suggest_float("alpha", 1e-6, 1e1, log=True)
        base = Lasso(alpha=alpha, random_state=RANDOM_STATE)
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "ridge":
        alpha = trial.suggest_float("alpha", 1e-6, 1e3, log=True)
        base = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "ols":
        base = LinearRegression()
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "rf":
        n_estimators = trial.suggest_int("n_estimators", 200, 1200, step=200)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        est = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_name == "extra":
        n_estimators = trial.suggest_int("n_estimators", 200, 1200, step=200)
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        est = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    else:  # "xgb"
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        est = XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    return est


def objective(trial, Xy, feature_names, response_name, cv, n_jobs_cpcv=10 ):
    estimator = build_estimator_from_trial(trial)

    y_true = Xy[response_name].values
    y_pred  = cpcv_predict(
        estimator,
        Xy[feature_names],
        Xy[response_name],
        cv=cv,
        method="predict",
        n_jobs=n_jobs_cpcv
    )

    y_pred = np.asarray(y_pred, float).ravel()
    mask = np.isfinite(y_pred)
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    if y_pred_m.size == 0:
        return 1e9

    val = np.corrcoef(y_true_m, y_pred_m)[0,1]
    


    trial.report(float(val), step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return float(val)
def find_best_model(Xy: pd.DataFrame, 
                   feature_names: list[str], 
                   response_name: str, 
                   cv:BaseCrossValidator,
                   n_trials: int = 50,
                   n_jobs_cpcv: int = 10,
                   seed: int = 42):
    """
    Runs Optuna, returns:
      - best fitted estimator (refit on FULL data)
      - best_params (dict) for overall best
      - study (Optuna Study)
      - best_per_model (dict mapping model -> {params, score})
    """
    Xy = Xy[['tradeDate']+feature_names+[response_name]].dropna()
    Xy.reset_index(drop=True, inplace= True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruners.MedianPruner(n_warmup_steps=1)
    )

    study.optimize(
        lambda t: objective(
            t, Xy, feature_names, response_name, cv,
            n_jobs_cpcv=n_jobs_cpcv 
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # ---- find best trial per model family ----
    best_per_model = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        model_type = trial.params["model"]
        score = trial.value
        # Keep only the best score (lower is better)
        if model_type not in best_per_model or score < best_per_model[model_type]["score"]:
            best_per_model[model_type] = {
                "params": trial.params,
                "score": score
            }

    # ---- rebuild overall best estimator ----
    best_params = study.best_trial.params
    best_estimator = build_estimator_from_trial(optuna.trial.FixedTrial(best_params))

    return best_estimator, best_params, study, best_per_model
