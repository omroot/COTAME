

import numpy as np
import pandas as pd

import optuna
from optuna import pruners

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import SGDRegressor, Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection._split import BaseCrossValidator
from research.model_selection.CombinatorialPurgedCV import cpcv_predict  

RANDOM_STATE = 42

def build_estimator_from_trial(trial: optuna.Trial):
    """Choose a model family and its hyperparameters conditionally."""
    model_name = trial.suggest_categorical(
        "model",
        ["enet", "lasso", "ols", "ridge", "rf", "extra", "hgbm"]
    )

    if model_name == "enet":
        alpha = trial.suggest_float("enet_alpha", 1e-6, 1e-1, log=True)
        l1_ratio = trial.suggest_float("enet_l1_ratio", 0.0, 1.0)
        loss = trial.suggest_categorical("enet_loss", ["squared_error", "huber"])
        # (Optional) huber epsilon if chosen
        epsilon = trial.suggest_float("enet_epsilon", 1e-3, 0.2) if loss == "huber" else 0.1
        base = SGDRegressor(
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            loss=loss,
            epsilon=epsilon,
            max_iter=3000,
            tol=1e-3,
            random_state=RANDOM_STATE
        )
        # linear models benefit from scaling
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "lasso":
        alpha = trial.suggest_float("lasso_alpha", 1e-6, 1e1, log=True)
        base = Lasso(alpha=alpha, random_state=RANDOM_STATE)
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "ridge":
        alpha = trial.suggest_float("ridge_alpha", 1e-6, 1e3, log=True)
        base = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "ols":
        base = LinearRegression()
        est = Pipeline([("scaler", StandardScaler()), ("model", base)])

    elif model_name == "rf":
        n_estimators = trial.suggest_int("rf_n_estimators", 200, 1200, step=200)
        max_depth = trial.suggest_int("rf_max_depth", 3, 30)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None])
        est = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    elif model_name == "extra":
        n_estimators = trial.suggest_int("extra_n_estimators", 200, 1200, step=200)
        max_depth = trial.suggest_int("extra_max_depth", 3, 30)
        min_samples_leaf = trial.suggest_int("extra_min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical("extra_max_features", ["sqrt", "log2", None])
        est = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

    else:  # "hgbm"
        learning_rate = trial.suggest_float("hgbm_learning_rate", 1e-3, 0.3, log=True)
        max_depth = trial.suggest_int("hgbm_max_depth", 3, 16)
        max_iter = trial.suggest_int("hgbm_max_iter", 100, 1000, step=100)
        l2 = trial.suggest_float("hgbm_l2_regularization", 0.0, 1.0)
        max_leaf_nodes = trial.suggest_int("hgbm_max_leaf_nodes", 15, 255)
        est = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            l2_regularization=l2,
            max_leaf_nodes=max_leaf_nodes,
            random_state=RANDOM_STATE
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
