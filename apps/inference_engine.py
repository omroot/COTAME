import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor


MODEL_CONSTRUCTORS = {
    "lasso": lambda p: Lasso(**p),
    "ridge": lambda p: Ridge(**p),
    "rf": lambda p: RandomForestRegressor(random_state=42, **p),
    "extra": lambda p: ExtraTreesRegressor(random_state=42, **p),
    "hgbm": lambda p: HistGradientBoostingRegressor(random_state=42, **p),
    "ols": lambda _: LinearRegression(),
    "enet": lambda p: SGDRegressor(random_state=42, **p),
}


class InferenceEngine:
    def __init__(self, dataset, selected_features_by_horizon, selected_models_by_horizon):
        """
        Parameters
        ----------
        dataset : pd.DataFrame
        selected_features_by_horizon : dict  {horizon: {response: [features]}}
        selected_models_by_horizon : dict    {horizon: {response: {model_name, params}} or None}
        """
        self.estimators = {}  # (horizon, response) -> fitted estimator
        self.feature_lists = {}  # (horizon, response) -> [feature_names]

        for horizon in ("nowcast", "forecast"):
            feat_map = selected_features_by_horizon.get(horizon)
            model_map = selected_models_by_horizon.get(horizon)
            if feat_map is None or model_map is None:
                continue
            for response, features in feat_map.items():
                if response not in model_map:
                    continue
                model_info = model_map[response]
                model_name = model_info["model_name"]
                params = model_info["params"]

                # Prepare data
                cols = features + [response]
                sub = dataset[cols].dropna()
                if len(sub) < 10:
                    continue
                X = sub[features].values
                y = sub[response].values

                # Fit
                est = MODEL_CONSTRUCTORS[model_name](params)
                est.fit(X, y)

                self.estimators[(horizon, response)] = est
                self.feature_lists[(horizon, response)] = features

    def predict(self, horizon, features_dict):
        """Return {response: predicted_value} for the given horizon."""
        results = {}
        for (h, response), est in self.estimators.items():
            if h != horizon:
                continue
            feat_names = self.feature_lists[(h, response)]
            try:
                X = np.array([[features_dict[fn] for fn in feat_names]])
                pred = est.predict(X)[0]
                results[response] = pred
            except (KeyError, ValueError):
                results[response] = None
        return results

    def get_model_info(self, horizon, response):
        """Return (model_name, feature_list) or None."""
        key = (horizon, response)
        if key not in self.estimators:
            return None
        est = self.estimators[key]
        return type(est).__name__, self.feature_lists[key]
