import json
import pathlib
import pandas as pd


BASE = pathlib.Path(__file__).resolve().parent.parent
CACHE_OUTPUT = BASE / "cache" / "output" / "wti" / "mm"
DATASET_PATH = BASE / "cache" / "preprocessed_data" / "wti_dataset.csv"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class AppData:
    def __init__(self):
        self.feature_definitions = _load_json(CACHE_OUTPUT / "feature_definitions.json")

        # Nowcast artifacts
        nc = CACHE_OUTPUT / "nowcast"
        self.nowcast_correlations = _load_json(nc / "01_feature_response_correlations.json")
        self.nowcast_selection_details = _load_json(nc / "02_feature_selection_details.json")
        self.nowcast_selected_features = _load_json(nc / "02_selected_features_by_response.json")
        self.nowcast_cv_scores = _load_json(nc / "03_model_selection_cv_scores.json")
        self.nowcast_selected_models = _load_json(nc / "03_selected_model_by_response.json")
        self.nowcast_shap = _load_json(nc / "04_shap_values.json")

        # Forecast artifacts
        fc = CACHE_OUTPUT / "forecast"
        self.forecast_correlations = _load_json(fc / "01_feature_response_correlations.json")
        self.forecast_selection_details = _load_json(fc / "02_feature_selection_details.json")
        self.forecast_selected_features = _load_json(fc / "02_selected_features_by_response.json")
        self.forecast_cv_scores = _load_json(fc / "03_model_selection_cv_scores.json")
        self.forecast_selected_models = _load_json(fc / "03_selected_model_by_response.json")
        self.forecast_shap = _load_json(fc / "04_shap_values.json")

        # Dataset
        self.dataset = pd.read_csv(DATASET_PATH, parse_dates=["tradeDate"])

    # Convenience accessors by horizon
    def correlations(self, horizon):
        return self.nowcast_correlations if horizon == "nowcast" else self.forecast_correlations

    def selection_details(self, horizon):
        return self.nowcast_selection_details if horizon == "nowcast" else self.forecast_selection_details

    def selected_features(self, horizon):
        return self.nowcast_selected_features if horizon == "nowcast" else self.forecast_selected_features

    def cv_scores(self, horizon):
        return self.nowcast_cv_scores if horizon == "nowcast" else self.forecast_cv_scores

    def selected_models(self, horizon):
        return self.nowcast_selected_models if horizon == "nowcast" else self.forecast_selected_models

    def shap_values(self, horizon):
        return self.nowcast_shap if horizon == "nowcast" else self.forecast_shap

    def get_last_n_rows(self, n=3):
        return self.dataset.tail(n).reset_index(drop=True)
