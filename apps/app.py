import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from apps.data_loader import AppData
from apps.inference_engine import InferenceEngine
from apps.tabs import tab_features, tab_selection, tab_models, tab_shap, tab_inference

# Load data
app_data = AppData()

# Build inference engine (nowcast only — forecast has no models yet)
engine = InferenceEngine(
    dataset=app_data.dataset,
    selected_features_by_horizon={
        "nowcast": app_data.nowcast_selected_features,
        "forecast": app_data.forecast_selected_features,
    },
    selected_models_by_horizon={
        "nowcast": app_data.nowcast_selected_models,
        "forecast": app_data.forecast_selected_models,
    },
)

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
app.title = "WTI / MM — COT Dashboard"

app.layout = dbc.Container([
    html.H2("WTI Managed Money — COT Analysis Dashboard", className="mt-3 mb-3"),
    dbc.Tabs([
        dbc.Tab(tab_features.make_layout(app_data), label="Features & Correlations", tab_id="tab-features"),
        dbc.Tab(tab_selection.make_layout(app_data), label="Feature Selection", tab_id="tab-selection"),
        dbc.Tab(tab_models.make_layout(app_data), label="Model Selection", tab_id="tab-models"),
        dbc.Tab(tab_shap.make_layout(app_data), label="SHAP Values", tab_id="tab-shap"),
        dbc.Tab(tab_inference.make_layout(app_data), label="Live Inference", tab_id="tab-inference"),
    ], id="main-tabs", active_tab="tab-features"),
], fluid=True)

# Register callbacks
tab_features.register_callbacks(app, app_data)
tab_selection.register_callbacks(app, app_data)
tab_models.register_callbacks(app, app_data)
tab_shap.register_callbacks(app, app_data)
tab_inference.register_callbacks(app, app_data, engine)

if __name__ == "__main__":
    app.run(debug=True)
