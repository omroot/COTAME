import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import plotly.graph_objects as go
import numpy as np


def make_layout(app_data):
    layout = dbc.Container([
        html.H4("SHAP Values & Model Interpretability"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizon"),
                dcc.Dropdown(
                    id="shap-horizon",
                    options=[
                        {"label": "Nowcast", "value": "nowcast"},
                        {"label": "Forecast", "value": "forecast"},
                    ],
                    value="nowcast",
                    clearable=False,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Response"),
                dcc.Dropdown(id="shap-response", clearable=False),
            ], width=5),
        ]),
        html.Br(),
        dbc.Card(dbc.CardBody(id="shap-model-info"), className="mb-3"),
        html.Div(id="shap-no-data", style={"display": "none"}),
        html.H5("Mean |SHAP| per Feature"),
        dcc.Graph(id="shap-bar"),
        html.H5("SHAP Value Distribution"),
        dcc.Graph(id="shap-strip", style={"height": "500px"}),
    ], fluid=True, className="mt-3")
    return layout


def register_callbacks(app, app_data):
    @app.callback(
        Output("shap-response", "options"),
        Output("shap-response", "value"),
        Input("shap-horizon", "value"),
    )
    def update_response_options(horizon):
        shap_data = app_data.shap_values(horizon)
        if shap_data is None:
            return [], None
        responses = list(shap_data["responses"].keys())
        options = [{"label": r, "value": r} for r in responses]
        return options, responses[0] if responses else None

    @app.callback(
        Output("shap-model-info", "children"),
        Output("shap-no-data", "style"),
        Output("shap-no-data", "children"),
        Output("shap-bar", "figure"),
        Output("shap-strip", "figure"),
        Input("shap-horizon", "value"),
        Input("shap-response", "value"),
    )
    def update_shap(horizon, response):
        empty_fig = go.Figure()
        shap_data = app_data.shap_values(horizon)
        if shap_data is None:
            msg = dbc.Alert("No SHAP data available for this horizon.", color="warning")
            return "", {"display": "block"}, msg, empty_fig, empty_fig
        if not response or response not in shap_data["responses"]:
            return "", {"display": "none"}, "", empty_fig, empty_fig

        resp = shap_data["responses"][response]
        model_name = resp["model_name"]
        feature_names = resp["feature_names"]
        shap_vals = np.array(resp["shap_values"])  # (n_samples, n_features)

        # Model info card
        models = app_data.selected_models(horizon)
        params = models.get(response, {}).get("params", {}) if models else {}
        info = html.Div([
            html.Strong(f"Model: {model_name}"),
            html.Br(),
            html.Span(f"Parameters: {params}"),
        ])

        # Mean |SHAP| bar chart (sorted descending)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        sort_idx = np.argsort(mean_abs)[::-1]
        sorted_names = [feature_names[i] for i in sort_idx]
        sorted_vals = mean_abs[sort_idx]

        bar_fig = go.Figure(data=go.Bar(
            y=sorted_names,
            x=sorted_vals,
            orientation="h",
            marker_color="#0d6efd",
        ))
        bar_fig.update_layout(
            title="Mean |SHAP| Value",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=300),
        )

        # Strip/beeswarm plot
        strip_fig = go.Figure()
        # Show features from bottom (least important) to top (most important)
        for idx in reversed(sort_idx):
            fname = feature_names[idx]
            vals = shap_vals[:, idx]
            strip_fig.add_trace(go.Box(
                x=vals,
                name=fname,
                orientation="h",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                marker=dict(size=3, opacity=0.5),
                line=dict(width=0),
                fillcolor="rgba(0,0,0,0)",
            ))

        strip_fig.update_layout(
            title="SHAP Value Distribution by Feature",
            xaxis_title="SHAP Value",
            showlegend=False,
            margin=dict(l=300),
        )

        return info, {"display": "none"}, "", bar_fig, strip_fig
