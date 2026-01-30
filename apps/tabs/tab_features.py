import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html
import plotly.graph_objects as go
import pandas as pd


def make_layout(app_data):
    features = app_data.feature_definitions["features"]
    table_data = [
        {"#": i + 1, "Name": f["name"], "Description": f["description"], "Category": f["category"]}
        for i, f in enumerate(features)
    ]

    layout = dbc.Container([
        html.H4("Feature Definitions & Correlations"),
        html.Hr(),
        html.H5("All 20 Features"),
        dash_table.DataTable(
            id="features-table",
            columns=[
                {"name": "#", "id": "#"},
                {"name": "Name", "id": "Name"},
                {"name": "Description", "id": "Description"},
                {"name": "Category", "id": "Category"},
            ],
            data=table_data,
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
            ],
            page_size=20,
        ),
        html.Hr(),
        html.H5("Spearman Correlations: Features vs Responses"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizon"),
                dcc.Dropdown(
                    id="features-horizon",
                    options=[
                        {"label": "Nowcast", "value": "nowcast"},
                        {"label": "Forecast", "value": "forecast"},
                    ],
                    value="nowcast",
                    clearable=False,
                ),
            ], width=3),
        ]),
        dcc.Graph(id="features-heatmap", style={"height": "600px"}),
    ], fluid=True, className="mt-3")
    return layout


def register_callbacks(app, app_data):
    @app.callback(
        Output("features-heatmap", "figure"),
        Input("features-horizon", "value"),
    )
    def update_heatmap(horizon):
        corr_data = app_data.correlations(horizon)["correlations"]
        feature_names = corr_data["Feature"]
        responses = [k for k in corr_data if k != "Feature"]

        z = []
        for feat_idx in range(len(feature_names)):
            row = [corr_data[r][feat_idx] for r in responses]
            z.append(row)

        # Shorten response names for display
        short_responses = [r.replace("ManagedMoney_", "MM_").replace("_change", "Δ")
                           .replace("_to_openinterest", "/OI")
                           .replace("prior_report_", "pr_").replace("forward_report_", "fw_")
                           .replace("forward_", "fw_")
                           for r in responses]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=short_responses,
            y=feature_names,
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:.3f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(
            title=f"Spearman Correlations — {horizon.title()}",
            xaxis_title="Response",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=350),
        )
        return fig
