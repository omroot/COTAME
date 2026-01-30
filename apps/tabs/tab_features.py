import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def make_layout(app_data):
    features = app_data.feature_definitions["features"]
    feature_names = [f["name"] for f in features]
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
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="features-heatmap", style={"height": "600px"}),
                width=7,
            ),
        ]),

        # --- Binned Scatter Plot section ---
        html.Hr(),
        html.H5("Binned Scatter Plot"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizon"),
                dcc.Dropdown(
                    id="scatter-horizon",
                    options=[
                        {"label": "Nowcast", "value": "nowcast"},
                        {"label": "Forecast", "value": "forecast"},
                    ],
                    value="nowcast",
                    clearable=False,
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Feature"),
                dcc.Dropdown(
                    id="scatter-feature",
                    options=[{"label": f, "value": f} for f in feature_names],
                    value=feature_names[0],
                    clearable=False,
                ),
            ], width=4),
            dbc.Col([
                dbc.Label("Response"),
                dcc.Dropdown(id="scatter-response", clearable=False),
            ], width=4),
            dbc.Col([
                dbc.Label("Bins"),
                dcc.Dropdown(
                    id="scatter-nbins",
                    options=[{"label": str(n), "value": n} for n in [5, 10, 15, 20]],
                    value=10,
                    clearable=False,
                ),
            ], width=2),
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="scatter-plot", style={"height": "550px"}),
                width=7,
            ),
        ]),
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

    # --- Binned scatter callbacks ---

    @app.callback(
        Output("scatter-response", "options"),
        Output("scatter-response", "value"),
        Input("scatter-horizon", "value"),
    )
    def update_scatter_responses(horizon):
        corr_data = app_data.correlations(horizon)["correlations"]
        responses = [k for k in corr_data if k != "Feature"]
        options = [{"label": r, "value": r} for r in responses]
        return options, responses[0] if responses else None

    @app.callback(
        Output("scatter-plot", "figure"),
        Input("scatter-horizon", "value"),
        Input("scatter-feature", "value"),
        Input("scatter-response", "value"),
        Input("scatter-nbins", "value"),
    )
    def update_scatter(horizon, feature, response, nbins):
        fig = go.Figure()
        if not feature or not response:
            return fig

        df = app_data.dataset
        if feature not in df.columns or response not in df.columns:
            fig.update_layout(title="Column not found in dataset")
            return fig

        sub = df[[feature, response]].dropna()
        if len(sub) < nbins:
            fig.update_layout(title="Not enough data points")
            return fig

        # Quantile-based binning
        sub = sub.copy()
        sub["bin"] = pd.qcut(sub[feature], q=nbins, duplicates="drop")
        grouped = sub.groupby("bin", observed=True).agg(
            feat_mean=(feature, "mean"),
            resp_mean=(response, "mean"),
            resp_std=(response, "std"),
            count=(response, "count"),
        ).sort_values("feat_mean").reset_index()

        # Scatter of bin means with error bars
        fig.add_trace(go.Scatter(
            x=grouped["feat_mean"],
            y=grouped["resp_mean"],
            error_y=dict(
                type="data",
                array=(grouped["resp_std"] / np.sqrt(grouped["count"])).values,
                visible=True,
            ),
            mode="markers+lines",
            marker=dict(size=10, color="#0d6efd"),
            line=dict(color="#0d6efd", width=1, dash="dot"),
            name="Bin mean",
            hovertemplate=(
                "Feature mean: %{x:.4f}<br>"
                "Response mean: %{y:.4f}<br>"
                "n=%{customdata}<extra></extra>"
            ),
            customdata=grouped["count"],
        ))

        # OLS trend line across bin means
        x_vals = grouped["feat_mean"].values
        y_vals = grouped["resp_mean"].values
        if len(x_vals) >= 2:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=intercept + slope * x_line,
                mode="lines",
                line=dict(color="red", width=2),
                name=f"Trend (slope={slope:.4f})",
            ))

        # Spearman correlation from cached data
        corr_data = app_data.correlations(horizon)["correlations"]
        feat_list = corr_data["Feature"]
        rho = None
        if feature in feat_list and response in corr_data:
            idx = feat_list.index(feature)
            rho = corr_data[response][idx]

        title = f"{feature}  vs  {response}"
        if rho is not None:
            title += f"  (ρ = {rho:.3f})"

        fig.update_layout(
            title=title,
            xaxis_title=feature,
            yaxis_title=response,
            margin=dict(l=80),
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )
        return fig
