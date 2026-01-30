import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html
import plotly.graph_objects as go
import json


def make_layout(app_data):
    layout = dbc.Container([
        html.H4("Model Selection & CV Scores"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizon"),
                dcc.Dropdown(
                    id="models-horizon",
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
                dcc.Dropdown(id="models-response", clearable=False),
            ], width=5),
        ]),
        html.Br(),
        html.Div(id="models-no-data", style={"display": "none"}),
        dcc.Graph(id="models-bar-chart"),
        html.H5("Best Hyperparameters per Model"),
        html.Div(id="models-params-table"),
        html.Br(),
        dbc.Accordion([
            dbc.AccordionItem(
                html.Div(id="models-optuna-table"),
                title="Optuna Trial History",
            )
        ], start_collapsed=True),
    ], fluid=True, className="mt-3")
    return layout


def register_callbacks(app, app_data):
    @app.callback(
        Output("models-response", "options"),
        Output("models-response", "value"),
        Input("models-horizon", "value"),
    )
    def update_response_options(horizon):
        cv = app_data.cv_scores(horizon)
        if cv is None:
            return [], None
        responses = list(cv["responses"].keys())
        options = [{"label": r, "value": r} for r in responses]
        return options, responses[0] if responses else None

    @app.callback(
        Output("models-no-data", "style"),
        Output("models-no-data", "children"),
        Output("models-bar-chart", "figure"),
        Output("models-params-table", "children"),
        Output("models-optuna-table", "children"),
        Input("models-horizon", "value"),
        Input("models-response", "value"),
    )
    def update_models(horizon, response):
        empty_fig = go.Figure()
        cv = app_data.cv_scores(horizon)
        if cv is None:
            msg = dbc.Alert("No model selection data available for this horizon.", color="warning")
            return {"display": "block"}, msg, empty_fig, "", ""
        if not response or response not in cv["responses"]:
            return {"display": "none"}, "", empty_fig, "", ""

        resp_data = cv["responses"][response]
        cpcv = resp_data["cpcv_correlation_by_model"]
        selected_model = resp_data["selected_model"]

        # Bar chart
        models = list(cpcv.keys())
        scores = list(cpcv.values())
        colors = ["#28a745" if m == selected_model else "#6c757d" for m in models]

        fig = go.Figure(data=go.Bar(
            x=models,
            y=scores,
            marker_color=colors,
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"CPCV Correlation by Model — Winner: {selected_model}",
            xaxis_title="Model",
            yaxis_title="CPCV Correlation",
        )

        # Params table
        best = resp_data["best_per_model_params"]
        param_rows = []
        for model_name, info in best.items():
            param_rows.append(
                html.Tr([
                    html.Td(model_name, style={"fontWeight": "bold"} if model_name == selected_model else {}),
                    html.Td(f"{info['score']:.4f}"),
                    html.Td(json.dumps(info["params"], indent=0) if info["params"] else "—"),
                ])
            )

        params_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Model"), html.Th("Best Score"), html.Th("Hyperparameters")])),
            html.Tbody(param_rows),
        ], bordered=True, striped=True, size="sm")

        # Optuna trials
        trials = resp_data.get("optuna_trial_history", [])
        trial_rows = [
            html.Tr([
                html.Td(t["number"]),
                html.Td(t["model"]),
                html.Td(f"{t['score']:.4f}"),
                html.Td(json.dumps(t["params"], indent=0)),
            ])
            for t in trials[:50]  # limit display
        ]
        optuna_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Trial #"), html.Th("Model"), html.Th("Score"), html.Th("Params")])),
            html.Tbody(trial_rows),
        ], bordered=True, striped=True, size="sm")

        return {"display": "none"}, "", fig, params_table, optuna_table
