import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html
import json


def make_layout(app_data):
    # Get response options from nowcast
    nc_responses = list(app_data.nowcast_selection_details["responses"].keys())

    layout = dbc.Container([
        html.H4("Feature Selection Details"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizon"),
                dcc.Dropdown(
                    id="selection-horizon",
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
                dcc.Dropdown(id="selection-response", clearable=False),
            ], width=5),
        ]),
        html.Br(),
        dbc.Card(dbc.CardBody(id="selection-summary"), className="mb-3"),
        html.H5("Feature Clusters"),
        html.Div(id="selection-clusters"),
        html.H5("Selected Features (after purging)", className="mt-3"),
        html.Div(id="selection-final"),
    ], fluid=True, className="mt-3")
    return layout


def register_callbacks(app, app_data):
    @app.callback(
        Output("selection-response", "options"),
        Output("selection-response", "value"),
        Input("selection-horizon", "value"),
    )
    def update_response_options(horizon):
        details = app_data.selection_details(horizon)
        responses = list(details["responses"].keys())
        options = [{"label": r, "value": r} for r in responses]
        return options, responses[0] if responses else None

    @app.callback(
        Output("selection-summary", "children"),
        Output("selection-clusters", "children"),
        Output("selection-final", "children"),
        Input("selection-horizon", "value"),
        Input("selection-response", "value"),
    )
    def update_selection(horizon, response):
        if not response:
            return "Select a response", [], []

        details = app_data.selection_details(horizon)
        resp_data = details["responses"].get(response, {})

        n_clusters = resp_data.get("number_of_feature_clusters", 0)
        selected_clusters = resp_data.get("selected_cluster_names", [])
        summary = html.P(f"{n_clusters} feature clusters found, {len(selected_clusters)} selected: {', '.join(selected_clusters)}")

        # Clusters table
        clusters = resp_data.get("feature_clusters", {})
        cluster_rows = []
        for cname, members in clusters.items():
            is_selected = cname in selected_clusters
            style = {"backgroundColor": "#d4edda"} if is_selected else {}
            cluster_rows.append(
                html.Tr([
                    html.Td(cname),
                    html.Td(", ".join(members)),
                    html.Td("Yes" if is_selected else "No"),
                ], style=style)
            )

        cluster_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Cluster"), html.Th("Features"), html.Th("Selected")])),
            html.Tbody(cluster_rows),
        ], bordered=True, striped=True, size="sm")

        # Final selected features
        purged = resp_data.get("purged_feature_clusters", {})
        final_features = resp_data.get("selected_features", [])

        purge_rows = []
        for cname in selected_clusters:
            purged_feats = purged.get(cname, [])
            kept = [f for f in clusters.get(cname, []) if f not in purged_feats]
            purge_rows.append(
                html.Tr([
                    html.Td(cname),
                    html.Td(", ".join(purged_feats) if purged_feats else "—"),
                    html.Td(", ".join(kept)),
                ])
            )

        final_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Cluster"), html.Th("Purged Features"), html.Th("Kept Features")])),
            html.Tbody(purge_rows),
        ], bordered=True, striped=True, size="sm")

        return summary, cluster_table, final_table
