import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme


def _fmt(val, decimals=4):
    """Format a float or return '—'."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def make_layout(app_data):
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
        html.H5("Feature Clusters (with Clustered MDA)"),
        html.Div(id="selection-clusters"),
        html.H5("Selected Features — SFI Scores (after purging)", className="mt-3"),
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
        cluster_mda = resp_data.get("cluster_mda", {})
        feature_sfi = resp_data.get("feature_sfi", {})

        summary = html.P(
            f"{n_clusters} feature clusters found, {len(selected_clusters)} selected: "
            f"{', '.join(selected_clusters)}"
        )

        # --- Clusters table with MDA (sortable) ---
        clusters = resp_data.get("feature_clusters", {})
        table_data = []
        for cname, members in clusters.items():
            is_selected = cname in selected_clusters
            mda_info = cluster_mda.get(cname, {})
            mda_mean = mda_info.get("mean")
            mda_std = mda_info.get("std")
            table_data.append({
                "Cluster": cname,
                "Features": "\n".join(members),
                "MDA Mean": mda_mean,
                "MDA SE": mda_std,
                "Selected": "Yes" if is_selected else "No",
            })

        cluster_table = dash_table.DataTable(
            columns=[
                {"name": "Cluster", "id": "Cluster"},
                {"name": "Features", "id": "Features", "presentation": "markdown"},
                {"name": "MDA Mean", "id": "MDA Mean", "type": "numeric",
                 "format": Format(precision=6, scheme=Scheme.fixed)},
                {"name": "MDA SE", "id": "MDA SE", "type": "numeric",
                 "format": Format(precision=6, scheme=Scheme.fixed)},
                {"name": "Selected", "id": "Selected"},
            ],
            data=table_data,
            sort_action="native",
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "fontSize": "13px",
                "whiteSpace": "pre-line",
            },
            style_header={"fontWeight": "bold"},
            style_data_conditional=[
                {
                    "if": {"filter_query": '{Selected} = "Yes"'},
                    "backgroundColor": "#d4edda",
                },
                {
                    "if": {"filter_query": '{Selected} = "No"'},
                    "backgroundColor": "#f8d7da",
                },
            ],
            page_size=20,
        )

        # --- Purged / Kept features with SFI scores ---
        purged = resp_data.get("purged_feature_clusters", {})

        purge_rows = []
        for cname in selected_clusters:
            purged_feats = purged.get(cname, [])
            all_members = clusters.get(cname, [])
            kept = [f for f in all_members if f not in purged_feats]
            sfi_data = feature_sfi.get(cname, {})

            def _feat_with_sfi(feat_name):
                sfi_info = sfi_data.get(feat_name)
                if sfi_info:
                    return f"{feat_name}  (SFI: {_fmt(sfi_info['mean'])} ± {_fmt(sfi_info['std'])})"
                return feat_name

            purged_items = [html.Li(_feat_with_sfi(pf)) for pf in purged_feats]
            kept_items = [html.Li(_feat_with_sfi(kf)) for kf in kept]

            purged_cell = (
                html.Ul(purged_items, style={"marginBottom": "0", "paddingLeft": "18px"})
                if purged_items else "—"
            )
            kept_cell = (
                html.Ul(kept_items, style={"marginBottom": "0", "paddingLeft": "18px"})
                if kept_items else "—"
            )

            purge_rows.append(
                html.Tr([
                    html.Td(cname, style={"verticalAlign": "top"}),
                    html.Td(purged_cell),
                    html.Td(kept_cell),
                ])
            )

        final_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Cluster"),
                html.Th("Retained Features (with SFI)"),
                html.Th("Purged Features (with SFI)"),
            ])),
            html.Tbody(purge_rows),
        ], bordered=True, size="sm")

        return summary, cluster_table, final_table
