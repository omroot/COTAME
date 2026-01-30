import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, callback_context
import numpy as np
import pandas as pd

from apps.feature_computer import compute_features


# Maps raw input ID suffixes to dataset column names for pre-filling
_PREFILL_MAP_T = {
    "MM_NetPos": "ManagedMoney_NetPosition",
    "MM_LongPos": "ManagedMoney_LongPosition",
    "MM_ShortPos": "ManagedMoney_ShortPosition",
    "AGG_OI": "AGG_OI",
    "F1_OI": "F1_OI",
    "F2_OI": "F2_OI",
    "F1_Price": "F1_RolledPrice",
    "F2_Price": "F2_RolledPrice",
    "F3_Price": "F3_RolledPrice",
    "F1_Vol20D": "F1_RolledPrice_rolling_20D_volatility",
    "F2_Vol20D": "F2_RolledPrice_rolling_20D_volatility",
    "F3_Vol20D": "F3_RolledPrice_rolling_20D_volatility",
    "Cum5D_F1_Vol": "prior_cumulative_5D_F1_Volume",
    "Cum5D_F2_Vol": "prior_cumulative_5D_F2_Volume",
}

# All raw input field definitions: (id_suffix, label, time_slots)
# time_slots is a list of suffixes like "_t", "_t1", "_t2"
RAW_FIELDS = [
    # Section 1: COT Positions
    ("MM_NetPos", "MM Net Position", ["_t", "_t1", "_t2"]),
    ("MM_LongPos", "MM Long Position", ["_t", "_t1", "_t2"]),
    ("MM_ShortPos", "MM Short Position", ["_t", "_t1", "_t2"]),
    # Section 2: Open Interest
    ("F1_OI", "F1 OI", ["_t", "_5d_ago"]),
    ("F2_OI", "F2 OI", ["_t", "_5d_ago"]),
    ("AGG_OI", "AGG OI", ["_t", "_t1", "_t2", "_5d_ago"]),
    # Section 3: Prices
    ("F1_Price", "F1 Rolled Price", ["_t", "_t1", "_t2"]),
    ("F2_Price", "F2 Rolled Price", ["_t", "_t1", "_t2"]),
    ("F3_Price", "F3 Rolled Price", ["_t", "_t1"]),
    ("F1_Vol20D", "F1 20D Volatility", ["_t"]),
    ("F2_Vol20D", "F2 20D Volatility", ["_t"]),
    ("F3_Vol20D", "F3 20D Volatility", ["_t"]),
    # Section 4: Volumes
    ("Cum5D_F1_Vol", "Cumul 5D F1 Volume", ["_t", "_t1"]),
    ("Cum5D_F2_Vol", "Cumul 5D F2 Volume", ["_t", "_t1"]),
]

SECTION_LABELS = {
    0: "COT Positions",
    3: "Open Interest",
    6: "Prices",
    12: "Volumes",
}


def _all_input_ids():
    """Return list of all raw input component IDs."""
    ids = []
    for suffix, _, slots in RAW_FIELDS:
        for slot in slots:
            ids.append(f"inf-{suffix}{slot}")
    return ids


def _get_prefill_values(app_data):
    """Extract default values from the last 3 rows of the dataset."""
    df = app_data.get_last_n_rows(3)
    # df row 0 = t-2, row 1 = t-1, row 2 = t (most recent)
    defaults = {}
    for suffix, _, slots in RAW_FIELDS:
        col = _PREFILL_MAP_T.get(suffix)
        if col is None:
            continue
        for slot in slots:
            if slot == "_t" and col in df.columns:
                defaults[f"inf-{suffix}_t"] = _safe_val(df, 2, col)
            elif slot == "_t1" and col in df.columns:
                defaults[f"inf-{suffix}_t1"] = _safe_val(df, 1, col)
            elif slot == "_t2" and col in df.columns:
                defaults[f"inf-{suffix}_t2"] = _safe_val(df, 0, col)
            elif slot == "_5d_ago" and col in df.columns:
                # Approximate: use 1 row back as proxy for "5 days ago"
                defaults[f"inf-{suffix}_5d_ago"] = _safe_val(df, 1, col)
    return defaults


def _safe_val(df, row_idx, col):
    try:
        v = df.iloc[row_idx][col]
        if pd.isna(v):
            return None
        return round(float(v), 4)
    except (IndexError, KeyError):
        return None


def make_layout(app_data):
    defaults = _get_prefill_values(app_data)

    sections = []
    current_section = None
    for i, (suffix, label, slots) in enumerate(RAW_FIELDS):
        if i in SECTION_LABELS:
            current_section = SECTION_LABELS[i]
            sections.append(html.H6(current_section, className="mt-3 mb-2 text-primary"))

        slot_labels = {"_t": "Week t", "_t1": "Week t-1", "_t2": "Week t-2", "_5d_ago": "5 days ago"}
        row_items = [dbc.Col(html.Strong(label), width=3)]
        for slot in slots:
            inp_id = f"inf-{suffix}{slot}"
            row_items.append(
                dbc.Col([
                    dbc.Label(slot_labels.get(slot, slot), size="sm"),
                    dbc.Input(
                        id=inp_id,
                        type="number",
                        value=defaults.get(inp_id),
                        size="sm",
                    ),
                ], width=2)
            )
        sections.append(dbc.Row(row_items, className="mb-1 align-items-end"))

    layout = dbc.Container([
        html.H4("Live Inference"),
        html.Hr(),
        html.H5("Raw Inputs (pre-filled from dataset)"),
        html.Div(sections),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Horizon"),
                dcc.RadioItems(
                    id="inf-horizon",
                    options=[
                        {"label": " Nowcast", "value": "nowcast"},
                        {"label": " Forecast", "value": "forecast"},
                    ],
                    value="nowcast",
                    inline=True,
                ),
            ], width=4),
            dbc.Col([
                dbc.Button("Predict", id="inf-predict-btn", color="primary", className="mt-3"),
            ], width=2),
        ]),
        html.Br(),
        html.Div(id="inf-results"),
        html.Br(),
        dbc.Accordion([
            dbc.AccordionItem(
                html.Div(id="inf-computed-features"),
                title="Computed Features (20)",
            )
        ], start_collapsed=True),
    ], fluid=True, className="mt-3")
    return layout


def register_callbacks(app, app_data, engine):
    all_ids = _all_input_ids()

    # Compute beta_ols from last 10 rows of dataset
    df = app_data.dataset.dropna(subset=["F1_RolledPrice", "F2_RolledPrice"]).tail(10)
    if len(df) >= 2:
        x = df["F2_RolledPrice"].values
        y = df["F1_RolledPrice"].values
        beta_ols = float(np.polyfit(x, y, 1)[0])
    else:
        beta_ols = 1.0

    @app.callback(
        Output("inf-results", "children"),
        Output("inf-computed-features", "children"),
        Input("inf-predict-btn", "n_clicks"),
        [State(cid, "value") for cid in all_ids],
        State("inf-horizon", "value"),
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, *args):
        horizon = args[-1]
        input_vals = args[:-1]

        # Build raw dict
        raw = {}
        for idx, cid in enumerate(all_ids):
            key = cid.replace("inf-", "")
            v = input_vals[idx]
            if v is None:
                return dbc.Alert(f"Missing input: {cid}", color="danger"), ""
            raw[key] = float(v)

        # Compute features
        try:
            features = compute_features(raw, beta_ols)
        except Exception as e:
            return dbc.Alert(f"Feature computation error: {e}", color="danger"), ""

        # Run inference
        if engine is None:
            return dbc.Alert("Inference engine not available (no fitted models for this horizon).", color="warning"), ""

        preds = engine.predict(horizon, features)
        if not preds:
            return dbc.Alert(f"No models available for horizon '{horizon}'.", color="warning"), ""

        # Results table
        rows = []
        for response, value in preds.items():
            model_info = engine.get_model_info(horizon, response)
            model_label = model_info[0] if model_info else "—"
            rows.append(html.Tr([
                html.Td(response),
                html.Td(f"{value:.4f}" if value is not None else "Error"),
                html.Td(model_label),
            ]))

        results = dbc.Table([
            html.Thead(html.Tr([html.Th("Response"), html.Th("Predicted Value"), html.Th("Model")])),
            html.Tbody(rows),
        ], bordered=True, striped=True, size="sm")

        # Computed features table
        feat_rows = [
            html.Tr([html.Td(k), html.Td(f"{v:.6f}")])
            for k, v in sorted(features.items())
        ]
        feat_table = dbc.Table([
            html.Thead(html.Tr([html.Th("Feature"), html.Th("Value")])),
            html.Tbody(feat_rows),
        ], bordered=True, striped=True, size="sm")

        return results, feat_table
