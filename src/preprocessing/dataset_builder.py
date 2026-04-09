"""
Weekly dataset builder for ML-based COT nowcasting.

WHY Monday-close features?
--------------------------
The COT report measures positioning as-of Tuesday, but we run the model
Tuesday morning (or Monday after close) — before Tuesday's market data
is available.  Features must reflect only information known at prediction
time, i.e. through Monday close.

For each COT Tuesday, market features (prices, volume, OI, spreads) are
taken from the most recent business day BEFORE Tuesday (typically Monday).
COT-derived features (prior report changes) are lagged by construction
and are safe — they reflect the previous week's report, released the
prior Friday.
"""

import pandas as pd


COT_POSITION_FIELDS = [
    'Commercial_NetPosition',
    'CommercialLongPosition',
    'CommercialShortPosition',
    'ManagedMoney_NetPosition',
    'ManagedMoney_LongPosition',
    'ManagedMoney_ShortPosition',
]


class WeeklyDataSetBuilder:

    def __init__(self) -> None:
        self.data = pd.DataFrame()

    def fit(self,
            cot_db: pd.DataFrame,
            synthetic_spread_db: pd.DataFrame,
            volume_db: pd.DataFrame,
            openinterest_db: pd.DataFrame,
            ) -> None:

        # ── COT spine: one row per Tuesday (as-of date) ──────────────
        cot_db = cot_db.copy()
        cot_db['tradeDate'] = pd.to_datetime(cot_db['tradeDate'])
        cot_db.sort_values('tradeDate', inplace=True)

        # ── Compute the feature cutoff date for each COT Tuesday ─────
        # WHY: we run the model before Tuesday's market data exists.
        # The latest data we have is Monday close (or the prior business
        # day if Monday is a holiday).  We use merge_asof to find the
        # most recent daily row strictly before each Tuesday.
        cot_db['feature_date'] = cot_db['tradeDate'] - pd.Timedelta(days=1)
        # feature_date is tentatively Monday; merge_asof will snap to
        # the nearest prior business day if Monday is missing.

        # ── Join daily price features (as of Monday close) ───────────
        spread_db = synthetic_spread_db.copy()
        spread_db['SyntheticF1MinusF2_RolledPrice'] = (
            spread_db['F1_RolledPrice'] - spread_db['beta_ols_250'] * spread_db['F2_RolledPrice']
        )
        spread_db['tradeDate'] = pd.to_datetime(spread_db['tradeDate'])
        spread_db.sort_values('tradeDate', inplace=True)

        price_cols = ['F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice',
                      'F1_RolledPrice_rolling_20D_volatility',
                      'F2_RolledPrice_rolling_20D_volatility',
                      'F3_RolledPrice_rolling_20D_volatility',
                      'SyntheticF1MinusF2_RolledPrice']

        dataset = pd.merge_asof(
            cot_db[['tradeDate', 'feature_date'] + COT_POSITION_FIELDS
                   + [c for c in cot_db.columns if c.endswith('_change') or c == 'release_date']],
            spread_db[['tradeDate'] + price_cols].rename(columns={'tradeDate': 'feature_date'}),
            on='feature_date',
            direction='backward',
        )

        # ── Join daily volume features (as of Monday close) ──────────
        vol_db = volume_db.copy()
        vol_db['tradeDate'] = pd.to_datetime(vol_db['tradeDate'])
        vol_db.sort_values('tradeDate', inplace=True)

        volume_cols = ['prior_cumulative_5D_F1_Volume', 'prior_cumulative_5D_F2_Volume']
        volume_cols = [c for c in volume_cols if c in vol_db.columns]

        dataset = pd.merge_asof(
            dataset,
            vol_db[['tradeDate'] + volume_cols].rename(columns={'tradeDate': 'feature_date'}),
            on='feature_date',
            direction='backward',
        )
        if 'prior_cumulative_5D_F1_Volume' in dataset.columns and 'prior_cumulative_5D_F2_Volume' in dataset.columns:
            dataset['prior_cumulative_5D_F1MinusF2_Volume'] = (
                dataset['prior_cumulative_5D_F1_Volume'] - dataset['prior_cumulative_5D_F2_Volume']
            )

        # ── Join daily OI features (as of Monday close) ──────────────
        oi_db = openinterest_db.copy()
        oi_db['tradeDate'] = pd.to_datetime(oi_db['tradeDate'])
        oi_db.sort_values('tradeDate', inplace=True)

        oi_cols = ['F1_OI', 'F2_OI', 'F3_OI', 'AGG_OI',
                   'prior_5D_F1_OI_change', 'prior_5D_F2_OI_change', 'prior_5D_AGG_OI_change']
        oi_cols = [c for c in oi_cols if c in oi_db.columns]

        dataset = pd.merge_asof(
            dataset,
            oi_db[['tradeDate'] + oi_cols].rename(columns={'tradeDate': 'feature_date'}),
            on='feature_date',
            direction='backward',
        )
        if 'prior_5D_F1_OI_change' in dataset.columns and 'prior_5D_F2_OI_change' in dataset.columns:
            dataset['prior_5D_F1MinusF2_openinterest_change'] = (
                dataset['prior_5D_F1_OI_change'] - dataset['prior_5D_F2_OI_change']
            )

        # ── COT / OI ratios ──────────────────────────────────────────
        # WHY use Monday's AGG_OI for the ratio?  Because that's the
        # latest OI we have.  The COT level is last Tuesday's (known
        # since Friday), so COT/OI uses last-known COT and Monday's OI.
        for f in COT_POSITION_FIELDS:
            dataset[f'{f}_to_openinterest'] = dataset[f] / dataset['AGG_OI']

        # ── Week-over-week changes (COT-derived, safe) ───────────────
        # These are computed on the weekly COT grid.  "prior_report_*"
        # is shift(1) = the PREVIOUS week's change, which was released
        # two Fridays ago — no lookahead risk.
        dataset.sort_values('tradeDate', inplace=True)

        for f in ['Commercial_NetPosition_to_openinterest',
                   'CommercialLongPosition_to_openinterest',
                   'CommercialShortPosition_to_openinterest',
                   'ManagedMoney_NetPosition_to_openinterest',
                   'ManagedMoney_LongPosition_to_openinterest',
                   'ManagedMoney_ShortPosition_to_openinterest']:
            dataset[f'{f}_change'] = dataset[f] - dataset[f].shift(1)
            dataset[f'prior_report_{f}_change'] = dataset[f'{f}_change'].shift(1)
            dataset[f'forward_{f}_change'] = dataset[f].shift(-1) - dataset[f]

        # ── Price/volume/spread changes (Monday-to-Monday) ───────────
        # These are week-over-week changes of the Monday-close values.
        for name in ['prior_cumulative_5D_F1_Volume',
                     'prior_cumulative_5D_F2_Volume',
                     'prior_cumulative_5D_F1MinusF2_Volume',
                     'F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice',
                     'SyntheticF1MinusF2_RolledPrice']:
            if name in dataset.columns:
                dataset[f'{name}_change'] = dataset[name] - dataset[name].shift(1)
                dataset[f'next_{name}_change'] = dataset[name].shift(-1) - dataset[name]

        self.data = dataset
