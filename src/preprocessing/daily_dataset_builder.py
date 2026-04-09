"""
Daily dataset builder for Kalman-Filter-based COT nowcasting.

WHY a daily dataset?
--------------------
COT positioning is reported weekly: as-of Tuesday, released Friday.
A nowcasting model needs daily market data to estimate positioning
between releases.  This builder assembles a daily-frequency dataset
where:

- Every row is a business day.
- Daily market features (price, volume, OI, spreads) are available
  on every row — these drive the KF state transition.
- COT observations appear ONLY on Fridays (the release date),
  not on Tuesdays (the as-of date) — because you can't use
  information before it's released.  The KF correction step
  fires on these Fridays.
- COT-derived features (last-known level, last-known change) use
  only released data: they jump on Fridays and stay flat until
  the next Friday.  This avoids lookahead bias.
"""

import pandas as pd

# The six COT positioning columns we care about.
COT_FIELDS = [
    'Commercial_NetPosition',
    'CommercialLongPosition',
    'CommercialShortPosition',
    'ManagedMoney_NetPosition',
    'ManagedMoney_LongPosition',
    'ManagedMoney_ShortPosition',
]


class DailyDataSetBuilder:
    """Assemble a daily-frequency dataset for COT nowcasting.

    Expects ``cot_db`` to have a ``release_date`` column (computed by
    ``COTPanel.fit``).  All COT-related columns are keyed on that date.

    Output columns:
        tradeDate               — business day
        day_of_week             — 0=Mon … 4=Fri
        <daily market features> — prices, returns, volume, OI, spreads
        cot_obs_<field>         — COT levels, only on Fridays (NaN otherwise)
        last_known_<field>      — most recent released COT level (ffill from Fridays)
        last_known_<field>_change — most recent week-over-week COT change (ffill)
    """

    def __init__(self) -> None:
        self.data = pd.DataFrame()

    def fit(
        self,
        cot_db: pd.DataFrame,
        price_panel: pd.DataFrame,
        volume_panel: pd.DataFrame,
        openinterest_panel: pd.DataFrame,
        synthetic_spread_db: pd.DataFrame,
    ) -> None:

        # ── Step 1: Daily spine from price panel ──────────────────────
        # WHY price panel? It has a row for every business day and
        # defines the calendar for the entire dataset.
        daily = price_panel[['tradeDate']].drop_duplicates().copy()
        daily['tradeDate'] = pd.to_datetime(daily['tradeDate'])
        daily.sort_values('tradeDate', inplace=True)
        daily.reset_index(drop=True, inplace=True)
        daily['day_of_week'] = daily['tradeDate'].dt.dayofweek

        # ── Step 2: Join raw daily panels ─────────────────────────────
        # WHY left join on daily spine? Each panel may have slightly
        # different date coverage.  The spine keeps all business days;
        # missing panel data becomes NaN (handled by the KF).
        for panel, cols in [
            (price_panel,        ['F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice',
                                  'F1MinusF2_RolledPrice',
                                  'F1_RolledPrice_rolling_20D_volatility']),
            (volume_panel,       ['F1_Volume', 'F2_Volume', 'F3_Volume']),
            (openinterest_panel, ['F1_OI', 'F2_OI', 'F3_OI', 'AGG_OI']),
        ]:
            p = panel[['tradeDate'] + [c for c in cols if c in panel.columns]].copy()
            p['tradeDate'] = pd.to_datetime(p['tradeDate'])
            daily = daily.merge(p, on='tradeDate', how='left')

        # Synthetic spread: hedge-ratio-adjusted front-back spread
        if 'beta_ols_250' in synthetic_spread_db.columns:
            sp = synthetic_spread_db[['tradeDate', 'beta_ols_250']].copy()
            sp['tradeDate'] = pd.to_datetime(sp['tradeDate'])
            daily = daily.merge(sp, on='tradeDate', how='left')

        # ── Step 3: Compute daily-frequency derived features ──────────
        # WHY compute these here and not in the panels?  The panels
        # produce weekly-aligned features (prior_5D_*).  Here we need
        # true daily changes for the KF's daily state transition.
        daily.sort_values('tradeDate', inplace=True)

        for col in ['F1_RolledPrice', 'F2_RolledPrice', 'F3_RolledPrice',
                     'F1MinusF2_RolledPrice']:
            if col in daily.columns:
                daily[f'{col}_daily_return'] = daily[col].pct_change()

        for col in ['F1_OI', 'F2_OI', 'AGG_OI']:
            if col in daily.columns:
                daily[f'{col}_daily_change'] = daily[col].diff()

        if 'beta_ols_250' in daily.columns:
            daily['SyntheticSpread'] = (
                daily['F1_RolledPrice'] - daily['beta_ols_250'] * daily['F2_RolledPrice']
            )
            daily['SyntheticSpread_daily_change'] = daily['SyntheticSpread'].diff()

        # ── Step 4: COT observations and last-known features ──────────
        # Both come from the same source: the COT panel keyed by
        # release_date.  We build one slice, merge once, then split
        # into observation columns (NaN on non-Fridays) and last-known
        # columns (forward-filled from Fridays).
        cot = cot_db[['release_date'] + COT_FIELDS].copy()
        cot['release_date'] = pd.to_datetime(cot['release_date'])
        cot.sort_values('release_date', inplace=True)

        # Week-over-week changes (on the weekly grid, before merging)
        for f in COT_FIELDS:
            cot[f'{f}_change'] = cot[f].diff()

        # Rename release_date → tradeDate for the merge
        cot = cot.rename(columns={'release_date': 'tradeDate'})
        daily = daily.merge(cot, on='tradeDate', how='left')

        # Observation columns: COT levels visible only on Fridays.
        # WHY Friday, not Tuesday?  The report is as-of Tuesday but
        # released on Friday.  The KF can only correct its state when
        # the data is actually available.  On Mon-Thu these columns
        # are NaN → the KF runs predict-only (no correction).
        for f in COT_FIELDS:
            daily[f'cot_obs_{f}'] = daily[f]

        # Last-known columns: most recent released value, forward-filled.
        # WHY a step function?  Between Fridays we don't have new COT
        # data.  The last-known value is the best we have.  It jumps
        # on Friday when a new report drops, stays flat all week.
        # This is the only safe way to use prior COT as a feature
        # without lookahead bias.
        for f in COT_FIELDS:
            daily[f'last_known_{f}'] = daily[f].ffill()
            daily[f'last_known_{f}_change'] = daily[f'{f}_change'].ffill()

        # Drop the raw COT columns (they duplicate cot_obs_* on Fridays
        # and are NaN elsewhere — no reason to keep both).
        daily.drop(columns=COT_FIELDS + [f'{f}_change' for f in COT_FIELDS], inplace=True)

        # ── Done ──────────────────────────────────────────────────────
        daily.sort_values('tradeDate', inplace=True)
        daily.reset_index(drop=True, inplace=True)
        self.data = daily
