"""COT feature panel: maps as-of dates to release dates and computes position changes."""

import pandas as pd
from typing import Optional


_LEGACY_POSITION_COLUMNS = [
    'Commercial_NetPosition',
    'CommercialLongPosition',
    'CommercialShortPosition',
    'ManagedMoney_NetPosition',
    'ManagedMoney_LongPosition',
    'ManagedMoney_ShortPosition',
]

_FULL_POSITION_COLUMNS = _LEGACY_POSITION_COLUMNS + [
    'SwapDealers_NetPosition',
    'SwapDealers_LongPosition',
    'SwapDealers_ShortPosition',
    'OtherReportables_NetPosition',
    'OtherReportables_LongPosition',
    'OtherReportables_ShortPosition',
    'NonReportables_NetPosition',
    'NonReportables_LongPosition',
    'NonReportables_ShortPosition',
]


def _tuesday_to_friday(dates: pd.Series) -> pd.Series:
    """Map COT as-of dates (Tuesday) to release dates (Friday).

    CFTC publishes the report on Friday for positions as-of the
    preceding Tuesday. This shifts each date forward to the nearest
    Friday. Handles edge cases where the as-of date is not a Tuesday
    (e.g. holidays).
    """
    days_ahead = (4 - dates.dt.dayofweek) % 7   # 4 = Friday
    return dates + pd.to_timedelta(days_ahead, unit='D')


class COTPanel:
    """Builds a COT feature panel with week-over-week position changes.

    Automatically detects which position columns are present in the input
    and computes changes for all of them.
    """

    def __init__(self) -> None:
        self.panel: Optional[pd.DataFrame] = None

    def fit(self, dataset: pd.DataFrame) -> None:
        """Compute COT features: release date mapping and position changes.

        Parameters
        ----------
        dataset : pd.DataFrame
            Must contain 'tradeDate', 'Name', and position columns
            (at minimum the legacy Commercial + ManagedMoney columns).
        """
        dataset['tradeDate'] = pd.to_datetime(dataset['tradeDate'])
        dataset.sort_values(by='tradeDate', ascending=True, inplace=True)

        dataset['release_date'] = _tuesday_to_friday(dataset['tradeDate'])

        # Detect which position columns are present
        position_columns = [
            column for column in _FULL_POSITION_COLUMNS
            if column in dataset.columns
        ]

        for column_name in position_columns:
            dataset[f'{column_name}_change'] = (
                dataset[column_name] - dataset[column_name].shift(1)
            )
            dataset[f'prior_report_{column_name}_change'] = (
                dataset[f'{column_name}_change'].shift(1)
            )
            dataset[f'forward_report_{column_name}_change'] = (
                dataset[f'{column_name}_change'].shift(-1)
            )

        self.panel = dataset


if __name__ == '__main__':
    test_dates = pd.to_datetime(pd.Series([
        '2025-03-18',  # Tuesday  -> Friday 2025-03-21
        '2025-03-25',  # Tuesday  -> Friday 2025-03-28
        '2025-03-17',  # Monday   -> Friday 2025-03-21
        '2025-03-21',  # Friday   -> stays  Friday 2025-03-21
        '2025-03-19',  # Wednesday-> Friday 2025-03-21
    ]))
    result = _tuesday_to_friday(test_dates)

    print('_tuesday_to_friday mapping:')
    print('-' * 45)
    for original, mapped in zip(test_dates, result):
        print(f'  {original.strftime("%Y-%m-%d")} ({original.day_name()[:3]})'
              f'  ->  {mapped.strftime("%Y-%m-%d")} ({mapped.day_name()[:3]})')
