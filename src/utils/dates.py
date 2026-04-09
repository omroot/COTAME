"""Date utilities for business day calculations and exchange holiday lookups."""

import datetime
import calendar
from typing import Optional

import numpy as np
import pandas as pd
from pandas_market_calendars import get_calendar


def get_time_of_day_as_float(timestamp: datetime.datetime) -> float:
    """Convert a datetime's time component to a fractional hour.

    Parameters
    ----------
    timestamp : datetime.datetime
        The datetime to convert.

    Returns
    -------
    float
        Time as hours (e.g., 14:30:00 -> 14.5).
    """
    return timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600


def get_first_of_next_month(any_date: datetime.date) -> datetime.date:
    """Return the first day of the month following the given date.

    Parameters
    ----------
    any_date : datetime.date
        Reference date.

    Returns
    -------
    datetime.date
        First day of the next month.
    """
    if any_date.month != 12:
        return datetime.date(any_date.year, any_date.month + 1, 1)
    return datetime.date(any_date.year + 1, 1, 1)


def get_last_day_of_month(any_date: datetime.date) -> datetime.date:
    """Return the last day of the month for the given date.

    Parameters
    ----------
    any_date : datetime.date
        Reference date.

    Returns
    -------
    datetime.date
        Last day of the same month.
    """
    next_month = any_date.replace(day=28) + datetime.timedelta(days=4)
    return next_month.replace(day=1) - datetime.timedelta(days=1)


def get_nth_business_day_of_month(
    year: int,
    month: int,
    n: int,
    business_days: list[datetime.date],
) -> Optional[datetime.date]:
    """Get the nth business day of a given month.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month (1-12).
    n : int
        Which business day to return (1-indexed).
    business_days : list[datetime.date]
        Pre-computed list of business days to filter from.

    Returns
    -------
    datetime.date or None
        The nth business day, or None if fewer than n business days exist.
    """
    month_start = datetime.date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    month_end = datetime.date(year, month, last_day)
    month_business_days = [day for day in business_days if month_start <= day <= month_end]
    return month_business_days[n - 1]


def count_business_days_series(
    start_dates: pd.Series,
    end_dates: pd.Series,
    business_days: pd.Series,
) -> pd.Series:
    """Count signed business days between each pair of start and end dates.

    Parameters
    ----------
    start_dates : pd.Series
        Series of start dates.
    end_dates : pd.Series
        Series of end dates.
    business_days : pd.Series
        Series of valid business dates to count from.

    Returns
    -------
    pd.Series
        Signed count of business days (positive if start < end, negative if start > end).
    """
    start_dates = pd.to_datetime(start_dates).dt.date
    end_dates = pd.to_datetime(end_dates).dt.date
    business_days = pd.to_datetime(business_days).dt.date
    business_days_set = set(business_days)

    results = []
    for start, end in zip(start_dates, end_dates):
        if start == end:
            results.append(0)
        else:
            lower_bound = min(start, end)
            upper_bound = max(start, end)
            count = sum(lower_bound < day <= upper_bound for day in business_days_set)
            if start > end:
                count = -count
            results.append(count)

    return pd.Series(results, index=start_dates.index)


CME_ENERGY_CALENDARS: dict[str, str] = {
    "CL": "CMEGlobex_CL",
    "XB": "CMEGlobex_RB",
    "HO": "CMEGlobex_HO",
    "QS": "CMEGlobex_Energy",
    "CO": "CMEGlobex_Energy",
}


def get_cme_holidays(
    ticker_symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[datetime.date]:
    """Get CME/NYMEX holidays for a given energy futures ticker.

    Parameters
    ----------
    ticker_symbol : str
        Exchange ticker symbol (CL, XB, HO, QS, CO).
    start_date : datetime.date
        Start of the date range.
    end_date : datetime.date
        End of the date range.

    Returns
    -------
    list[datetime.date]
        Holiday dates within the specified range.
    """
    calendar_name = CME_ENERGY_CALENDARS.get(ticker_symbol)
    if calendar_name is None:
        raise ValueError(
            f"Unknown ticker '{ticker_symbol}'. "
            f"Supported: {list(CME_ENERGY_CALENDARS)}"
        )
    return get_holidays(calendar_name, start_date, end_date)


def get_holidays(
    exchange_name: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[datetime.date]:
    """Get holidays for a specific exchange within a date range.

    Parameters
    ----------
    exchange_name : str
        Name of the exchange calendar (e.g., 'XNYS' for NYSE).
    start_date : datetime.date
        Start of the date range.
    end_date : datetime.date
        End of the date range.

    Returns
    -------
    list[datetime.date]
        Holiday dates within the specified range.
    """
    holidays = pd.to_datetime(
        pd.Series(get_calendar(exchange_name).holidays().holidays)
    ).dt.date
    return holidays[(holidays >= start_date) & (holidays <= end_date)].unique().tolist()


def get_nyse_business_dates(
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[datetime.date]:
    """Get NYSE business dates (excluding weekends and holidays) within a range.

    Parameters
    ----------
    start_date : datetime.date
        Start of the date range.
    end_date : datetime.date
        End of the date range.

    Returns
    -------
    list[datetime.date]
        Business dates within the specified range.
    """
    all_dates = pd.date_range(start_date, end_date, freq='D')
    weekend_mask = (all_dates.dayofweek == 5) | (all_dates.dayofweek == 6)
    holidays = get_holidays(exchange_name='NYSE', start_date=start_date, end_date=end_date)
    holiday_mask = all_dates.isin(holidays)
    non_business_mask = weekend_mask | holiday_mask
    business_dates = pd.to_datetime(pd.Series(list(all_dates[~non_business_mask]))).dt.date.tolist()
    return business_dates


if __name__ == "__main__":
    from datetime import date

    holidays = get_cme_holidays("CL", date(2026, 1, 1), date(2026, 12, 31))
    print(f"NYMEX CL holidays for 2026 ({len(holidays)} days):")
    for holiday in holidays:
        print(f"  {holiday} ({holiday.strftime('%A')})")
