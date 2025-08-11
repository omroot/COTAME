
from typing import Optional
import datetime
import calendar
import numpy as np
import pandas as pd
from pandas_market_calendars import get_calendar



def get_timeOfDay_as_float(dt: datetime.datetime) -> float:
    """ Transform a datetime into a float """
    return dt.hour +dt.minute / 60 + dt.second / (60*60)

def get_first_of_next_month(anydate: datetime.date)->datetime.date:
    """ Returns the first day of the next month relative to the given day. """
    if anydate.month !=12:
        return datetime.date(anydate.year, anydate.month+1,1)
    return datetime.date(anydate.year +1,1,1)

def get_last_day_of_month(any_day: datetime.date) -> datetime.date:
    """Returns the last day of the month for the given date."""
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # safely in next month
    return next_month.replace(day=1) - datetime.timedelta(days=1)      # go back one day to last of current month


def get_nth_business_day_of_month(year: int,
                                  month: int,
                                  n: int,
                                 business_days: list[datetime.date]) -> Optional[datetime.date]:
    """Get the nth business day of a given month"""
    # Get first and last day of month
    month_start = datetime.date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    month_end = datetime.date(year, month, last_day)
    month_business_days = [d for d in business_days if month_start <= d <= month_end]
    return month_business_days[n-1]


# def count_business_days_series(start_dates: pd.Series,
#                                end_dates: pd.Series,
#                                business_days: pd.Series) -> pd.Series:
#     """
#     Count the number of business days (from a given list) between each pair of start_date and end_date.
#
#     Parameters:
#     - start_dates (pd.Series): Series of start dates (datetime64 or string).
#     - end_dates (pd.Series): Series of end dates (datetime64 or string).
#     - business_days (pd.Series): Series of valid business dates (datetime64).
#
#     Returns:
#     - pd.Series: Series of business day counts between each start and end date.
#     """
#     # Ensure datetime format
#     start_dates = pd.to_datetime(start_dates).dt.date
#     end_dates = pd.to_datetime(end_dates).dt.date
#     business_days = pd.to_datetime(business_days).date  # returns numpy array of datetime.date
#
#     # Sort business days for efficient search
#     business_days_sorted = sorted(business_days)
#
#     # Use searchsorted for efficient interval count
#     start_pos = pd.Series(pd.Index(business_days_sorted).searchsorted(start_dates, side='left'))
#     end_pos = pd.Series(pd.Index(business_days_sorted).searchsorted(end_dates, side='right'))
#
#     return end_pos - start_pos
#
#
#
#
#
# def count_business_days_series(start_dates: pd.Series,
#                                end_dates: pd.Series,
#                                business_days: pd.Series) -> pd.Series:
#     """
#     Count the number of business days (from a given list) between each pair of start_date and end_date.
#
#     Parameters:
#     - start_dates (pd.Series): Series of start dates.
#     - end_dates (pd.Series): Series of end dates.
#     - business_days (pd.Series): Series of valid business dates.
#
#     Returns:
#     - pd.Series: Series of business day counts between each start and end date.
#     """
#     # Convert all to datetime.date
#     start_dates = pd.to_datetime(start_dates).dt.date
#     end_dates = pd.to_datetime(end_dates).dt.date
#     business_days = pd.to_datetime(business_days).dt.date
#
#     # Create a set for fast lookup
#     business_days_set = set(business_days)
#
#     # Count business days for each (start, end) pair
#     results = []
#     for start, end in zip(start_dates, end_dates):
#         if start > end:
#             results.append(0)
#         else:
#             count = sum(start < day <= end for day in business_days_set)
#             results.append(count)
#
#     return pd.Series(results, index=start_dates.index)

import pandas as pd
from datetime import date

def count_business_days_series(start_dates: pd.Series,
                               end_dates: pd.Series,
                               business_days: pd.Series) -> pd.Series:
    """
    Count the number of business days between each pair of start_date and end_date.
    - Positive count if start_date < end_date
    - Negative count if start_date > end_date
    - Zero if start_date == end_date

    Parameters:
    - start_dates (pd.Series): Series of start dates.
    - end_dates (pd.Series): Series of end dates.
    - business_days (pd.Series): Series of valid business dates.

    Returns:
    - pd.Series: Series of business day counts (signed).
    """
    # Convert to datetime.date
    start_dates = pd.to_datetime(start_dates).dt.date
    end_dates = pd.to_datetime(end_dates).dt.date
    business_days = pd.to_datetime(business_days).dt.date

    # Fast lookup
    business_days_set = set(business_days)

    results = []
    for start, end in zip(start_dates, end_dates):
        if start == end:
            results.append(0)
        else:
            # Define the range boundaries
            start_bound = min(start, end)
            end_bound = max(start, end)
            count = sum(start_bound < day <= end_bound for day in business_days_set)

            # Apply sign
            if start > end:
                count = -count
            results.append(count)

    return pd.Series(results, index=start_dates.index)



def get_holidays(
    exchange_name: str,
    start_date: datetime.date,
    end_date:  datetime.date
) -> list[ datetime.date]:
    """
    Get holidays for a specific exchange between start_date and end_date.

    Parameters:
        exchange_name (str): Name of the exchange (e.g., 'XNYS' for NYSE).
        start_date ( datetime.date): Start date for holiday retrieval.
        end_date ( datetime.date): End date for holiday retrieval.

    Returns:
        List[ datetime.date]: List of holidays between start_date and end_date.
    """

    holidays = pd.to_datetime(pd.Series(get_calendar(exchange_name).holidays().holidays)).dt.date

    return holidays[( holidays>= start_date ) & (holidays <= end_date ) ].unique().tolist()

def get_nyse_business_dates(start_date: datetime.date,
                            end_date:  datetime.date
                        ) -> list[ datetime.date]:
    """ Get  business dates between two dates """
    dates = pd.date_range(start_date, end_date, freq = 'D')
    weekend_mask = (dates.dayofweek ==5) | (dates.dayofweek == 6)
    holidays = get_holidays(exchange_name =  'NYSE', start_date = start_date, end_date = end_date)
    holiday_mask = dates.isin(holidays)
    non_business_day_mask = weekend_mask | holiday_mask
    business_dates = pd.to_datetime(pd.Series ( list( dates[~non_business_day_mask] ) ) ).dt.date.tolist()

    return business_dates
