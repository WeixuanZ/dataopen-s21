from datetime import date
from functools import partial
from typing import Callable

import pandas as pd

from .values import Populations


ADMISSION_INDICATORS={'daily_norm': 'Daily hospital occupancy',
       'daily_icu': 'Daily ICU occupancy',
       'weekly_norm': 'Weekly new hospital admissions per 100k',
       'weekly_icu': 'Weekly new ICU admissions per 100k'}


def get_yearweek(yearweekstr: str) -> tuple:
    """Transform string of form '2020-W10' into tuple (2020, 10)
    """
    return tuple(map(int, yearweekstr.split('-W')))


def get_week(yearweekstr: str) -> int:
    return get_yearweek(yearweekstr)[1]


def get_week_from_datestr(datestr: str) -> int:
    """Get the week number from ISO formatted string
    """
    return date.fromisoformat(datestr).isocalendar()[1]


def normalize_admission_val(get_population: Callable, row: pd.Series) -> float:
    """Function that normalises the admission to be per 100k of the population of that country.

    Args:
        get_population(callable): Function that returns the population given the name of the country
        row(pd.Series): Row from admission dataframe

    Returns:
        float: Admission per 100k population
    """
    val = row['value']

    # Note that in the provided admission dataframe, only the daily data is given in absolute value
    if row['indicator'] in (ADMISSION_INDICATORS['weekly_norm'], ADMISSION_INDICATORS['weekly_icu']):
        return val

    return val * 100000 / get_population(row['country'])


def process_admission(admission_df: pd.DataFrame):
    """Function that process the admission dataset, returning normalised weekly admission number for both normal and icu

    The weekly data is found by either
        1. summing daily_norm, normalised by the population, of the same week
        2. using the weekly norm

    Args: 
        get_population(callable): Function that returns the population given the name of the country

    Returns: 
        pd.DataFrame: Columns are (country, year_week, norm, icu)
    """
    get_population = Populations()  # Populations() is set up as a closure, enabling O(1) lookup

    admission_df['year_week'] = admission_df['year_week'].apply(get_yearweek)  # change week string into tuple of ints
    admission_df['value'] = admission_df.apply(partial(normalize_admission_val, get_population), axis=1)  # notmalise daily data by the population

    norm_admission = admission_df[(admission_df['indicator'] == ADMISSION_INDICATORS['daily_norm']) | (admission_df['indicator'] == ADMISSION_INDICATORS['weekly_norm'])].groupby(['country', 'year_week'], as_index=False).sum().rename(columns={'value': 'norm'})
    icu_admission = admission_df[(admission_df['indicator'] == ADMISSION_INDICATORS['daily_icu']) | (admission_df['indicator'] == ADMISSION_INDICATORS['weekly_icu'])].groupby(['country', 'year_week'], as_index=False).sum().rename(columns={'value': 'icu'})

    return pd.merge(norm_admission, icu_admission, how="outer", on=['country', 'year_week'])
