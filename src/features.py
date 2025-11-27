import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from src import config

def add_time_features(df):
    """Adds hour, day, month, etc."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

def add_holiday_features(df):
    """Adds a binary flag for US Federal Holidays."""
    cal = calendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    return df

def add_seasonality(df):
    """Adds custom seasonality mapping."""
    df['season_type'] = df.index.month.map({
        6:1, 7:1, 8:1, 9:1, # Summer
        12:2, 1:2, 2:2,     # Winter
        3:0, 4:0, 5:0, 10:0, 11:0 # Shoulder
    })
    return df

def add_lags(df):
    """Adds historical lag features defined in config."""
    for lag in config.LAG_HOURS:
        df[f'lag_{lag}'] = df[config.TARGET].shift(lag)
    return df

def generate_features(df):
    """Master function to run all feature engineering steps."""
    print("Generating features...")
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_seasonality(df)
    df = add_lags(df)
    
    # Drop rows with NaN created by lags
    df = df.dropna()
    return df