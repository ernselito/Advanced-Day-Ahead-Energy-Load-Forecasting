import pandas as pd
import kagglehub
import os
from src import config

def download_data():
    """Downloads data from Kaggle and moves it to the data folder."""
    print("Downloading data...")
    path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    # In a real production app, you might move the file to config.DATA_DIR here
    # For now, we return the path provided by kagglehub
    return os.path.join(path, 'PJME_hourly.csv')

def load_and_clean_data(filepath):
    """
    Loads CSV, handles duplicate timestamps, and resamples to hourly frequency.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Set index
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    # Handle duplicates and gaps
    df = df.groupby(df.index).mean()
    df = df.resample('H').mean()
    df = df.ffill() # Forward fill missing data
    
    return df