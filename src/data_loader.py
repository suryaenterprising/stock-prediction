# src/data_loader.py
import yfinance as yf
import pandas as pd

def load_data(symbol, start="2020-01-01", end=None):
    """
    Load historical stock data from Yahoo Finance.
    """
    print(f"📥 Downloading {symbol} data from {start} to {end or 'today'}...")
    df = yf.download(symbol, start=start, end=end)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol} between {start} and {end}")
    
    df.reset_index(inplace=True)  # Make Date a column instead of index
    return df
