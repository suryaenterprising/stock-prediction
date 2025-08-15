# src/utils.py
import pandas as pd
import numpy as np

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'Date' column is datetime and sort DataFrame.
    """
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators: SMA, RSI, MACD.
    Fully robust for single-row or single-column DataFrames.
    """
    # Flatten multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Ensure Close column exists
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            raise ValueError("❌ Neither 'Close' nor 'Adj Close' found in DataFrame.")

    # Convert columns safely
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            series = df[col]

            # If DataFrame slice, take first column
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]

            # If scalar or 0-d array, wrap as Series
            if np.isscalar(series) or (hasattr(series, 'ndim') and series.ndim == 0):
                series = pd.Series([series], index=df.index)

            # If 1-element array, flatten
            if isinstance(series, np.ndarray) and series.size == 1:
                series = pd.Series(series.flatten(), index=df.index)

            # Ensure index matches
            if not series.index.equals(df.index):
                series.index = df.index

            df[col] = pd.to_numeric(series, errors='coerce')

    # Drop missing Close
    df.dropna(subset=['Close'], inplace=True)

    # ------------------ Moving Averages ------------------
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # ------------------ RSI ------------------
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ------------------ MACD ------------------
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with NaNs introduced by rolling calculations
    df = df.dropna()

    return df
