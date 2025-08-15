import pandas as pd
import os

RAW_PATH = os.path.join("data", "raw", "nifty50.csv")
PROCESSED_PATH = os.path.join("data", "processed", "nifty50_features.csv")


def add_features(df):
    # Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD & Signal Line
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df


def add_labels(df, threshold=0.002):
    """
    Create ternary labels based on next day's percentage return.
    threshold = 0.002 means ±0.2% change
    """
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Label_ternary'] = df['Return'].apply(
        lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
    )
    return df


def process_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"❌ Raw data file not found at {RAW_PATH}. Please run data_loader.py first.")

    # Read CSV
    df = pd.read_csv(RAW_PATH)

    # Convert price columns to numeric (in case they're strings)
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN after conversion
    df.dropna(subset=['Close'], inplace=True)

    # Add features
    df = add_features(df)

    # Add labels
    df = add_labels(df)

    # Remove any NaNs caused by rolling calculations
    df.dropna(inplace=True)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Processed data saved to {PROCESSED_PATH}")
    print(df.head())


if __name__ == "__main__":
    process_data()
