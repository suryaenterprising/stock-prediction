# src/predict.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import time
import requests
import argparse

MODEL_PATH = "models/nifty50_model.pkl"
INTERVAL_MINUTES = 5  # Default interval

# ------------------ Telegram Config ------------------
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"   # <-- replace
CHAT_ID = "YOUR_CHAT_ID"            # <-- replace

def send_telegram(message: str):
    """Send message to Telegram bot"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Telegram error: {e}")

# ------------------ Utility Functions ------------------
def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            raise ValueError("❌ Neither 'Close' nor 'Adj Close' found in DataFrame.")

    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if np.isscalar(series) or (hasattr(series, 'ndim') and series.ndim == 0):
                series = pd.Series([series], index=df.index)
            if isinstance(series, np.ndarray) and series.size == 1:
                series = pd.Series(series.flatten(), index=df.index)
            if not series.index.equals(df.index):
                series.index = df.index
            df[col] = pd.to_numeric(series, errors='coerce')

    df.dropna(subset=['Close'], inplace=True)

    # Features
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df.dropna()

# ------------------ Load Model ------------------
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# ------------------ Prediction ------------------
def predict_next(symbol: str):
    end_date = datetime.today()
    df = yf.download(symbol, start="2015-01-01", end=end_date.strftime("%Y-%m-%d"))

    if df.empty:
        print(f"⚠️ No data found for {symbol}.")
        return None

    # Ensure numeric
    for col in df.columns:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if np.isscalar(series) or (hasattr(series, 'ndim') and series.ndim == 0):
            series = pd.Series([series], index=df.index)
        if isinstance(series, np.ndarray) and series.size == 1:
            series = pd.Series(series.flatten(), index=df.index)
        if not series.index.equals(df.index):
            series.index = df.index
        df[col] = pd.to_numeric(series, errors='coerce')

    try:
        df = _ensure_datetime(df)
        df = add_features(df)
    except Exception as e:
        print(f"⚠️ Feature engineering error: {e}")
        return None

    if df.empty:
        print(f"⚠️ Data after feature engineering is empty. Cannot predict.")
        return None

    last_row = df.iloc[[-1]]

    try:
        X = last_row[model.feature_names_in_]
        prediction = model.predict(X)
        return prediction[0]
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None

# ------------------ Live Loop ------------------
def run_live(symbol: str, interval: int = INTERVAL_MINUTES):
    print(f"🔄 Starting live prediction for {symbol} every {interval} minutes...")
    try:
        while True:
            value = predict_next(symbol)
            if value is not None:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                msg = f"{ts}\n📈 {symbol} Predicted Value: {value}"
                print(msg)
                send_telegram(msg)
            time.sleep(interval * 60)
    except KeyboardInterrupt:
        print("🛑 Live prediction stopped by user.")

# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live stock/index predictor with Telegram alerts")
    parser.add_argument("--symbol", type=str, required=True, help="Stock/Index symbol, e.g., ^NSEI")
    parser.add_argument("--live", action="store_true", help="Run live prediction loop")
    parser.add_argument("--interval", type=int, default=INTERVAL_MINUTES, help="Interval in minutes for live prediction")
    args = parser.parse_args()

    if args.live:
        run_live(args.symbol, args.interval)
    else:
        value = predict_next(args.symbol)
        if value is not None:
            msg = f"📈 Predicted value for {args.symbol}: {value}"
            print(msg)
            send_telegram(msg)
