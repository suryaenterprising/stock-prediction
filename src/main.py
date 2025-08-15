# src/predict.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from utils import add_features, _ensure_datetime

MODEL_PATH = "models/nifty50_model.pkl"

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# ------------------ Prediction ------------------
def predict_next(symbol: str, start_date: str = "2015-01-01"):
    end_date = datetime.today()
    print(f"🔄 Downloading data for {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date.strftime("%Y-%m-%d"))

    if df.empty:
        print(f"⚠️ No data found for {symbol}.")
        return

    try:
        df = _ensure_datetime(df)
        df = add_features(df)
    except Exception as e:
        print(f"⚠️ Error in feature engineering: {e}")
        return

    if df.empty:
        print(f"⚠️ Data after feature engineering is empty. Cannot predict.")
        return

    # Use last row for prediction
    last_row = df.iloc[[-1]]

    try:
        prediction = model.predict(last_row)
        print(f"📈 Next predicted value for {symbol}: {prediction[0]}")
    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")

# ------------------ CLI ------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict next stock/index value using trained model")
    parser.add_argument("--symbol", type=str, required=True, help="Stock/Index symbol, e.g., ^NSEI")
    parser.add_argument("--live", action="store_true", help="Run live prediction for latest data")
    args = parser.parse_args()

    if args.live:
        print(f"🔄 Running live prediction for {args.symbol}...")
    predict_next(args.symbol)
