from dataclasses import dataclass
import os
import joblib
import numpy as np
import pandas as pd
from src.plotter import plot_price_with_signals, plot_equity_curves

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "nifty50_model.pkl")

@dataclass
class Metrics:
    total_return: float
    market_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float

def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "").replace("/", "_")

def _compute_metrics(df: pd.DataFrame) -> Metrics:
    strat = df["Cumulative_Strategy_Return"].values
    mkt   = df["Cumulative_Market_Return"].values
    # Daily stats
    rets = df["Strategy_Return"].replace([np.inf,-np.inf], np.nan).dropna()
    sharpe = (np.sqrt(252) * rets.mean() / (rets.std() if rets.std() != 0 else np.nan)) if not rets.empty else np.nan

    # CAGR
    n_days = len(df)
    if n_days > 0:
        years = n_days / 252
        cagr = (strat[-1]) ** (1/years) - 1 if strat[-1] > 0 and years > 0 else np.nan
    else:
        cagr = np.nan

    # Max drawdown (on strategy curve)
    cum = df["Cumulative_Strategy_Return"].copy()
    roll_max = cum.cummax()
    drawdown = (cum / roll_max) - 1
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    # Trades: count of signal changes (ignoring flat)
    pos = df["Signal"].fillna(0).astype(int)
    trades = int((pos.diff().abs() > 0).sum())

    # Win rate: fraction of days in position with positive daily PnL
    pnl = df.loc[pos != 0, "Strategy_Return"]
    win_rate = (pnl > 0).mean() if not pnl.empty else np.nan

    return Metrics(
        total_return = strat[-1] - 1 if len(strat) else np.nan,
        market_return= mkt[-1] - 1   if len(mkt)   else np.nan,
        cagr=cagr,
        sharpe=float(sharpe) if sharpe==sharpe else np.nan,  # guard nan
        max_drawdown=float(max_dd) if max_dd==max_dd else np.nan,
        trades=trades,
        win_rate=float(win_rate) if win_rate==win_rate else np.nan
    )

def backtest_strategy(strategy: str, symbol: str, short_window: int = 20, long_window: int = 50, fee_bps: float = 5.0, allow_flat: bool = True):
    """
    strategy: 'ma' or 'ml'
    symbol: e.g. '^NSEI' or 'RELIANCE.NS'
    """
    safe = _safe_symbol(symbol)
    processed_path = os.path.join(PROCESSED_DIR, f"{safe}_features.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}. Run main.py first.")

    df = pd.read_csv(processed_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # ---- Signals
    if strategy == "ma":
        short_col = f"SMA{short_window}"
        long_col  = f"SMA{long_window}"
        if short_col not in df.columns or long_col not in df.columns:
            raise ValueError(f"Missing columns {short_col}/{long_col} in processed file.")

        sig = np.where(df[short_col] > df[long_col], 1, -1)
        if allow_flat:
            sig[(df[short_col] <= df[long_col]) & (df[short_col] >= df[long_col])] = 0
        df["Signal"] = sig

    elif strategy == "ml":
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first with: python -m src.train_model")
        model = joblib.load(MODEL_PATH)
        feature_cols = ["SMA20", "SMA50", "SMA200", "RSI", "MACD", "Signal_Line"]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing ML feature columns: {missing}")
        X = df[feature_cols].fillna(0)
        df["Signal"] = model.predict(X).astype(int)
    else:
        raise ValueError("strategy must be 'ma' or 'ml'")

    # ---- Returns
    df["Market_Return"] = df["Close"].pct_change().fillna(0)

    # Transaction costs (bps per turnover day)
    turn = (df["Signal"].diff().abs() > 0).astype(int)
    fees = turn * (fee_bps / 10000.0)

    df["Strategy_Return_raw"] = df["Market_Return"] * df["Signal"].shift(1).fillna(0)
    df["Strategy_Return"] = df["Strategy_Return_raw"] - fees

    df["Cumulative_Market_Return"]   = (1 + df["Market_Return"]).cumprod()
    df["Cumulative_Strategy_Return"] = (1 + df["Strategy_Return"]).cumprod()

    metrics = _compute_metrics(df)

    # ---- Plots
    price_path  = plot_price_with_signals(df, symbol)
    equity_path = plot_equity_curves(df, symbol)

    # Save backtest CSV
    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", f"{safe}_{strategy}_backtest.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Results ===")
    print(f"Strategy:        {strategy.upper()} on {symbol}")
    print(f"Total Return:    {metrics.total_return:.2%}")
    print(f"Market Return:   {metrics.market_return:.2%}")
    print(f"CAGR:            {metrics.cagr:.2%}" if metrics.cagr==metrics.cagr else "CAGR:            n/a")
    print(f"Sharpe:          {metrics.sharpe:.2f}" if metrics.sharpe==metrics.sharpe else "Sharpe:          n/a")
    print(f"Max Drawdown:    {metrics.max_drawdown:.2%}" if metrics.max_drawdown==metrics.max_drawdown else "Max Drawdown:    n/a")
    print(f"Trades:          {metrics.trades}")
    print(f"Win Rate:        {metrics.win_rate:.2%}" if metrics.win_rate==metrics.win_rate else "Win Rate:        n/a")
    print("\nSaved:")
    print(f"- {price_path}")
    print(f"- {equity_path}")
    print(f"- {out_csv}")

    return df, metrics
