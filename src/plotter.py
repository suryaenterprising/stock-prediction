import os
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"

def _safe_symbol(symbol: str) -> str:
    # filenames can't include '^' on Windows easily
    return symbol.replace("^", "").replace("/", "_")

def plot_price_with_signals(df: pd.DataFrame, symbol: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{_safe_symbol(symbol)}_signals.png")

    plt.figure(figsize=(13, 6))
    plt.plot(df["Date"], df["Close"], label="Close")

    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]
    if not buys.empty:
        plt.scatter(buys["Date"], buys["Close"], marker="^", s=60, label="Buy")
    if not sells.empty:
        plt.scatter(sells["Date"], sells["Close"], marker="v", s=60, label="Sell")

    plt.title(f"{symbol} — Price with Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out

def plot_equity_curves(df: pd.DataFrame, symbol: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{_safe_symbol(symbol)}_equity.png")

    plt.figure(figsize=(13, 5))
    plt.plot(df["Date"], df["Cumulative_Market_Return"], label="Market")
    plt.plot(df["Date"], df["Cumulative_Strategy_Return"], label="Strategy")
    plt.title(f"{symbol} — Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Growth (×)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out
