import streamlit as st
from data_loader import load_data
from feature_engineering import add_features
from models import MovingAverageCrossover, MLTradingModel
from backtest import backtest_from_signals
import pandas as pd

st.set_page_config(page_title="NSE Strategy Dashboard", layout="wide")

st.title("📈 NSE Strategy Dashboard")
symbol = st.text_input("Symbol (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS, ^NSEI)", "RELIANCE.NS")
col1, col2, col3 = st.columns(3)
start = col1.text_input("Start Date (YYYY-MM-DD)", "2020-01-01")
end   = col2.text_input("End Date (YYYY-MM-DD or blank for today)", "")
strategy = col3.selectbox("Strategy", ["Moving Average", "Machine Learning"])

if st.button("Run"):
    with st.spinner("Fetching & processing..."):
        df = load_data(symbol, start=start, end=end or None)
        df = add_features(df)

        if strategy == "Moving Average":
            short = st.number_input("Short Window", min_value=5, value=20, step=1)
            long  = st.number_input("Long Window", min_value=10, value=50, step=1)
            mac = MovingAverageCrossover(short, long)
            df['Signal'] = mac.generate_signals(df)
        else:
            ml = MLTradingModel()
            acc = ml.train(df, target_col='Label_ternary')
            st.success(f"ML Accuracy: {acc:.2f}")
            df['Signal'] = ml.predict(df)

        bt_df, m = backtest_from_signals(df, 'Signal', fee_bps=5, allow_flat=True)

    st.subheader("Performance")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Return", f"{m.total_return:.2%}")
    kpi2.metric("CAGR", f"{m.cagr:.2%}")
    kpi3.metric("Sharpe", f"{m.sharpe:.2f}")
    kpi4.metric("Max DD", f"{m.max_drawdown:.2%}")

    st.subheader("Equity Curves")
    st.line_chart(bt_df.set_index('Date')[['Equity_Market','Equity_Strategy']])

    st.subheader("Price & Signals")
    show = bt_df[['Date','Close']]
    if 'SMA20' in bt_df.columns: show['SMA20'] = bt_df['SMA20']
    if 'SMA50' in bt_df.columns: show['SMA50'] = bt_df['SMA50']
    st.line_chart(show.set_index('Date'))

    st.subheader("Backtest Data (tail)")
    st.dataframe(bt_df.tail(300))
