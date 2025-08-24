# 📈 Stock Prediction by MACHINE LEARNING & Backtesting  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)](https://pandas.pydata.org/)  
[![NumPy](https://img.shields.io/badge/NumPy-Matrix%20Math-orange?logo=numpy)](https://numpy.org/)  
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-green?logo=scikit-learn)](https://scikit-learn.org/stable/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red?logo=plotly)](https://matplotlib.org/)  
[![License](https://img.shields.io/badge/License-MIT-purple)](./LICENSE)  

🚀 A Python-based framework for **stock price prediction, strategy backtesting, and machine learning-driven trading**.  
This project combines **technical analysis + ML models** to generate **BUY/SELL signals** with backtested results.  

---

## 🔧 Tools & Technologies Used  

- **Programming Language:** Python 🐍  
- **Data Analysis:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest, SVM, etc.)  
- **Visualization:** Matplotlib, Seaborn  
- **Data Source:** Yahoo Finance API (`yfinance`)  
- **Backtesting:** Custom backtest engine for MA crossover & ML strategies  

---
## ⚙️ Process Workflow  

The project follows this step-by-step workflow:  

1. **Data Collection** 📥  
   - Stock data is fetched automatically from **Yahoo Finance** using `yfinance`.  
   - Data includes **Open, High, Low, Close, Volume (OHLCV)**.  

2. **Data Preprocessing & Feature Engineering** 🛠️  
   - Missing values handled, data cleaned.  
   - Technical indicators calculated:  
     - **SMA (Simple Moving Average)**  
     - **EMA (Exponential Moving Average)**  
     - **RSI (Relative Strength Index)**  
     - **MACD (Moving Average Convergence Divergence)**  
     - **Bollinger Bands**  
   - Features standardized for ML models.  

3. **Strategy Selection** 🎯  
   - **Option A → Moving Average Crossover Strategy**  
     - Generates **BUY** when short MA > long MA.  
     - Generates **SELL** when short MA < long MA.  
   - **Option B → Machine Learning Model**  
     - Trains ML classifier (Logistic Regression / Random Forest / SVM).  
     - Predicts next-day trend (Up/Down).  

4. **Backtesting Engine** 📊  
   - Historical trades are simulated.  
   - Portfolio value, PnL (Profit & Loss), and accuracy are calculated.  
   - Equity curve generated.  

5. **Results & Visualization** 📈  
   - Buy/Sell points plotted on stock chart.  
   - Equity curve compared with baseline (Buy & Hold).  
   - Performance metrics exported:  
     - Total Return  
     - Sharpe Ratio  
     - Win Rate  
     - Max Drawdown  

---


## ⚡ Installation  

```bash
git clone https://github.com/suryaenterprising/stock-prediction.git
cd stock-prediction
pip install -r requirements.txt






