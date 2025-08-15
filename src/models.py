import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# Moving Average Crossover
# =========================
class MovingAverageCrossover:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame):
        """Generates +1 (buy), -1 (sell), 0 (hold) signals based on MA crossover."""
        df = df.copy()
        df['MA_Short'] = df['Close'].rolling(window=self.short_window, min_periods=1).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_window, min_periods=1).mean()

        df['Signal'] = 0
        df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
        df.loc[df['MA_Short'] < df['MA_Long'], 'Signal'] = -1
        return df['Signal']

# =========================
# Machine Learning Model
# =========================
class MLTradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = None

    def train(self, df: pd.DataFrame, target_col: str = 'Label_ternary'):
        """Train ML model on historical features."""
        features = [col for col in df.columns if col not in ['Date', 'Close', target_col]]
        self.features = features
        X = df[features]
        y = df[target_col]

        # Drop NaNs
        X = X.dropna()
        y = y.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc

    def predict(self, df: pd.DataFrame):
        """Predict buy/sell/hold signals using trained ML model."""
        if self.features is None:
            raise ValueError("Model has not been trained yet!")
        X = df[self.features].dropna()
        preds = self.model.predict(X)

        # Pad predictions with NaNs for missing rows at start
        full_preds = pd.Series(index=df.index, dtype=float)
        full_preds.loc[X.index] = preds
        return full_preds

# =========================
# Legacy train_model function
# =========================
def train_model(df: pd.DataFrame):
    """Legacy version — simple RandomForest on Label_ternary target."""
    features = [col for col in df.columns if col not in ['Date', 'Close', 'Label_ternary']]
    X = df[features].dropna()
    y = df['Label_ternary'].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc
