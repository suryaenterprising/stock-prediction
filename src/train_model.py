import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_PATH = os.path.join("data", "processed", "nifty50_features.csv")
MODEL_PATH = os.path.join("models", "nifty50_model.pkl")

def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run the main pipeline to create it: "
            "python -m src.main \"^NSEI\" --start 2015-01-01"
        )

    df = pd.read_csv(PROCESSED_PATH).dropna()
    feature_cols = ["SMA20", "SMA50", "SMA200", "RSI", "MACD", "Signal_Line"]
    target_col = "Label_ternary"

    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Model Training Complete")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
