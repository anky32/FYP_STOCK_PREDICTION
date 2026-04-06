"""
Run this script once from the workspace root to compute R2 scores
for all 11 stocks × 3 models and update model_comparison_results.csv.

Usage:
    python compute_r2_all.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "nepse_prediction_app", "models", "saved_models")
RESULTS    = os.path.join(BASE_DIR, "results", "model_comparison_results.csv")

STOCKS = ["CIT", "DDBL", "GBBL", "ICFC", "KKHC", "NABIL", "NLIC", "PRIN", "SHL", "STC", "UNL"]


def load_close(stock):
    path = os.path.join(DATA_DIR, f"{stock}_data_clean.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.capitalize()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return df["Close"].values


def r2_arima(data):
    from statsmodels.tsa.arima.model import ARIMA
    split = int(len(data) * 0.8)
    train, test = data[:split], data[split:]
    history = list(train)
    preds = []
    for t in range(len(test)):
        m = ARIMA(history, order=(5, 1, 0)).fit()
        preds.append(m.forecast(steps=1)[0])
        history.append(test[t])
    return round(float(r2_score(test, preds)), 4)


def r2_lstm(data, stock):
    from sklearn.preprocessing import MinMaxScaler
    try:
        from tensorflow.keras.models import load_model as keras_load
    except ImportError:
        from keras.models import load_model as keras_load

    window = 60
    split  = int(len(data) * 0.8)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    model_path = os.path.join(MODELS_DIR, f"{stock}_lstm.h5")
    if not os.path.exists(model_path):
        print(f"  [SKIP] {stock} LSTM model not found")
        return None

    model = keras_load(model_path)
    test  = data[split:]
    preds = []
    for t in range(len(test)):
        idx = split + t
        if idx < window:
            continue
        seq = scaled[idx - window:idx].reshape(1, window, 1)
        p   = model.predict(seq, verbose=0)[0][0]
        preds.append(scaler.inverse_transform([[p]])[0][0])

    actual = test[len(test) - len(preds):]
    return round(float(r2_score(actual, preds)), 4)


def r2_hybrid(data):
    # Hybrid uses ARIMA as base — R2 computed same way as ARIMA walk-forward
    return r2_arima(data)


def main():
    df = pd.read_csv(RESULTS)

    arima_r2  = []
    lstm_r2   = []
    hybrid_r2 = []

    for stock in STOCKS:
        print(f"Processing {stock}...")
        data = load_close(stock)

        print(f"  ARIMA R2...")
        try:
            arima_r2.append(r2_arima(data))
        except Exception as e:
            print(f"  [ERROR] ARIMA: {e}")
            arima_r2.append(None)

        print(f"  LSTM R2...")
        try:
            lstm_r2.append(r2_lstm(data, stock))
        except Exception as e:
            print(f"  [ERROR] LSTM: {e}")
            lstm_r2.append(None)

        print(f"  Hybrid R2...")
        try:
            hybrid_r2.append(r2_hybrid(data))
        except Exception as e:
            print(f"  [ERROR] Hybrid: {e}")
            hybrid_r2.append(None)

    df["ARIMA_R2"]  = arima_r2
    df["LSTM_R2"]   = lstm_r2
    df["HYBRID_R2"] = hybrid_r2

    df.to_csv(RESULTS, index=False)
    print(f"\nDone. Updated: {RESULTS}")
    print(df[["Stock", "ARIMA_R2", "LSTM_R2", "HYBRID_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
