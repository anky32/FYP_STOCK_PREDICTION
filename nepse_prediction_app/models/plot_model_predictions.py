import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ======================
# PATH FIX (IMPORTANT)
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

data_folder = os.path.join(BASE_DIR, "data", "cleaned")
plot_folder = os.path.join(BASE_DIR, "results", "plots")
model_folder = os.path.join(BASE_DIR, "nepse_prediction_app", "models", "saved_models")

os.makedirs(plot_folder, exist_ok=True)


for file in os.listdir(data_folder):

    if file.endswith(".csv"):

        stock = file.split("_")[0]

        print("\nGenerating plots for:", stock)

        df = pd.read_csv(os.path.join(data_folder, file))

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # 🔥 Fix warning (optional)
        df = df.asfreq('D').ffill()

        # 🔥 MOVING AVERAGES
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()

        data = df["Close"]

        train_size = int(len(data) * 0.8)

        train = data[:train_size]
        test = data[train_size:]

        # ===============================
        # ARIMA
        # ===============================
        with open(os.path.join(model_folder, f"{stock}_arima.pkl"), "rb") as f:
            arima_model = pickle.load(f)

        arima_pred = arima_model.forecast(steps=len(test))

        # ===============================
        # LSTM
        # ===============================
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

        window = 60
        X = []

        for i in range(window, len(scaled_data)):
            X.append(scaled_data[i-window:i, 0])

        X = np.array(X).reshape(-1, window, 1)

        split = int(len(X) * 0.8)
        X_test = X[split:]

        lstm_model = load_model(os.path.join(model_folder, f"{stock}_lstm.h5"), compile=False)

        lstm_pred = lstm_model.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred)

        # ===============================
        # HYBRID
        # ===============================
        hybrid_model = load_model(os.path.join(model_folder, f"{stock}_hybrid_lstm.h5"), compile=False)

        train_pred = arima_model.predict(start=1, end=len(train) - 1)

        residuals = train[1:].values - train_pred
        residuals = np.array(residuals).reshape(-1, 1)

        scaler_res = MinMaxScaler()
        scaled_res = scaler_res.fit_transform(residuals)

        window = 10
        X_res = []

        for i in range(window, len(scaled_res)):
            X_res.append(scaled_res[i-window:i, 0])

        X_res = np.array(X_res).reshape(-1, window, 1)

        X_res_test = X_res[-len(test):]

        lstm_res_pred = hybrid_model.predict(X_res_test, verbose=0)
        lstm_res_pred = scaler_res.inverse_transform(lstm_res_pred)

        hybrid_pred = arima_pred[-len(lstm_res_pred):] + lstm_res_pred.flatten()

        # ===============================
        # PLOT
        # ===============================
        plt.figure(figsize=(12, 6))

        plt.plot(df.index, df["Close"], label="Actual Price", alpha=0.5)

        plt.plot(df.index, df["MA_20"], label="MA 20", linestyle="--")
        plt.plot(df.index, df["MA_50"], label="MA 50", linestyle="--")

        plt.plot(test.index[-len(hybrid_pred):], arima_pred[-len(hybrid_pred):], label="ARIMA")
        plt.plot(test.index[-len(lstm_pred):], lstm_pred, label="LSTM")
        plt.plot(test.index[-len(hybrid_pred):], hybrid_pred, label="Hybrid")

        plt.title(f"{stock} Model Comparison with Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.legend()

        save_path = os.path.join(plot_folder, f"{stock}_comparison.png")
        plt.savefig(save_path)
        plt.close()

        print("Saved:", save_path)


print("\n All plots generated successfully.")