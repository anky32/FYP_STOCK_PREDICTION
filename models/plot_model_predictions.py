import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ===============================
# SETTINGS
# ===============================

data_folder = "data/cleaned"
plot_folder = "results/plots"

os.makedirs(plot_folder, exist_ok=True)


# ===============================
# LOOP THROUGH ALL STOCK FILES
# ===============================

for file in os.listdir(data_folder):

    if file.endswith(".csv"):

        stock = file.split("_")[0]

        print("\nGenerating plots for:", stock)

        path = os.path.join(data_folder, file)

        df = pd.read_csv(path)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        data = df["Close"]

        train_size = int(len(data)*0.8)

        train = data[:train_size]
        test = data[train_size:]


        # ===============================
        # LOAD ARIMA MODEL
        # ===============================

        with open(f"models/saved_models/{stock}_arima.pkl", "rb") as f:
            arima_model = pickle.load(f)

        arima_pred = arima_model.forecast(steps=len(test))


        # ===============================
        # LSTM PREDICTION
        # ===============================

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(data.values.reshape(-1,1))

        window = 60

        X = []
        for i in range(window, len(scaled_data)):
            X.append(scaled_data[i-window:i,0])

        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        train_size = int(len(X)*0.8)
        X_test = X[train_size:]


        lstm_model = load_model(f"models/saved_models/{stock}_lstm.h5", compile=False)

        lstm_pred = lstm_model.predict(X_test)

        lstm_pred = scaler.inverse_transform(lstm_pred)


        # ===============================
        # HYBRID MODEL
        # ===============================

        hybrid_model = load_model(f"models/saved_models/{stock}_hybrid_lstm.h5", compile=False)

        train_pred = arima_model.predict(start=1, end=len(train)-1)

        residuals = train[1:].values - train_pred
        residuals = np.array(residuals).reshape(-1,1)

        scaler_res = MinMaxScaler()
        scaled_res = scaler_res.fit_transform(residuals)

        window = 10

        X_res = []

        for i in range(window, len(scaled_res)):
            X_res.append(scaled_res[i-window:i,0])

        X_res = np.array(X_res)
        X_res = X_res.reshape(X_res.shape[0], X_res.shape[1], 1)

        X_res_test = X_res[-len(test):]

        lstm_res_pred = hybrid_model.predict(X_res_test)

        lstm_res_pred = scaler_res.inverse_transform(lstm_res_pred)

        hybrid_pred = arima_pred[-len(lstm_res_pred):] + lstm_res_pred.flatten()

        actual = test.values[-len(hybrid_pred):]


        # ===============================
        # ARIMA PLOT
        # ===============================

        plt.figure(figsize=(10,6))

        plt.plot(test.index, test.values, label="Actual Price")
        plt.plot(test.index, arima_pred, label="ARIMA Prediction")

        plt.title(f"{stock} ARIMA Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.legend()

        plt.savefig(f"{plot_folder}/{stock}_arima_prediction.png")

        plt.close()


        # ===============================
        # LSTM PLOT
        # ===============================

        plt.figure(figsize=(10,6))

        plt.plot(test.index[-len(lstm_pred):], test.values[-len(lstm_pred):], label="Actual Price")
        plt.plot(test.index[-len(lstm_pred):], lstm_pred, label="LSTM Prediction")

        plt.title(f"{stock} LSTM Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.legend()

        plt.savefig(f"{plot_folder}/{stock}_lstm_prediction.png")

        plt.close()


        # ===============================
        # HYBRID PLOT
        # ===============================

        plt.figure(figsize=(10,6))

        plt.plot(test.index[-len(hybrid_pred):], actual, label="Actual Price")
        plt.plot(test.index[-len(hybrid_pred):], hybrid_pred, label="Hybrid Prediction")

        plt.title(f"{stock} Hybrid Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.legend()

        plt.savefig(f"{plot_folder}/{stock}_hybrid_prediction.png")

        plt.close()

        # ========================
        # FINAL COMBINED PLOT
        # ========================

        plt.figure(figsize=(12,6))

        plt.plot(test.index[-len(hybrid_pred):], actual, label="Actual Price")

        plt.plot(test.index[-len(hybrid_pred):], arima_pred[-len(hybrid_pred):], label="ARIMA")

        plt.plot(test.index[-len(lstm_pred):], lstm_pred, label="LSTM")

        plt.plot(test.index[-len(hybrid_pred):], hybrid_pred, label="Hybrid")

        plt.title(f"{stock} Model Comparison")

        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.legend()

        plt.savefig(f"{plot_folder}/{stock}_model_comparison.png")

        plt.close()


print("\nAll combined comparison plots generated.")


print("\nAll stock prediction plots generated successfully.")