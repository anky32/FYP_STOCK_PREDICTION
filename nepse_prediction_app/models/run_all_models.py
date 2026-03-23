import os
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ======================
# PATH SETUP (IMPORTANT)
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

data_folder = os.path.join(BASE_DIR, "data", "cleaned")
results_folder = os.path.join(BASE_DIR, "results")

os.makedirs(results_folder, exist_ok=True)


# ======================
# RESULT STORAGE (FIX)
# ======================
results = []


# ======================
# SEQUENCE FUNCTION
# ======================
def create_sequences(data, window):
    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


# ======================
# MAIN LOOP
# ======================
for file in os.listdir(data_folder):

    if file.endswith(".csv"):

        stock = file.split("_")[0]
        print("\nProcessing:", stock)

        df = pd.read_csv(os.path.join(data_folder, file))

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # 🔥 Optional (removes warning)
        df = df.asfreq('D').ffill()

        data = df["Close"]

        # ======================
        # TRAIN TEST SPLIT
        # ======================
        train_size = int(len(data) * 0.8)

        train = data[:train_size]
        test = data[train_size:]

        # ======================
        # 🔹 ARIMA MODEL
        # ======================
        arima_model = ARIMA(train, order=(5, 1, 0))
        arima_fit = arima_model.fit()

        arima_pred = arima_fit.forecast(steps=len(test))

        arima_mae = mean_absolute_error(test, arima_pred)
        arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))

        # ======================
        # 🔹 LSTM MODEL
        # ======================
        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

        window = 60

        X, y = create_sequences(scaled_data, window)

        split = int(len(X) * 0.8)

        X_train = X[:split]
        X_test = X[split:]

        y_train = y[:split]
        y_test = y[split:]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(window, 1)))
        model.add(LSTM(64))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

        lstm_pred = model.predict(X_test, verbose=0)

        lstm_pred = scaler.inverse_transform(lstm_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        lstm_mae = mean_absolute_error(y_test_inv, lstm_pred)
        lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred))

        # ======================
        # 🔹 HYBRID MODEL
        # ======================
        residuals = train.values[1:] - arima_fit.predict(start=1, end=len(train) - 1)

        residuals = np.array(residuals).reshape(-1, 1)

        scaler2 = MinMaxScaler()
        scaled_res = scaler2.fit_transform(residuals)

        Xr, yr = create_sequences(scaled_res, 10)

        split2 = int(len(Xr) * 0.8)

        Xr_train = Xr[:split2]
        Xr_test = Xr[split2:]

        yr_train = yr[:split2]
        yr_test = yr[split2:]

        Xr_train = Xr_train.reshape(Xr_train.shape[0], Xr_train.shape[1], 1)
        Xr_test = Xr_test.reshape(Xr_test.shape[0], Xr_test.shape[1], 1)

        model2 = Sequential()
        model2.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
        model2.add(LSTM(50))
        model2.add(Dense(1))

        model2.compile(optimizer="adam", loss="mse")
        model2.fit(Xr_train, yr_train, epochs=10, verbose=0)

        res_pred = model2.predict(Xr_test, verbose=0)

        res_pred = scaler2.inverse_transform(res_pred)

        hybrid_pred = arima_pred[-len(res_pred):] + res_pred.flatten()
        actual = test.values[-len(hybrid_pred):]

        hybrid_mae = mean_absolute_error(actual, hybrid_pred)
        hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))

        # ======================
        # STORE RESULTS
        # ======================
        results.append([
            stock,
            arima_mae, arima_rmse,
            lstm_mae, lstm_rmse,
            hybrid_mae, hybrid_rmse
        ])


# ======================
# SAVE RESULTS
# ======================
results_df = pd.DataFrame(results, columns=[
    "Stock",
    "ARIMA_MAE", "ARIMA_RMSE",
    "LSTM_MAE", "LSTM_RMSE",
    "HYBRID_MAE", "HYBRID_RMSE"
])

save_path = os.path.join(results_folder, "model_comparison_results.csv")

results_df.to_csv(save_path, index=False)

print("\n✅ Final Results:")
print(results_df)

print(f"\n📁 Saved to: {save_path}")