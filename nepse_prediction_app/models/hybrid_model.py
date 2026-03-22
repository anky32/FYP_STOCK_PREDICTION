import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def run_hybrid_model(df):

    data = df["Close"]

    # ============================
    # 🔹 ARIMA MODEL
    # ============================
    arima = ARIMA(data, order=(5,1,0)).fit()

    arima_future = arima.forecast(steps=5)

    # ============================
    # 🔹 RESIDUALS
    # ============================
    train_pred = arima.predict(start=1, end=len(data)-1)

    residuals = data[1:].values - train_pred
    residuals = np.array(residuals).reshape(-1,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(residuals)

    # ============================
    # 🔹 SEQUENCES
    # ============================
    def create_sequences(data, window):
        X = []
        for i in range(window, len(data)):
            X.append(data[i-window:i, 0])
        return np.array(X)

    window = 10
    X = create_sequences(scaled, window)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # ============================
    # 🔹 LSTM MODEL
    # ============================
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window,1)))
    model.add(LSTM(64))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, scaled[window:], epochs=5, batch_size=16, verbose=0)

    # ============================
    # 🔥 MULTI-STEP RESIDUAL PREDICTION
    # ============================
    last_window = scaled[-window:]
    future_residuals = []

    current_window = last_window.copy()

    for _ in range(5):
        seq = current_window.reshape(1, window, 1)

        pred = model.predict(seq, verbose=0)

        future_residuals.append(pred[0][0])

        current_window = np.append(current_window[1:], pred, axis=0)

    # 🔹 inverse scaling
    future_residuals = scaler.inverse_transform(
        np.array(future_residuals).reshape(-1,1)
    ).flatten()

    # ============================
    # 🔥 FINAL HYBRID OUTPUT
    # ============================
    hybrid_predictions = arima_future.values + future_residuals

    return hybrid_predictions.tolist()