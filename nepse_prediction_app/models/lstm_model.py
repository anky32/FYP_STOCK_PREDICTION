import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense


def run_lstm_model(df, stock_name="default"):

    model_path = f"models/saved_models/{stock_name}_lstm.h5"

    data = df["Close"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    window = 60

    def create_sequences(data):
        X = []
        for i in range(window, len(data)):
            X.append(data[i-window:i, 0])
        return np.array(X)

    X = create_sequences(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # ============================
    # 🔥 LOAD MODEL IF EXISTS
    # ============================
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(window,1)))
        model.add(LSTM(64))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        model.fit(X, scaled_data[window:], epochs=5, batch_size=32, verbose=0)

        # 🔥 SAVE MODEL
        os.makedirs("models/saved_models", exist_ok=True)
        model.save(model_path)

    # ============================
    # 🔥 PREDICT FUTURE
    # ============================
    last_seq = scaled_data[-window:]
    predictions = []

    for _ in range(5):
        seq = last_seq.reshape(1, window, 1)
        pred = model.predict(seq, verbose=0)

        predictions.append(pred[0][0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    )

    return predictions.flatten().tolist()