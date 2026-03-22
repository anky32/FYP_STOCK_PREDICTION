import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def run_lstm_model(df):

    data = df["Close"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, window):
        X = []
        for i in range(window, len(data)):
            X.append(data[i-window:i, 0])
        return np.array(X)

    window = 60
    X = create_sequences(scaled_data, window)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window,1)))
    model.add(LSTM(64))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, scaled_data[window:], epochs=5, batch_size=32, verbose=0)

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