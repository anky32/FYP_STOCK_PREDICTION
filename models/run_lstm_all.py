import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


folder = "data/cleaned"

results = []


def create_sequences(data, window_size):

    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i,0])

    return np.array(X), np.array(y)


for file in os.listdir(folder):

    if file.endswith(".csv"):

        stock_name = file.split("_")[0]

        print("\nTraining LSTM for:", stock_name)

        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        df["Date"] = pd.to_datetime(df["Date"])

        df.set_index("Date", inplace=True)

        data = df["Close"].values.reshape(-1,1)


        scaler = MinMaxScaler(feature_range=(0,1))

        scaled_data = scaler.fit_transform(data)


        window_size = 60

        X, y = create_sequences(scaled_data, window_size)


        train_size = int(len(X)*0.8)

        X_train = X[:train_size]
        X_test = X[train_size:]

        y_train = y[:train_size]
        y_test = y[train_size:]


        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)


        model = Sequential()

        model.add(LSTM(50, return_sequences=True, input_shape=(window_size,1)))

        model.add(LSTM(50))

        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")


        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)


        predictions = model.predict(X_test)

        predictions = scaler.inverse_transform(predictions)

        y_test = scaler.inverse_transform(y_test.reshape(-1,1))


        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))


        results.append([stock_name, mae, rmse])

        print(stock_name, "MAE:", mae, "RMSE:", rmse)


results_df = pd.DataFrame(results, columns=["Stock","MAE","RMSE"])


print("\nFinal LSTM Results")

print(results_df)


results_df.to_csv("lstm_results.csv", index=False)