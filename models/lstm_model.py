import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("data/cleaned/GBBL_data_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

data = df["Close"].values.reshape(-1,1)


# ==============================
# SCALE DATA
# ==============================

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)


# ==============================
# CREATE SEQUENCES (WINDOW)
# ==============================

def create_sequences(data, window_size):

    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i,0])

    return np.array(X), np.array(y)


window_size = 60

X, y = create_sequences(scaled_data, window_size)


# ==============================
# TRAIN TEST SPLIT
# ==============================

train_size = int(len(X)*0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)


# ==============================
# BUILD LSTM MODEL
# ==============================

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(window_size,1)))

model.add(LSTM(64))

model.add(Dense(25))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# ==============================
# TRAIN MODEL
# ==============================

model.fit(X_train, y_train, epochs=20, batch_size=32)


# ==============================
# PREDICT
# ==============================

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

y_test = scaler.inverse_transform(y_test.reshape(-1,1))


# ==============================
# EVALUATE
# ==============================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\nLSTM Model Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)


# ==============================
# PLOT
# ==============================

plt.figure(figsize=(12,6))

plt.plot(df.index[-len(y_test):], y_test, label="Actual Price")

plt.plot(df.index[-len(predictions):], predictions, label="Predicted Price")

plt.title("LSTM Stock Price Prediction")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.show()