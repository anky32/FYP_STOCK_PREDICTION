import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/cleaned/GBBL_data_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

data = df["Close"]


# =========================
# TRAIN TEST SPLIT
# =========================

train_size = int(len(data)*0.8)

train = data[:train_size]
test = data[train_size:]


# =========================
# ARIMA MODEL
# =========================

arima_model = ARIMA(train, order=(5,1,0))
arima_fit = arima_model.fit()

arima_pred = arima_fit.forecast(steps=len(test))


# =========================
# TRAIN RESIDUALS
# =========================

train_pred = arima_fit.predict(start=1, end=len(train)-1)

train_actual = train[1:]

residuals = train_actual.values - train_pred


# =========================
# SCALE RESIDUALS
# =========================

residuals = np.array(residuals).reshape(-1,1)

scaler = MinMaxScaler()

scaled_residuals = scaler.fit_transform(residuals)


# =========================
# CREATE SEQUENCES
# =========================

def create_sequences(data, window):

    X = []
    y = []

    for i in range(window, len(data)):
        X.append(data[i-window:i,0])
        y.append(data[i,0])

    return np.array(X), np.array(y)


window = 10

X, y = create_sequences(scaled_residuals, window)


# =========================
# LSTM TRAIN TEST SPLIT
# =========================

train_size = int(len(X)*0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)


# =========================
# LSTM MODEL
# =========================

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(window,1)))
model.add(LSTM(64))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")


model.fit(X_train, y_train, epochs=25, batch_size=16)


# =========================
# PREDICT RESIDUALS
# =========================

lstm_pred = model.predict(X_test)

lstm_pred = scaler.inverse_transform(lstm_pred)


# =========================
# HYBRID PREDICTION
# =========================

hybrid_pred = arima_pred[-len(lstm_pred):] + lstm_pred.flatten()

actual = test.values[-len(hybrid_pred):]


# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(actual, hybrid_pred)

rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))


print("\nHybrid Model Evaluation")

print("MAE:", mae)

print("RMSE:", rmse)


# =========================
# PLOT RESULTS
# =========================

plt.figure(figsize=(12,6))

plt.plot(test.index[-len(hybrid_pred):], actual, label="Actual Price")

plt.plot(test.index[-len(hybrid_pred):], hybrid_pred, label="Hybrid Prediction")

plt.legend()

plt.title("Hybrid ARIMA-LSTM Prediction")

plt.xlabel("Date")
plt.ylabel("Price")

plt.show()