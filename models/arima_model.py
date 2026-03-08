import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/cleaned/GBBL_data_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])

df.set_index("Date", inplace=True)

# Use only Close price
data = df["Close"]

# Train/Test split (80/20)
train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

print("Training samples:", len(train))
print("Testing samples:", len(test))

# Train ARIMA model
model = ARIMA(train, order=(5,1,0))

model_fit = model.fit()

print(model_fit.summary())

# Predict on test data
predictions = model_fit.forecast(steps=len(test))

# Evaluation metrics
mae = mean_absolute_error(test, predictions)

rmse = np.sqrt(mean_squared_error(test, predictions))

print("\nModel Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)

# Plot predictions
plt.figure(figsize=(12,6))

plt.plot(train.index, train, label="Training Data")

plt.plot(test.index, test, label="Actual Price")

plt.plot(test.index, predictions, label="Predicted Price")

plt.title("ARIMA Stock Price Prediction")

plt.xlabel("Date")

plt.ylabel("Price")

plt.legend()

plt.show()