import pandas as pd
import numpy as np
import os
import pickle

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

folder = "data/cleaned"
results = []

for file in os.listdir(folder):

    if file.endswith(".csv"):

        stock = file.split("_")[0]
        print("Training ARIMA for:", stock)

        path = os.path.join(folder, file)

        df = pd.read_csv(path)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        data = df["Close"].dropna()

        train_size = int(len(data)*0.8)

        train = data[:train_size]
        test = data[train_size:]

        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=len(test))

        mae = mean_absolute_error(test.values, predictions)
        rmse = np.sqrt(mean_squared_error(test.values, predictions))

        results.append([stock, mae, rmse])

        os.makedirs("models/saved_models", exist_ok=True)

        with open(f"models/saved_models/{stock}_arima.pkl", "wb") as f:
            pickle.dump(model_fit, f)

        print(stock,"MAE:",mae,"RMSE:",rmse)

results_df = pd.DataFrame(results, columns=["Stock","MAE","RMSE"])

os.makedirs("results", exist_ok=True)

results_df.to_csv("results/arima_results.csv", index=False)

print("\nARIMA Results Saved")