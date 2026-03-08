import pandas as pd
import numpy as np
import os

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

folder = "data/cleaned"

results = []

for file in os.listdir(folder):

    if file.endswith(".csv"):

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

        stock_name = file.split("_")[0]

        results.append([stock_name, mae, rmse])

        print(stock_name, "done")


results_df = pd.DataFrame(results, columns=["Stock","MAE","RMSE"])

print("\nFinal Results")
print(results_df)

results_df.to_csv("arima_results.csv", index=False)