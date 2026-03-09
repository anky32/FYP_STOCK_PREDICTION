import pandas as pd

arima=pd.read_csv("results/arima_results.csv")
lstm=pd.read_csv("results/lstm_results.csv")
hybrid=pd.read_csv("results/hybrid_results.csv")

df=arima.merge(lstm,on="Stock",suffixes=("_ARIMA","_LSTM"))

df=df.merge(hybrid,on="Stock")

df.rename(columns={"MAE":"MAE_Hybrid","RMSE":"RMSE_Hybrid"},inplace=True)

df.to_csv("results/model_comparison_results.csv",index=False)

print(df)