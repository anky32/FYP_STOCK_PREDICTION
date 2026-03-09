import pandas as pd
import numpy as np
import os

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

folder = "data/cleaned"

results = []

def create_sequences(data, window):

    X=[]
    y=[]

    for i in range(window,len(data)):
        X.append(data[i-window:i,0])
        y.append(data[i,0])

    return np.array(X),np.array(y)


for file in os.listdir(folder):

    if file.endswith(".csv"):

        stock=file.split("_")[0]

        print("\nHybrid training:",stock)

        path=os.path.join(folder,file)

        df=pd.read_csv(path)

        df["Date"]=pd.to_datetime(df["Date"])
        df.set_index("Date",inplace=True)

        data=df["Close"]

        train_size=int(len(data)*0.8)

        train=data[:train_size]
        test=data[train_size:]


        arima=ARIMA(train,order=(5,1,0)).fit()

        arima_pred=arima.forecast(steps=len(test))


        train_pred=arima.predict(start=1,end=len(train)-1)

        residuals=train[1:].values-train_pred

        residuals=np.array(residuals).reshape(-1,1)

        scaler=MinMaxScaler()

        scaled=scaler.fit_transform(residuals)

        window=10

        X,y=create_sequences(scaled,window)

        train_size=int(len(X)*0.8)

        X_train=X[:train_size]
        X_test=X[train_size:]

        y_train=y[:train_size]
        y_test=y[train_size:]

        X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

        model=Sequential()

        model.add(LSTM(64,return_sequences=True,input_shape=(window,1)))
        model.add(LSTM(64))
        model.add(Dense(1))

        model.compile(optimizer="adam",loss="mse")

        model.fit(X_train,y_train,epochs=25,batch_size=16,verbose=0)

        model.save(f"models/saved_models/{stock}_hybrid_lstm.h5")

        lstm_pred=model.predict(X_test)

        lstm_pred=scaler.inverse_transform(lstm_pred)

        hybrid_pred=arima_pred[-len(lstm_pred):]+lstm_pred.flatten()

        actual=test.values[-len(hybrid_pred):]

        mae=mean_absolute_error(actual,hybrid_pred)
        rmse=np.sqrt(mean_squared_error(actual,hybrid_pred))

        results.append([stock,mae,rmse])

        print(stock,"MAE:",mae,"RMSE:",rmse)

results_df=pd.DataFrame(results,columns=["Stock","MAE","RMSE"])

os.makedirs("results", exist_ok=True)

results_df.to_csv("results/hybrid_results.csv",index=False)

print("\nHybrid Results Saved")