from statsmodels.tsa.arima.model import ARIMA

def run_arima_model(df):

    data = df["Close"]

    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=5)

    return forecast.tolist()