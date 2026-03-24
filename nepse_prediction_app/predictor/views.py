from django.shortcuts import render
from src.predict_pipeline import run_prediction
import pandas as pd
import os
from django.conf import settings
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
from io import BytesIO


def dashboard_view(request):
    return render(request, "predictor/dashboard.html")


def predict_view(request):

    data = None

    if request.method == "POST":
        stock = request.POST.get("stock")
        model = request.POST.get("model")

        data = run_prediction(stock, model)

    return render(request, "predictor/predict.html", {
        "data": data
    })


def analysis_view(request):

    file_path = os.path.join(settings.BASE_DIR.parent, "results", "model_comparison_results.csv")

    df = pd.read_csv(file_path)

    data = df.to_dict(orient="records")

    # 🔥 CREATE BAR CHART
    stocks = df["Stock"]
    arima = df["ARIMA_MAE"]
    lstm = df["LSTM_MAE"]
    hybrid = df["HYBRID_MAE"]

    x = range(len(stocks))

    plt.figure(figsize=(10,5))

    plt.bar(x, arima, width=0.2, label="ARIMA")
    plt.bar([i + 0.2 for i in x], lstm, width=0.2, label="LSTM")
    plt.bar([i + 0.4 for i in x], hybrid, width=0.2, label="Hybrid")

    plt.xticks([i + 0.2 for i in x], stocks, rotation=45)
    plt.legend()
    plt.title("Model Comparison (MAE)")

    # 🔥 CONVERT TO IMAGE
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    graph = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return render(request, "predictor/analysis.html", {
        "data": data,
        "graph": graph
    })