from django.shortcuts import render, redirect
from src.predict_pipeline import run_prediction
import pandas as pd
import os
from django.conf import settings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from .models import Feedback
from .forms import RegisterForm

from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required


# 🔹 DASHBOARD (PROTECTED)
@login_required(login_url='/login/')
def dashboard_view(request):
    return render(request, "predictor/dashboard.html")


# 🔹 PREDICTION (PROTECTED)
@login_required(login_url='/login/')
def predict_view(request):

    data = None

    if request.method == "POST":
        stock = request.POST.get("stock")
        model = request.POST.get("model")

        data = run_prediction(stock, model)

    return render(request, "predictor/predict.html", {
        "data": data
    })


# 🔹 ANALYSIS (PROTECTED)
@login_required(login_url='/login/')
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

    plt.figure(figsize=(10, 5))

    plt.bar(x, arima, width=0.2, label="ARIMA")
    plt.bar([i + 0.2 for i in x], lstm, width=0.2, label="LSTM")
    plt.bar([i + 0.4 for i in x], hybrid, width=0.2, label="Hybrid")

    plt.xticks([i + 0.2 for i in x], stocks, rotation=45)
    plt.legend()
    plt.title("Model Comparison (MAE)")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    graph = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return render(request, "predictor/analysis.html", {
        "data": data,
        "graph": graph
    })


# 🔹 FEEDBACK (PROTECTED)
@login_required(login_url='/login/')
def feedback_view(request):

    success = False

    if request.method == "POST":
        Feedback.objects.create(
            stock=request.POST.get("stock"),
            model=request.POST.get("model"),
            rating=request.POST.get("rating"),
            comment=request.POST.get("comment")
        )
        success = True

    return render(request, "predictor/feedback.html", {
        "success": success
    })


# 🔹 LOGIN (ALWAYS SHOW LOGIN PAGE FIRST)
def login_view(request):

    error = None

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('dashboard')
        else:
            error = "Invalid username or password"

    return render(request, "predictor/login.html", {"error": error})


# 🔹 REGISTER
def register_view(request):

    form = RegisterForm()

    if request.method == "POST":
        form = RegisterForm(request.POST)

        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')

    return render(request, "predictor/register.html", {"form": form})


# 🔹 LOGOUT
def logout_view(request):
    logout(request)
    return redirect('/login/')