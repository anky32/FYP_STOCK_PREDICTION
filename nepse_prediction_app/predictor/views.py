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

from .models import Feedback, OTPCode
from .forms import RegisterForm

from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.utils import timezone


STOCKS = ["GBBL", "CIT", "NABIL", "DDBL", "ICFC", "KKHC", "NLIC", "PRIN", "SHL", "STC", "UNL"]


# 🔹 DASHBOARD
@login_required(login_url='/login/')
def dashboard_view(request):
    return render(request, "predictor/dashboard.html", {"stocks": STOCKS})


# 🔹 PREDICTION
@login_required(login_url='/login/')
def predict_view(request):
    data = None
    if request.method == "POST":
        stock = request.POST.get("stock")
        model = request.POST.get("model")
        data = run_prediction(stock, model)
    return render(request, "predictor/predict.html", {"data": data})


# 🔹 ANALYSIS
@login_required(login_url='/login/')
def analysis_view(request):
    file_path = os.path.join(settings.BASE_DIR.parent, "results", "model_comparison_results.csv")
    df = pd.read_csv(file_path)
    data = df.to_dict(orient="records")

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

    return render(request, "predictor/analysis.html", {"data": data, "graph": graph})


# 🔹 FEEDBACK
@login_required(login_url='/login/')
def feedback_view(request):
    success = False
    if request.method == "POST":
        stock   = request.POST.get("stock")
        model   = request.POST.get("model")
        rating  = request.POST.get("rating")
        comment = request.POST.get("comment")

        Feedback.objects.create(
            stock=stock, model=model, rating=rating, comment=comment
        )

        # Notify admin
        stars = '★' * int(rating) + '☆' * (5 - int(rating))
        try:
            send_mail(
                subject=f"[NEPSE AI] New Feedback — {stock} ({model})",
                message=(
                    f"New feedback received on NEPSE AI\n\n"
                    f"Stock  : {stock}\n"
                    f"Model  : {model.upper()}\n"
                    f"Rating : {stars} ({rating}/5)\n"
                    f"User   : {request.user.username} ({request.user.email})\n\n"
                    f"Comment:\n{comment or 'No comment provided'}\n"
                ),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=['pandeyarpan84@gmail.com'],
                fail_silently=True,
            )
        except Exception as e:
            print(f"[FEEDBACK EMAIL ERROR] {e}")

        success = True
    return render(request, "predictor/feedback.html", {"success": success})


# 🔹 LOGIN — step 1: credentials, step 2: OTP
def login_view(request):
    error = None

    if request.method == "POST":
        identifier = request.POST.get("username", "").strip()
        password = request.POST.get("password")

        # Try username first, then fall back to email lookup
        user = authenticate(request, username=identifier, password=password)
        if user is None:
            u = User.objects.filter(email__iexact=identifier).first()
            if u:
                user = authenticate(request, username=u.username, password=password)

        if user:
            if not user.email:
                login(request, user)
                return redirect('dashboard')

            # Generate OTP
            code = OTPCode.generate_code()
            OTPCode.objects.create(email=user.email, code=code)

            # Store username in session for OTP step
            request.session['otp_username'] = user.username

            email_sent = False
            try:
                send_mail(
                    subject="Your NEPSE AI Login OTP",
                    message=f"Your one-time password is: {code}\n\nThis code expires in 10 minutes.",
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[user.email],
                    fail_silently=False,
                )
                email_sent = True
            except Exception as e:
                print(f"[EMAIL ERROR] {e}")

            # In dev (DEBUG=True) show OTP on screen if email not configured
            if not email_sent and settings.DEBUG:
                request.session['dev_otp'] = code

            return redirect('verify_otp')
        else:
            error = "Invalid username or password"

    return render(request, "predictor/login.html", {"error": error})


# 🔹 OTP VERIFY
def verify_otp_view(request):
    username = request.session.get('otp_username')
    if not username:
        return redirect('login')

    error = None

    if request.method == "POST":
        entered = request.POST.get("otp", "").strip()
        try:
            user = User.objects.get(username=username)
            otp = OTPCode.objects.filter(
                email=user.email,
                code=entered,
                is_used=False
            ).order_by('-created_at').first()

            if otp and not otp.is_expired():
                otp.is_used = True
                otp.save()
                del request.session['otp_username']
                login(request, user)
                return redirect('dashboard')
            elif otp and otp.is_expired():
                error = "OTP has expired. Please log in again."
            else:
                error = "Invalid OTP. Please try again."
        except User.DoesNotExist:
            return redirect('login')

    # Mask email for display
    try:
        user = User.objects.get(username=username)
        email = user.email
        parts = email.split('@')
        masked = parts[0][:2] + '***@' + parts[1] if len(parts) == 2 else email
    except Exception:
        masked = "your email"

    return render(request, "predictor/verify_otp.html", {
        "masked_email": masked,
        "error": error,
        "dev_otp": request.session.get('dev_otp') if settings.DEBUG else None,
    })


# 🔹 RESEND OTP
def resend_otp_view(request):
    username = request.session.get('otp_username')
    if not username:
        return redirect('login')

    try:
        user = User.objects.get(username=username)
        code = OTPCode.generate_code()
        OTPCode.objects.create(email=user.email, code=code)
        send_mail(
            subject="Your NEPSE AI Login OTP",
            message=f"Your new one-time password is: {code}\n\nThis code expires in 10 minutes.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=True,
        )
    except Exception:
        pass

    return redirect('verify_otp')


# 🔹 REGISTER
def register_view(request):
    form = RegisterForm()
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    return render(request, "predictor/register.html", {"form": form})


# 🔹 LOGOUT
def logout_view(request):
    logout(request)
    return redirect('/login/')
