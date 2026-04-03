from django.urls import path
from django.shortcuts import redirect
from .views import (
    dashboard_view, predict_view, analysis_view, feedback_view,
    login_view, register_view, logout_view,
    verify_otp_view, resend_otp_view,
)

def home_redirect(request):
    return redirect('/login/')

urlpatterns = [
    path('', home_redirect),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('predict/', predict_view, name='predict'),
    path('analysis/', analysis_view, name='analysis'),
    path('feedback/', feedback_view, name='feedback'),
    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('verify-otp/', verify_otp_view, name='verify_otp'),
    path('resend-otp/', resend_otp_view, name='resend_otp'),
]