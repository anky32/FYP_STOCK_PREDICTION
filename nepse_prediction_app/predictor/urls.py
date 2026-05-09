from django.urls import path
from django.shortcuts import redirect
from .views import (
    dashboard_view, predict_view, analysis_view, feedback_view,
    login_view, register_view, logout_view,
    verify_otp_view, resend_otp_view,
    admin_login_view, admin_logout_view,
    admin_panel_view, admin_users_view, admin_feedback_view,
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

    # ── Admin Panel ──────────────────────────────
    path('admin-login/',          admin_login_view,    name='admin_login'),
    path('admin-logout/',         admin_logout_view,   name='admin_logout'),
    path('admin-panel/',          admin_panel_view,    name='admin_panel'),
    path('admin-panel/users/',    admin_users_view,    name='admin_users'),
    path('admin-panel/feedback/', admin_feedback_view, name='admin_feedback'),
]