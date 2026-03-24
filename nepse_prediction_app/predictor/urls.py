from django.urls import path
from .views import predict_view, dashboard_view
from .views import predict_view, dashboard_view, analysis_view

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('predict/', predict_view, name='predict'),
    path('analysis/', analysis_view, name='analysis'),
]