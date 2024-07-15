from django.urls import path
from .views import *

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
]