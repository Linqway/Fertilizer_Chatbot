from django.contrib import admin
from django.urls import path

from api import views

urlpatterns = [
    path('train/', views.Train.as_view(), name="train"),
    path('predict/', views.Predict.as_view(), name="predict"),
    path('form/', views.Form.as_view(), name="form"),
]