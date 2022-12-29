from django.contrib import admin
from django.urls import path,include
from predict import views

urlpatterns = [
    path('',views.leaf,name="leaf"),
    path('test',views.test,name="test"),
    path('tnk',views.tnk,name="tnk")
]
