from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.home, name="home"),
    path("index/", views.index, name="index"),
    path("layout/", views.index, name="layout"),
    path('transaction_form/', views.transaction_form, name="transaction_form"),
]

