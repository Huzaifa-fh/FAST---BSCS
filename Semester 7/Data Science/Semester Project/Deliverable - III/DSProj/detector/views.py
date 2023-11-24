from dill import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect

# Create your views here.

def index(request):
    return render(request, "detector/index.html")

def home(request):
    return render(request, "detector/home.html")
