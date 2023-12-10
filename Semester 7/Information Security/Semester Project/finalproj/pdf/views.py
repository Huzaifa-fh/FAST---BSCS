import os

from dill import settings
from django.contrib.sessions.backends import file
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from pdf.forms import UploadFileForm
from .models import pdf
from .predictor import predictor
from django.conf import settings

def index(request):
    return render(request, "pdf/index.html")

def layout(request):
    return render(request, "pdf/layout.html")

def home(request):
    return render(request, "pdf/home.html")


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded file from the form
            file = request.FILES['file']

            # Check if the file is a PDF
            if file.content_type != 'application/pdf':
                return render(request, 'pdf/error.html')

            # Print information for debugging
            print("File name:", file.name)
            print("File size:", file.size)
            print("File content type:", file.content_type)

            # Create the pdf object and save it to the database
            Pdf = pdf.objects.create(pdf_file=file)
            Pdf.save()
            return render(request, 'pdf/check.html', {'pdf_file': Pdf})
        else:
            # Print form errors for debugging
            print("Form errors:", form.errors)
    else:
        form = UploadFileForm()

    return render(request, 'pdf/index.html', {'form': form})


def predict_malicious(request, file_path):
    path = os.path.join(settings.BASE_DIR, 'media', file_path)
    model_result = predictor(path)
    response_content = f'File path "{file_path}" used for prediction. Model result: {model_result}'
    send_path = '/media/' + file_path
    return render(request, 'pdf/result.html', {'pdf_file': send_path, 'model_result': model_result})