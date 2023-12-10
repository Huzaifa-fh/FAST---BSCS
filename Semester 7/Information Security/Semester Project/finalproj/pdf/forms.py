# forms.py
from django import forms
#from .models import UploadedFile

class UploadFileForm(forms.Form):
    file = forms.FileField()