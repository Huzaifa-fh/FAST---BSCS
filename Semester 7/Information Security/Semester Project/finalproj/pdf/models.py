from django.db import models

# Create your models here.

class pdf(models.Model):
    pdf_file = models.FileField(null=True)


