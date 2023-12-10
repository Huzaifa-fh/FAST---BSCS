from .feature_extractor import feature_extraction, run_pdfid
from django.conf import settings
from joblib import load
import os


def predictor(pdf_path):
    # Assuming 'random_forest_model.joblib' is in the same directory as your Django project
    model_file_path = os.path.join(settings.BASE_DIR, 'pdf/random_forest_model.joblib')

    # Load the model using the absolute path
    model = load(model_file_path)
    # Load the saved model

    output = run_pdfid(pdf_path)
    print('\n\n\n\n\n\n\n\n\n\n')
    print(pdf_path)
    print(output)
    features = feature_extraction(output)
    result = model.predict(features.reshape(1, -1))
    return result

