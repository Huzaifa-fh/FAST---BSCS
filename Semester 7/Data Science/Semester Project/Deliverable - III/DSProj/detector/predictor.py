from django.conf import settings
import joblib
from joblib import load
import numpy as np
import os


def predictor(features):
    # Assuming 'random_forest_model.joblib' is in the same directory as your Django project
    model_file_path = os.path.join(settings.BASE_DIR, 'detector/best_model.joblib')

    # Load the model using the absolute path
    model = load(model_file_path)

    features = np.array(features)
    result = model.predict(features.reshape(1, -1))
    print(result)
    return result

