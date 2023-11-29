from django.conf import settings
import pandas as pd
from joblib import load
import numpy as np
import os


# Create a function to convert the 'got' part
def convert_got_part(got_part):
    return {
        'names': [item[0] for item in got_part],
        'formats': [item[1] for item in got_part],
        'offsets': list(range(0, len(got_part) * 8, 8)),
        'itemsize': len(got_part) * 8,
    }


def predictor(features):
    # Assuming 'random_forest_model.joblib' is in the same directory as your Django project
    model_file_path = os.path.join(settings.BASE_DIR, 'detector/best_model.joblib')

    model = load(model_file_path)

    features = np.array(features)
    features = features.reshape((1, -1))

    column_names = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest", "isFlaggedFraud",
                    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

    # Create a Pandas DataFrame
    df = pd.DataFrame(features, columns=column_names)
    print(df)

    result = model.predict(df)
    print(result)

    return result

