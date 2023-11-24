import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt


# Read the cleaned_dataset_unchanged.csv file
df2 = pd.read_csv("dataset.csv")


# Assuming your data is stored in a DataFrame called 'df2'

# Drop 'nameOrig' and 'nameDest' columns
df2 = df2.drop(['nameOrig', 'nameDest'], axis=1)

# Split the data into features (X) and target variable (y)
X2 = df2.drop('isFraud', axis=1)
y2 = df2['isFraud']

# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Standardize the features
scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

# Function to print evaluation metrics
def print_metrics(y_true, y_pred, model_name):
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred)}")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{model_name} ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


#----------------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

# Model 1: Random Forest Classifier
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='roc_auc')
grid_search_rf.fit(X2_train_scaled, y2_train)

# Get the best hyperparameters
best_params_rf = grid_search_rf.best_params_

# Train the model with the best hyperparameters
rf_model2_tuned = RandomForestClassifier(random_state=42, **best_params_rf)
rf_model2_tuned.fit(X2_train_scaled, y2_train)
rf_predictions2_tuned = rf_model2_tuned.predict(X2_test_scaled)

# Print metrics for the tuned Random Forest model
print_metrics(y2_test, rf_predictions2_tuned, "Tuned Random Forest")

# Model 2: Logistic Regression
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='roc_auc')
grid_search_lr.fit(X2_train_scaled, y2_train)

# Get the best hyperparameters
best_params_lr = grid_search_lr.best_params_

# Train the model with the best hyperparameters
lr_model2_tuned = LogisticRegression(random_state=42, **best_params_lr)
lr_model2_tuned.fit(X2_train_scaled, y2_train)
lr_predictions2_tuned = lr_model2_tuned.predict(X2_test_scaled)

# Print metrics for the tuned Logistic Regression model
print_metrics(y2_test, lr_predictions2_tuned, "Tuned Logistic Regression")

# Model 3: Support Vector Classifier (SVC)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search_svc = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svc, cv=5, scoring='roc_auc')
grid_search_svc.fit(X2_train_scaled, y2_train)

# Get the best hyperparameters
best_params_svc = grid_search_svc.best_params_

# Train the model with the best hyperparameters
svc_model2_tuned = SVC(probability=True, random_state=42, **best_params_svc)
svc_model2_tuned.fit(X2_train_scaled, y2_train)
svc_predictions2_tuned = svc_model2_tuned.predict(X2_test_scaled)

# Print metrics for the tuned Support Vector Classifier model
print_metrics(y2_test, svc_predictions2_tuned, "Tuned Support Vector Classifier")


#----------------------------------MODEL SELECTION-------------------------------------

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{model_name} ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }

# Assuming you have the necessary imports

# Evaluate the models
rf_metrics = evaluate_model(y2_test, rf_predictions2_tuned, "Tuned Random Forest")
lr_metrics = evaluate_model(y2_test, lr_predictions2_tuned, "Tuned Logistic Regression")
svc_metrics = evaluate_model(y2_test, svc_predictions2_tuned, "Tuned Support Vector Classifier")

# Compare models based on multiple metrics
models_metrics = {
    "Random Forest": rf_metrics,
    "Logistic Regression": lr_metrics,
    "Support Vector Classifier": svc_metrics
}

# Choose the model with the best average performance across metrics
best_model = max(models_metrics, key=lambda k: sum(models_metrics[k].values()) / len(models_metrics[k]))

print(f"The best model based on overall performance is: {best_model}")

# Save the best model using joblib
if best_model == "Random Forest":
    joblib.dump(rf_model2_tuned, 'best_model.joblib')
elif best_model == "Logistic Regression":
    joblib.dump(lr_model2_tuned, 'best_model.joblib')
elif best_model == "Support Vector Classifier":
    joblib.dump(svc_model2_tuned, 'best_model.joblib')
