from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = read_csv('dataset.csv')
X = df.iloc[:, 0: 21]
y = df.iloc[:, 21]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("---Random Forest---")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the model to a file
model_filename = 'random_forest_model.joblib'
joblib.dump(clf, model_filename)

y_pred = clf.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy_score)
print("\nConfusion Matrix:\n", cm)
