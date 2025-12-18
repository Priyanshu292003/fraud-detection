import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_isolation_forest.pkl")

model = joblib.load(MODEL_PATH)

X_train, X_test, y_train, y_test = load_and_preprocess()

# Isolation Forest predictions: -1 (anomaly), 1 (normal)
y_pred_raw = model.predict(X_test)

# Convert to fraud labels: 1 = fraud, 0 = normal
y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
