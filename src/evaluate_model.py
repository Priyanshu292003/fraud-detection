import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_rf_model.pkl")

model = joblib.load(MODEL_PATH)

X_train, X_test, y_train, y_test = load_and_preprocess()

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
