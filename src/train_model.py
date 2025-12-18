import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess


# Absolute project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_rf_model.pkl")

X_train, X_test, y_train, y_test = load_and_preprocess()

model = RandomForestClassifier(
    n_estimators=100,   # reduced for faster training
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

print("Training Random Forest...")
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)
print("Random Forest model trained and saved successfully.")
