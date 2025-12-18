import os
import joblib
from sklearn.ensemble import IsolationForest
from data_preprocessing import load_and_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_isolation_forest.pkl")

X_train, X_test, y_train, y_test = load_and_preprocess()

model = IsolationForest(
    n_estimators=200,
    contamination=0.002,   # approx fraud rate
    random_state=42,
    n_jobs=-1
)

print("Training Isolation Forest...")
model.fit(X_train)

joblib.dump(model, MODEL_PATH)
print("Isolation Forest model trained and saved.")
