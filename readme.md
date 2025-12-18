# 💳 Financial Fraud Detection System (ML + Cloud Deployment)

An end-to-end **machine learning–based financial fraud detection system** built using real-world transaction data and deployed as a **public REST API**.

🔗 **Live API:**  
https://fraud-detection-api-tmua.onrender.com

---

## 🚀 Project Overview

Financial fraud causes massive losses every year, while fraudulent transactions represent **less than 1%** of total data.  
This project addresses that challenge by building a **production-grade fraud detection pipeline** with model training, evaluation, and cloud deployment.

The system predicts whether a transaction is fraudulent and returns a **risk probability**, enabling flexible, business-driven decision making.

---

## 🧠 Models Used

### 1️⃣ Random Forest (Primary Model)
- Supervised learning
- Handles non-linear patterns well
- Performs strongly on imbalanced tabular data
- Used as the **production model**

### 2️⃣ Isolation Forest (Secondary / Anomaly Detection)
- Unsupervised learning
- Detects novel or unseen fraud patterns
- Useful when labels are unavailable or delayed

---

## 📊 Key ML Concepts Implemented

- Severe **class imbalance handling**
- Feature scaling & preprocessing
- Model comparison (supervised vs unsupervised)
- Precision–Recall focused evaluation
- Probability-based risk scoring
- Train–inference feature consistency

---

## 🌐 REST API (Flask)

The trained model is exposed via a Flask-based REST API.

### 🔹 Endpoints

#### Health Check
