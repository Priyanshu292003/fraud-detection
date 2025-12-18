import requests

URL = "http://127.0.0.1:5000/predict"

transaction = {
    "Time": 100000,
    "Amount": 250,
    "V1": -1.2,
    "V2": 0.5,
    "V3": -2.1,
    "V4": 1.3,
    "V5": -0.4,
    "V6": 0.7,
    "V7": -1.8,
    "V8": 0.2,
    "V9": -0.9,
    "V10": -2.3,
    "V11": 1.1,
    "V12": -1.5,
    "V13": 0.4,
    "V14": -3.2,
    "V15": 0.8,
    "V16": -1.7,
    "V17": -2.6,
    "V18": -0.3,
    "V19": 0.5,
    "V20": 0.1,
    "V21": -0.6,
    "V22": 0.9,
    "V23": -0.2,
    "V24": 0.4,
    "V25": -0.1,
    "V26": 0.3,
    "V27": -0.7,
    "V28": 0.2
}

response = requests.post(URL, json=transaction)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
