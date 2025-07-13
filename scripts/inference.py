# scripts/inference.py

import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('models/logistic_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Sample input: [Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex_M, ChestPainType_ATA, ...]
sample_data = np.array([[55, 140, 230, 0, 150, 1.2, 1, 0, 1, 0, 1, 0, 1, 1, 0]])
sample_data_scaled = scaler.transform(sample_data)

# Predict
prediction = model.predict(sample_data_scaled)
print("Predicted class (0 = No Heart Disease, 1 = Heart Disease):", prediction[0])