# Heart Disease Analysis

This project explores the Heart Failure Prediction dataset from Kaggle using Python and Pandas.

## Goals
- Load and explore real-world health data
- Answer questions like:
  - What is the average age of patients?
  - Do older patients have higher heart disease risk?
- Practice data science workflows from data loading to GitHub deployment

## Tools Used
- Python
- Pandas
- Google Colab

## 🧹 Data Preprocessing

The `preprocess.py` script handles the entire data preparation pipeline for the heart disease dataset. It includes:

- **Loading** the CSV dataset.
- **Cleaning** by removing missing values.
- **Encoding** categorical features using one-hot encoding (e.g. Sex, Chest Pain Type).
- **Splitting** the dataset into training and testing sets (80/20 split).
- **Scaling** numerical features using `StandardScaler` for better model performance.

This modular structure makes it easy to maintain and reuse for model training or experimentation.

📊 Exploratory Data Analysis (EDA)
	•	Analyzed variables like Age, Cholesterol, Max Heart Rate, Fasting Blood Sugar, and more.
	•	Explored trends in heart disease presence using bar charts, histograms, scatter plots with regression lines, and heatmaps.
	•	Key findings were added as markdown summaries alongside each visualization.

🧠 Model Training & Evaluation
	•	Used Logistic Regression to predict heart disease.
	•	Achieved an accuracy of 85.33% on the test set.
	•	Evaluated performance using:
	•	Precision (How many predicted positives were correct?)
	•	Recall (How many actual positives were detected?)
	•	F1-Score (Balance between precision and recall)
	•	Confusion Matrix for visual interpretation.
  
  🧠 Model Serialization for Deployment

To make this project ready for deployment:
	•	Trained Logistic Regression model was saved as: models/logistic_model.pkl
	•	Trained Scaler (StandardScaler) was saved as: models/scaler.pkl

These .pkl (pickle) files store the trained objects and can be used later for prediction in production environments such as web apps or APIs.

This is a critical step in making the model reusable, scalable, and ready for deployment using tools like Azure AI Foundry or Flask/Django.

## 🧠 Inference & Deployment Prep

We saved the trained Logistic Regression model and the Scaler as `.joblib` files. These are used for making predictions on new patient data.

```python
# Sample usage
from scripts.inference import predict_heart_disease

# Input format: [Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex_M, ChestPainType_ATA, ...]
sample_input = [55, 140, 230, 0, 150, 1.2, 1, 0, 1, 0, 1, 0, 1, 1, 0]
result = predict_heart_disease(sample_input)
print("Prediction:", result)  # 0 = No Heart Disease, 1 = Heart Disease
