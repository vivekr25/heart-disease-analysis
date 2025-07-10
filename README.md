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
  