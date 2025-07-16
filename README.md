# 🫀 Heart Disease Prediction - End-to-End Data Science Project

This project explores the Heart Failure Prediction dataset using Python and Pandas, applies machine learning, and deploys a prediction model as a live web app.

---

## 🎯 Project Goals

- Load and explore real-world health data
- Identify key risk factors of heart disease through EDA
- Train a classification model to predict heart disease
- Deploy a working web app for predictions
- Document the entire lifecycle on GitHub

---

## 🛠 Tools & Technologies

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (ML modeling)
- Flask (Web App)
- Google Colab + VSCode (Development)
- Render.com (Deployment)
- Git + GitHub (Version Control)

---

## 📊 Exploratory Data Analysis (EDA)

We explored variables like:

- **Age, MaxHR, Cholesterol, Oldpeak, RestingBP**
- **FastingBS**, **ExerciseAngina**, **ChestPainType**
- Used bar charts, histograms, box plots, scatter plots, and heatmaps
- Key findings:
  - **Exercise-induced angina**, **Oldpeak**, and **ST_Slope** showed strong correlation with heart disease.
  - **MaxHR** was notably lower in patients with heart disease.
  - Cholesterol wasn’t as strongly predictive as expected.

---

## 🤖 Model Training & Evaluation

- Algorithm: **Logistic Regression**
- Accuracy: **85.33%**
- Evaluated with:
  - **Confusion Matrix**
  - **Precision, Recall, F1-score**
- Serialized the trained model and scaler for deployment

---

## 🚀 Deployment Summary

- Built a Flask web app with form-based input
- Connected the app to the serialized model for predictions
- Deployed the app to Render.com

✅ **Live App**: [Heart Disease Predictor](https://heart-disease-predictor-noin.onrender.com)  
(Enter values to get a risk prediction instantly!)

---

## 💻 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/vivekr25/heart-disease-analysis.git
cd heart-disease-analysis

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py

Visit http://127.0.0.1:5000 in your browser.

heart-disease-analysis/
├── data/
│   └── heart.csv
├── models/
│   ├── logistic_model.joblib
│   └── scaler.joblib
├── notebooks/
│   ├── HeartRateExploration.ipynb
│   └── HeartRatePrediction.ipynb
├── scripts/
│   ├── preprocess.py
│   ├── retrain_and_export.py
│   └── inference.py
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── render.yaml

📌 Highlights

✅ Fully documented EDA & modeling
✅ Deployment-ready code
✅ Hosted web app for real-time use
✅ Beginner-friendly, modular structure

⸻

📣 Author

👤 Vivek Raghunathan
💼 Data Analyst | Aspiring Data Scientist
📫 Feel free to fork, star ⭐ and give feedback!