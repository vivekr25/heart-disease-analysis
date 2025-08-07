# app.py

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/logistic_model.joblib')
scaler = joblib.load('models/scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form values and convert to float
            features = [float(x) for x in request.form.values()]
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            result = '💔 High Risk: Heart Disease Detected' if prediction == 1 else '💚 Low Risk: No Heart Disease'
            
            # Go to result page and show SHAP plot + prediction
            return render_template('result.html', prediction_text=result)
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")
    else:
        return render_template('index.html')
        
if __name__ == '__main__':
    app.run(debug=True)

from flask import send_file

@app.route('/explain')
def explain():
    return send_file("static/shap_summary.png", mimetype='image/png')   