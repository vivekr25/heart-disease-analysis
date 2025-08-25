from flask import Flask, request, render_template, url_for
import joblib, json, os
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # headless plotting (important for servers)
import matplotlib.pyplot as plt
from uuid import uuid4

# Explicitly declare static folder (served at /static)
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load model + scaler
model = joblib.load('models/logistic_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Load SHAP background and feature names (you just created these)
bg = np.load('models/shap_background_scaled.npy')
with open('models/feature_names.json') as f:
    FEATURE_NAMES = json.load(f)

# One-time SHAP explainer for Logistic Regression
# Function: LinearExplainer(model, background, feature_names=...)
explainer = shap.LinearExplainer(model, bg, feature_names=FEATURE_NAMES)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1) Read form values in the SAME ORDER as FEATURE_NAMES
            raw_vals = [float(x) for x in request.form.values()]
            X_row = np.array([raw_vals], dtype=float)

            # 2) Scale to match training
            X_row_scaled = scaler.transform(X_row)

            # 3) Predict class
            pred = model.predict(X_row_scaled)[0]
            result = '💔 High Risk: Heart Disease Detected' if pred == 1 else '💚 Low Risk: No Heart Disease'

            # 4) SHAP for THIS one row
            sv = explainer(X_row_scaled)

            # 5) Ensure static dir exists, save waterfall image to absolute static path
            os.makedirs(app.static_folder, exist_ok=True)
            img_name = f"shap_{uuid4().hex}.png"
            img_path = os.path.join(app.static_folder, img_name)

            plt.figure()
            # Function: waterfall shows which features pushed risk up (red) or down (blue)
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()

            # 6) Build URL for the saved static file and render
            shap_url = url_for('static', filename=img_name)
            return render_template('result.html',
                                   prediction_text=result,
                                   shap_img=shap_url)
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {e}")
    return render_template('index.html')

if __name__ == '__main__':
    # Bind to all interfaces and Render's assigned port
    # Syntax: app.run(host, port, debug) — on Render it uses $PORT; locally it defaults to 5000 (or your PORT env var)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)