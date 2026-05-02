"""
Stroke Prediction Flask App
"""

import os, json, pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "models", "artifacts.pkl"), "rb") as f:
    artifacts = pickle.load(f)

with open(os.path.join(BASE, "models", "results.json")) as f:
    model_results = json.load(f)

best_model  = artifacts["best_model"]
best_name   = artifacts["best_name"]
scaler      = artifacts["scaler"]
le_dict     = artifacts["le_dict"]
feat_names  = artifacts["feature_names"]
threshold   = artifacts["threshold"]

def encode_input(form):
    cat_map = {
        "gender":         form.get("gender", "Male"),
        "ever_married":   form.get("ever_married", "No"),
        "work_type":      form.get("work_type", "Private"),
        "Residence_type": form.get("Residence_type", "Urban"),
        "smoking_status": form.get("smoking_status", "never smoked"),
    }
    row = {}
    for col, val in cat_map.items():
        le = le_dict[col]
        classes = list(le.classes_)
        if val not in classes:
            val = classes[0]
        row[col] = le.transform([val])[0]

    row["age"]               = float(form.get("age", 50))
    row["hypertension"]      = int(form.get("hypertension", 0))
    row["heart_disease"]     = int(form.get("heart_disease", 0))
    row["avg_glucose_level"] = float(form.get("avg_glucose_level", 100))
    row["bmi"]               = float(form.get("bmi", 27))

    arr = np.array([[row[f] for f in feat_names]])
    return arr

@app.route("/")
def index():
    return render_template("index.html", best_name=best_name, results=model_results)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        arr  = encode_input(data)

        if best_name == "SVC":
            arr_sc = scaler.transform(arr)
            proba  = best_model.predict_proba(arr_sc)[0][1]
        else:
            proba = best_model.predict_proba(arr)[0][1]

        prediction = int(proba >= threshold)
        risk_pct   = round(proba * 100, 1)

        if risk_pct < 20:
            risk_level = "Low"
        elif risk_pct < 50:
            risk_level = "Moderate"
        elif risk_pct < 70:
            risk_level = "High"
        else:
            risk_level = "Very High"

        factors = []
        if float(data.get("age", 0)) >= 65:
            factors.append("Age \u2265 65")
        if int(data.get("hypertension", 0)):
            factors.append("Hypertension")
        if int(data.get("heart_disease", 0)):
            factors.append("Heart Disease")
        if float(data.get("bmi", 0)) >= 30:
            factors.append("Obesity (BMI \u2265 30)")
        if float(data.get("avg_glucose_level", 0)) >= 140:
            factors.append("Elevated Glucose")
        if data.get("smoking_status") == "smokes":
            factors.append("Active Smoker")

        return jsonify({
            "prediction": prediction,
            "probability": risk_pct,
            "risk_level":  risk_level,
            "factors":     factors,
            "model_used":  best_name,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", results=model_results, best_name=best_name)

@app.route("/api/results")
def api_results():
    return jsonify(model_results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
