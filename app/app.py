"""
Software Defect Prediction Dashboard
Flask app that loads the trained XGBoost model (SMOTE-balanced, trained on
NASA PROMISE KC1) and predicts defect probability for a given module.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "xgboost_kc1.joblib"
SCALER_PATH = BASE / "models" / "scaler_kc1.joblib"
FEATURES_PATH = BASE / "models" / "features_kc1.joblib"
MEDIANS_PATH = BASE / "models" / "kc1_medians.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)
MEDIANS = json.loads(MEDIANS_PATH.read_text())

# Fields shown on the form; every other feature defaults to its training-set median.
VISIBLE_FIELDS = [
    ("loc", "Lines of Code"),
    ("v(g)", "Cyclomatic Complexity v(g)"),
    ("n", "Halstead Length (n)"),
    ("v", "Halstead Volume (v)"),
    ("uniq_Op", "Unique Operators"),
    ("uniq_Opnd", "Unique Operands"),
    ("branchCount", "Branch Count"),
    ("lOComment", "Lines of Comments"),
]

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = {k: MEDIANS[k] for k, _ in VISIBLE_FIELDS}

    if request.method == "POST":
        row = dict(MEDIANS)  # start from medians, overwrite visible fields
        for key, _ in VISIBLE_FIELDS:
            raw = request.form.get(key, "")
            try:
                val = float(raw)
            except ValueError:
                val = MEDIANS[key]
            row[key] = val
            form_values[key] = val

        X = np.array([[row[f] for f in FEATURES]])
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0, 1])

        if prob >= 0.66:
            risk, css = "High", "risk-high"
        elif prob >= 0.33:
            risk, css = "Medium", "risk-medium"
        else:
            risk, css = "Low", "risk-low"

        result = {"probability": round(prob * 100, 1), "risk": risk, "css": css}

    return render_template("index.html", fields=VISIBLE_FIELDS, values=form_values, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
