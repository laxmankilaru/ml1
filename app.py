from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
pipe = joblib.load("model.pkl")

@app.route("/")
def index():
    return jsonify({"message": "IPL API is running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        pred = pipe.predict_proba(input_df)[0]
        return jsonify({
            "lose": round(pred[0] * 100, 1),
            "win": round(pred[1] * 100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400