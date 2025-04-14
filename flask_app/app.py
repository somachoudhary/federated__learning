from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import build_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

try:
    model = build_model(input_shape=(5,), binary=True)
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/global_model.weights.h5"))
    model.load_weights(weights_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not all(key in data for key in ["Axis1", "Axis2", "Axis3", "Vector Magnitude", "RR"]):
            return jsonify({"error": "Missing required fields"}), 400
        features = np.array([
            data["Axis1"],
            data["Axis2"],
            data["Axis3"],
            data["Vector Magnitude"],
            data["RR"]
        ]).reshape(1, -1)
        scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scaler.pkl"))
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            features = scaler.transform(features)
        else:
            return jsonify({"error": "Scaler not found"}), 500
        pred = model.predict(features, verbose=0)[0][0]
        print(f"Input: {features.tolist()}, Raw prediction: {pred}")
        result = "Stress" if pred > 0.5 else "No Stress"
        return jsonify({"prediction": result, "confidence": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)