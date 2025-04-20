from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# Define model
def build_model(input_shape=(5,), binary=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid' if binary else 'softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load or initialize model
model = None
weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/global_model.weights.h5"))
try:
    model = build_model(input_shape=(5,), binary=True)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("✅ Model weights loaded.")
    else:
        print("⚠️ Model weights not found. Predictions may be unreliable.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Load or generate scaler
scaler = None
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../scaler.pkl"))
try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded.")
except FileNotFoundError:
    print("⚠️ Scaler not found. Generating from CSVs...")
    csv_files = [f"dataset/user_{i}.csv" for i in range(1, 23) if os.path.exists(f"dataset/user_{i}.csv")]
    if csv_files:
        all_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        scaler = StandardScaler()
        scaler.fit(all_data[["Axis1", "Axis2", "Axis3", "Vector Magnitude", "RR"]].values)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print("✅ Scaler generated and saved.")
    else:
        print("❌ No CSVs found. Predictions will fail.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    if scaler is None or model is None:
        return jsonify({"error": "Model or scaler not available. Please ensure scaler.pkl and model weights are present."}), 500
    if not os.path.exists(weights_path):
        return jsonify({"error": "Model weights not found. Please train the model first."}), 500

    try:
        data = request.json
        if not all(key in data for key in ["Axis1", "Axis2", "Axis3", "Vector Magnitude", "RR"]):
            return jsonify({"error": "Missing required fields"}), 400
        features = np.array([
            float(data["Axis1"]),
            float(data["Axis2"]),
            float(data["Axis3"]),
            float(data["Vector Magnitude"]),
            float(data["RR"])
        ]).reshape(1, -1)
        features = scaler.transform(features)
        pred = model.predict(features, verbose=0)[0][0]
        print(f"Input: {features.tolist()}, Raw prediction: {pred}")
        label = "Stress" if pred > 0.5 else "No Stress"
        confidence = float(pred if pred > 0.5 else 1 - pred)
        return jsonify({"prediction": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
