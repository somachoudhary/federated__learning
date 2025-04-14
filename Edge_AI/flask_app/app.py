# flask_app/app.py

from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained federated model
MODEL_PATH = os.path.join("models", "global_model.h5")
model = load_model(MODEL_PATH)

# Dummy function: convert form inputs to features
def preprocess_input(data):
    try:
        # Assuming input is comma-separated values
        values = [float(x.strip()) for x in data.split(",")]
        return np.array(values).reshape(1, -1)
    except:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form.get("input_data")

    features = preprocess_input(input_data)
    if features is None:
        return jsonify({"error": "Invalid input format. Please use comma-separated numbers."}), 400

    # Predict
    prediction = model.predict(features)
    predicted_class = int(np.argmax(prediction)) if prediction.shape[1] > 1 else float(prediction[0][0])
    
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
