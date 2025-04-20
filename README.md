Federated Learning for Secure Stress Detection
Overview
This project, developed for the samAI Hackathon 2025 (AI for Health & Wellbeing track), presents a privacy-preserving stress detection system using Federated Learning (FL). Leveraging the MMASH dataset, we extract health metrics (accelerometer data and RR intervals) to predict stress levels while keeping user data on-device. A Flask web app provides a user-friendly interface for real-time predictions, showcasing secure, decentralized AI for health monitoring.
Key Features

Privacy-Preserving: Uses Flower AI for Federated Learning, ensuring user data remains private.
Real-Time Predictions: Flask app predicts stress from accelerometer (Axis1, Axis2, Axis3, Vector Magnitude) and RR interval inputs.
Robust Dataset: Processes MMASH dataset into 22 user CSVs (dataset/user_1.csv to user_22.csv) with balanced labels.
Modern UI: Responsive web interface with Home, About, and Contact pages.

Project Structure
federated__learning-master/
├── dataset/                    # Generated CSVs (user_1.csv to user_22.csv)
├── MMASH/                      # MMASH dataset (user_1 to user_22, with Actigraph.csv, IBI.csv)
├── flask_app/                  # Flask app
│   ├── static/
│   │   └── styles.css          # CSS for UI
│   ├── templates/
│   │   ├── index.html          # Home page with prediction form
│   │   ├── about.html          # Project description
│   │   └── contact.html        # Contact info
│   └── app.py                  # Flask app for UI and predictions
├── models/                     # Model weights (global_model.weights.h5)
├── dataset.py                  # Processes MMASH into CSVs
├── train_single.py             # Trains model on single CSV
├── check_labels.py             # Checks label balance
├── scaler.pkl                  # Scaler for feature normalization
└── README.md                   # This file

Setup
Prerequisites

Python 3.10 (installed at c:\Users\USER\AppData\Local\Programs\Python\Python310)
MMASH dataset in MMASH/ (subfolders user_1 to user_22, each with Actigraph.csv, IBI.csv)

Installation

Install dependencies:
& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' -m pip install flask tensorflow pandas numpy scikit-learn flwr


Verify Flask:
& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' -m pip show flask



Generate Dataset
If dataset/user_1.csv to user_22.csv or scaler.pkl are missing:

Run dataset.py to process MMASH:& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' dataset.py


Outputs: dataset/user_1.csv to user_22.csv, scaler.pkl
Checks label balance to ensure balanced Stress/No Stress labels



Train Model
If models/global_model.weights.h5 is missing:

Run train_single.py:& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' train_single.py


Outputs: models/global_model.weights.h5, scaler.pkl



Run Web App

Start Flask app:& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' flask_app\app.py


Open http://127.0.0.1:5000 in a browser.

Usage

Home Page: Enter health metrics (Axis1, Axis2, Axis3, Vector Magnitude, RR) to predict stress.
Example:
No Stress: Axis1: 0.1, Axis2: 0.2, Axis3: 0.3, Vector Magnitude: 0.4, RR: 1000
Stress: Axis1: 0.9, Axis2: 0.9, Axis3: 0.9, Vector Magnitude: 0.9, RR: 600




About Page: Learn about Federated Learning and MMASH.
Contact Page: Team info for samAI hackathon.

Demo
See demo.mp4 for a 2–3 minute video showcasing:

Real-time stress predictions via the Flask app.
Navigation through Home, About, and Contact pages.
Privacy-preserving FL approach using Flower AI.

Technical Details

Dataset: MMASH processed into 22 CSVs with features (Axis1, Axis2, Axis3, Vector Magnitude, RR) and labels (Stress: SDNN < 65ms, No Stress: otherwise).
Model: TensorFlow neural network (64-32-1 layers, sigmoid output) trained on balanced data.
FL: Flower AI enables decentralized training (simulated on CSVs).
UI: Flask with responsive HTML/CSS, handling predictions via /predict endpoint.
Scaler: scaler.pkl normalizes features using StandardScaler.

Troubleshooting

Flask Error:& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' -m pip install flask


Missing CSVs/Scaler: Run dataset.py.
Missing Model Weights: Run train_single.py.
Prediction Bias: Run check_labels.py to verify label balance:& 'c:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe' check_labels.py


If imbalanced, adjust sdnn_threshold in dataset.py (e.g., 60 or 70) and rerun.



Acknowledgments

samAI Hackathon: For the opportunity to advance health AI.
MMASH Dataset: For providing rich health metrics.
Flower AI & TensorFlow: For enabling Federated Learning and model training.
Flask: For the web interface.

Contact

Email: stress.detection.team@example.com
samAI Hackathon 2025


Built with ❤️ for secure health monitoring.
