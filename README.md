# Federated Learning for Stress Detection
## Overview
Privacy-preserving stress detection using MMASH, Flower, and Flask.
## Setup
1. Install: `pip install flwr tensorflow pandas sklearn matplotlib flask`
2. Run: `python dataset.py`
3. Run FL:
   - `Start-Process python -ArgumentList "server.py"`
   - `1..22 | ForEach-Object { if ($_ -le 11) { Start-Process python -ArgumentList "client.py", "dataset\user_1.csv" } else { Start-Process python -ArgumentList "client.py", "dataset\user_2.csv" } }`
4. Visualize: `python plot_results.py`
5. Run web app: `python flask_app/app.py`, visit http://127.0.0.1:5000
## Demo
Input health data (Axis1, Axis2, Axis3, Vector Magnitude, RR Interval) to predict stress.