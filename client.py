import flwr as fl
import numpy as np
import pandas as pd
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(user_csv):
    """Load and preprocess MMASH data for a user."""
    df = pd.read_csv(user_csv)
    df = df.dropna()
    X = df[["Axis1", "Axis2", "Axis3", "Vector Magnitude", "RR"]].values
    y = df["label"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, user_csv):
        self.input_shape = (5,)  # Axis1, Axis2, Axis3, Vector Magnitude, RR
        self.model = build_model(input_shape=self.input_shape, binary=True)
        self.x_train, self.x_test, self.y_train, self.y_test = load_data(user_csv)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide user CSV path, e.g., 'dataset/1001.csv'")
        sys.exit(1)
    user_csv = sys.argv[1]
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(user_csv))