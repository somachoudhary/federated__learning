import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import flwr as fl
from utils import preprocess_user_data

# 📌 Get user_id from command line
if len(sys.argv) < 2:
    print("⚠️ Please provide a user ID (e.g., user_1)")
    sys.exit(1)

user_id = sys.argv[1]
print(f"🚀 Starting client for user: {user_id}")

# 📂 Path to the combined CSV file for the user
file_path = f'dataset/{user_id}.csv'

# 🧠 Flower Client Definition
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        try:
            user_df = pd.read_csv(file_path)
            X, y = preprocess_user_data(user_df)
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            self.model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
        except Exception as e:
            print(f"❌ Error during client setup: {e}")
            sys.exit(1)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=32, verbose=1)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"✅ Test Accuracy for {user_id}: {accuracy:.4f}")
        return loss, len(self.x_test), {"accuracy": accuracy}

# 🌐 Start Flower Client
fl.client.start_client(server_address="127.0.0.1:8080", client=FLClient())
