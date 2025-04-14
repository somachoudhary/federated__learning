import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_user_data(user_df):
    """
    Takes in a user dataframe and returns preprocessed features (X) and labels (y).
    Assumes the label is in the last column.
    """
    if user_df is None or user_df.empty:
        raise ValueError("User DataFrame is empty or None.")

    # Drop missing values
    user_df = user_df.dropna()

    # Assume label is in last column
    X = user_df.iloc[:, :-1].values
    y = user_df.iloc[:, -1].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
