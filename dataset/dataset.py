import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compute_sdnn(rr_series, window_size=300):
    """Calculate SDNN (std dev of RR intervals) over a sliding window."""
    sdnn = []
    for i in range(0, len(rr_series) - window_size + 1):
        window = rr_series[i:i + window_size]
        sdnn.append(np.std(window) if len(window) > 0 else np.nan)
    return np.array(sdnn)

def preprocess_user_data(user_df):
    """Split and scale data (for client use later)."""
    user_df = user_df.dropna()
    X = user_df.iloc[:, :-1].values
    y = user_df.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_test, y_test)

# Set the base path to MMASH dataset
base_path = "dataset/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/MMASH/DataPaper"
print(f"📍 Checking base path: {base_path}")
if not os.path.exists(base_path):
    print(f"❌ Error: Base path {base_path} does not exist!")
    exit(1)

mmash_data = {}

# Load all CSVs into a dictionary: {user_id: {file_name: dataframe}}
for user_folder in os.listdir(base_path):
    user_path = os.path.join(base_path, user_folder)
    print(f"📁 Checking: {user_path}")
    if os.path.isdir(user_path):
        user_data = {}
        for file in os.listdir(user_path):
            print(f"  📄 Found file: {file}")
            if file.endswith(".csv"):
                file_path = os.path.join(user_path, file)
                print(f"    📥 Trying to read: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    user_data[file] = df
                    print(f"    ✅ Loaded: {file} (rows: {len(df)})")
                except Exception as e:
                    print(f"    ❌ Error loading {file_path}: {e}")
        if user_data:
            mmash_data[user_folder] = user_data

print(f"\n✅ Loaded data for {len(mmash_data)} users.")
print("👤 Users:", list(mmash_data.keys()))

# Save preprocessed combined data as one file per user
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

for user_id, files_dict in mmash_data.items():
    print(f"🔄 Processing user: {user_id}")
    try:
        actigraph = files_dict.get("Actigraph.csv")
        rr = files_dict.get("RR.csv")

        if actigraph is not None and rr is not None:
            print(f"  📊 Actigraph rows: {len(actigraph)}, RR rows: {len(rr)}")
            actigraph = actigraph.dropna()
            rr = rr.dropna()

            # Select Actigraph features
            actigraph_features = actigraph[["Axis1", "Axis2", "Axis3", "Vector Magnitude"]]
            min_len = len(actigraph_features)

            # Align RR data
            rr_feature = rr.iloc[:min_len, 1]  # Assume 2nd column is RR interval
            actigraph_features = actigraph_features.iloc[:min_len].copy()
            actigraph_features["RR"] = rr_feature.values

            # Compute SDNN for stress labels
            sdnn = compute_sdnn(rr_feature.values)
            if len(sdnn) < len(actigraph_features):
                sdnn = np.pad(sdnn, (0, len(actigraph_features) - len(sdnn)), mode='edge')
            else:
                sdnn = sdnn[:len(actigraph_features)]
                actigraph_features = actigraph_features.iloc[:len(sdnn)]

            # Assign labels: SDNN < 50ms = stress (1), else no stress (0)
             actigraph_features["label"] = (sdnn < 50).astype(int)
            # Save file
            save_path = os.path.join(output_dir, f"{user_id}.csv")
            actigraph_features.to_csv(save_path, index=False)
            print(f"✅ Saved combined data with labels: {save_path} (rows: {len(actigraph_features)})")
        else:
            print(f"⚠️ Missing Actigraph.csv or RR.csv for user {user_id}. Skipping...")

    except Exception as e:
        print(f"❌ Error processing user {user_id}: {e}")

print("\n📂 Checking output directory:")
for f in os.listdir(output_dir):
    if f.endswith(".csv"):
        print(f"  ✅ Found: {f}")
from sklearn.preprocessing import StandardScaler
import pickle
# In data loading (e.g., get_data):
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)