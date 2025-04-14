import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Preprocess function (optional – if you need it later for training)
def preprocess_user_data(user_df):
    user_df = user_df.dropna()
    X = user_df.iloc[:, :-1].values
    y = user_df.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (x_train, y_train), (x_test, y_test)

# Set the base path to the MMASH dataset directory
base_path = "dataset/multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0/MMASH/DataPaper"
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
                    print(f"    ✅ Loaded: {file}")
                except Exception as e:
                    print(f"    ❌ Error loading {file_path}: {e}")
        
        if user_data:
            mmash_data[user_folder] = user_data

print(f"\n✅ Loaded data for {len(mmash_data)} users.")
print("👤 Users:", list(mmash_data.keys()))

# Optional preview
if mmash_data:
    first_user = list(mmash_data.keys())[0]
    print(f"\n📂 Data files for first user ({first_user}):", list(mmash_data[first_user].keys()))
    
    if mmash_data[first_user]:
        first_file = list(mmash_data[first_user].keys())[0]
        print(f"\n📄 Sample file: {first_file}")
        print(mmash_data[first_user][first_file].head())
    else:
        print("⚠️ No CSV files found for this user.")
else:
    print("⚠️ No users with data found.")

# Save preprocessed combined data as one file per user (e.g., dataset/1001.csv)
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Save processed Actigraph + RR features per user
for user_id, files_dict in mmash_data.items():
    try:
        actigraph = files_dict.get("Actigraph.csv")
        rr = files_dict.get("RR.csv")  # Optional

        if actigraph is not None:
            actigraph = actigraph.dropna()

            # Select columns of interest from Actigraph
            actigraph_features = actigraph[["Axis1", "Axis2", "Axis3", "Vector Magnitude"]]
            min_len = len(actigraph_features)

            # Add RR feature if available
            if rr is not None:
                rr = rr.dropna()
                rr_feature = rr.iloc[:min_len, 1]  # Assume 2nd column is RR interval
                actigraph_features = actigraph_features.iloc[:min_len]
                actigraph_features["RR"] = rr_feature.values

            # Add dummy label
            actigraph_features["label"] = [0] * len(actigraph_features)

            # Save file
            save_path = os.path.join(output_dir, f"{user_id}.csv")
            actigraph_features.to_csv(save_path, index=False)
            print(f"✅ Saved combined data: {save_path}")
        else:
            print(f"⚠️ Actigraph.csv not found for user {user_id}. Skipping...")

    except Exception as e:
        print(f"❌ Error processing user {user_id}: {e}")
