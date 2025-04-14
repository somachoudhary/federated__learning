import pandas as pd
df1 = pd.read_csv("dataset/user_1.csv")
df2 = pd.read_csv("dataset/user_2.csv")
print("User 1 labels:", df1["label"].value_counts())
print("User 2 labels:", df2["label"].value_counts())