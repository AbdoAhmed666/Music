import pandas as pd
import os

def load_data(data_dir="data"):
    high_path = os.path.join(data_dir, "high_popularity_spotify_data.csv")
    low_path = os.path.join(data_dir, "low_popularity_spotify_data.csv")

    high_df = pd.read_csv(high_path)
    low_df = pd.read_csv(low_path)

    high_df["popularity_label"] = 1   # Popular
    low_df["popularity_label"] = 0   # Not Popular

    df = pd.concat([high_df, low_df], ignore_index=True)
    return df
