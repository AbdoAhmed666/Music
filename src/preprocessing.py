from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd

def preprocess(df, balance=True):
    features = [
        "energy", "tempo", "danceability", "loudness",
        "liveness", "valence", "speechiness", "instrumentalness",
        "mode", "key", "duration_ms", "acousticness"
    ]

    X = df[features]
    y_class = df["popularity_label"]
    y_reg = df["track_popularity"]

    # NaN
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=features)

    #  Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------
    # 🔹Classification
    # ------------------------
    X_class, y_class_proc = X_scaled, y_class

    if balance:  # SMOTE
        smote = SMOTE(random_state=42)
        X_class, y_class_proc = smote.fit_resample(X_class, y_class_proc)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class_proc, test_size=0.2, random_state=42
    )

    # ------------------------
    # 🔹Regression
    # ------------------------
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )

    return (X_train_c, X_test_c, y_train_c, y_test_c,
            X_train_r, X_test_r, y_train_r, y_test_r,
            scaler, features)
