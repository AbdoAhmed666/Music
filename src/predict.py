# ==========================================
# 🎵 src/predict.py
# ==========================================
import joblib
import numpy as np
import pandas as pd
import os

MODEL_DIR = "models"

# -------------------------------
# 🔒 Secure file loading
# -------------------------------
def safe_load(path):
    """Safely load a file (returns None if the file does not exist)."""
    return joblib.load(path) if os.path.exists(path) else None

# -------------------------------
# 🔮 Main prediction function
# -------------------------------
def predict_from_features_dict(features_dict):
    """
    Makes a prediction from the given audio features (features_dict)
    and automatically handles the presence or absence of additional model files.
    """

    # File paths
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    reg_path = os.path.join(MODEL_DIR, "regressor.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "features.pkl")

    # Load files (if available)
    clf = safe_load(clf_path)
    reg = safe_load(reg_path)
    scaler = safe_load(scaler_path)
    feature_cols = safe_load(features_path)

    # Check if the main model exists
    if clf is None:
        return {"error": "⚠️ No classifier.pkl model found. Please train the model first."}

    # 🧩 Convert dict to DataFrame
    X = pd.DataFrame([features_dict])

    # Clean values in the data
    def clean_value(v):
        if isinstance(v, list):  # If the value is a list (e.g., MFCC)
            return np.mean(v)
        try:
            return float(v)
        except:
            return 0.0

    X = X.applymap(clean_value)

    # 🧹 Safely convert all columns to numeric values
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # 🔁 Reorder columns according to feature_cols
    if feature_cols is not None:
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_cols]

    # ⚖️ Apply the Scaler while keeping column names
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    # 🔮 Predict popularity (classification)
    class_pred = int(clf.predict(X)[0])
    prob = None
    try:
        prob = float(clf.predict_proba(X)[0].max())
    except:
        pass

    # 🔢 Predict numeric popularity (regression)
    popularity_pred = None
    if reg is not None:
        try:
            popularity_pred = float(reg.predict(X)[0])
        except:
            pass

    # 📦 Final result
    return {
        "is_popular": {
            "text": "✅ Popular" if class_pred == 1 else "❌ Not Popular",
            "value": bool(class_pred)
        },
        "probability": f"{prob * 100:.2f}%" if prob is not None else "Unavailable",
        "predicted_popularity": round(popularity_pred, 2) if popularity_pred is not None else "Unavailable",
        "used_features": {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in features_dict.items()
        }
    }
