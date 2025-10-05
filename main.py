from src.data_loader import load_data
from src.preprocessing import preprocess
from src.analysis import run_analysis
from src.train import train_models
from src.predict import predict

def main():
    print("📥 Loading Data...")
    df = load_data()

    print("🔍 Running Analysis...")
    run_analysis(df)

    print("⚙️ Preprocessing...")
    (X_train_c, X_test_c, y_train_c, y_test_c,
     X_train_r, X_test_r, y_train_r, y_test_r,
     scaler, features) = preprocess(df)

    print("🎯 Training Models...")
    train_models(X_train_c, X_test_c, y_train_c, y_test_c,
                 X_train_r, X_test_r, y_train_r, y_test_r,
                 scaler, features)

    print("✅ Training Complete!")

    # prediction
    sample_song = {
        "energy": 0.7, "tempo": 120, "danceability": 0.8,
        "loudness": -5, "liveness": 0.2, "valence": 0.6,
        "speechiness": 0.05, "instrumentalness": 0.0,
        "mode": 1, "key": 5, "duration_ms": 210000, "acousticness": 0.3
    }
    result = predict(sample_song)
    print("🔮 Prediction:", result)

if __name__ == "__main__":
    main()
