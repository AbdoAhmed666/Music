# api.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os, uuid, joblib, pandas as pd
from src.audio_features import extract_audio_features, save_waveform_plot, save_spectrogram_plot
from src.predict import predict_from_features_dict
from src.generator import generate_from_features
from src.audio_separator import separate_audio
import os
import librosa
import pandas as pd

# App initialization
app = FastAPI(title="Spotify ML - Audio Upload & Prediction")

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/generated", exist_ok=True)
os.makedirs("models", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "models/classifier.pkl"
REG_PATH = "models/regressor.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/features.pkl"


# Home / Dashboard
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# Manual prediction form
@app.get("/predict-form", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request})


# Handle manual prediction (Form)
@app.post("/predict-manual", response_class=HTMLResponse)
async def predict_manual(request: Request,
                         energy: float = Form(...),
                         tempo: float = Form(...),
                         danceability: float = Form(...)):
    features = {
        "energy": energy, "tempo": tempo, "danceability": danceability,
    }
    try:
        pred = predict_from_features_dict(features)
    except Exception as e:
        pred = {"error": str(e), "used_features": features}

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": pred,
        "features": features,
        "waveform": None,
        "spectrogram": None
    })


# Upload audio → analyze → predict
@app.post("/upload-audio", response_class=HTMLResponse)
async def upload_audio(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())[:8]
    file_ext = os.path.splitext(file.filename)[1] or ".wav"
    save_path = os.path.join("uploads", f"{uid}{file_ext}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract audio features
    features = extract_audio_features(save_path)

    # ✅ Check if the song already exists in the local dataset
    csv_path = "data/songs_dataset.csv"
    found_in_db = False
    known_label = None

    if os.path.exists(csv_path):
        try:
            df_db = pd.read_csv(csv_path, on_bad_lines='skip')

            # Ensure numeric columns exist and are converted properly
            for col in ["tempo", "energy", "danceability"]:
                if col in df_db.columns:
                    df_db[col] = pd.to_numeric(df_db[col], errors="coerce").fillna(0)
                else:
                    df_db[col] = 0.0

            # Convert current song features to numeric
            tempo = float(features.get("tempo", 0) or 0)
            energy = float(features.get("energy", 0) or 0)
            danceability = float(features.get("danceability", 0) or 0)

            # Compare with tolerance to find matching record
            match = df_db[
                (abs(df_db["tempo"] - tempo) < 1.0) &
                (abs(df_db["energy"] - energy) < 0.02) &
                (abs(df_db["danceability"] - danceability) < 0.02)
            ]

            if not match.empty and (tempo > 0 and energy > 0 and danceability > 0):
                found_in_db = True
                if "popularity_label" in match.columns:
                    known_label = int(match["popularity_label"].iloc[0])
            else:
                found_in_db = False

        except Exception as e:
            print(f"⚠️ Database check error after conversion: {e}")

    # Generate waveform + spectrogram plots
    wf_path = os.path.join("static/generated", f"waveform_{uid}.png")
    sp_path = os.path.join("static/generated", f"spectrogram_{uid}.png")
    try:
        save_waveform_plot(save_path, out_path=wf_path)
        save_spectrogram_plot(save_path, out_path=sp_path)
    except Exception as e:
        print("Plot save error:", e)

    # Prediction
    try:
        if found_in_db and known_label is not None:
            pred = {
                "is_popular": "✅ Popular" if known_label == 1 else "❌ Not Popular",
                "probability": "100%",
                "predicted_popularity": known_label,
                "used_features": features,
                "from_database": True
            }
        else:
            pred = predict_from_features_dict(features)
            pred["from_database"] = False

        if "is_popular" not in pred:
            pred["is_popular"] = "❓ Unknown"
        if "probability" not in pred:
            pred["probability"] = "N/A"
        if "predicted_popularity" not in pred:
            pred["predicted_popularity"] = "N/A"

    except Exception as e:
        pred = {"error": str(e), "used_features": features, "from_database": False}

    return templates.TemplateResponse("result.html", {
        "request": request,
        "features": features,
        "prediction": pred,
        "waveform": "/" + wf_path.replace("\\", "/"),
        "spectrogram": "/" + sp_path.replace("\\", "/")
    })


# Add new song manually and retrain
@app.post("/add_song", response_class=HTMLResponse)
async def add_song(request: Request, file: UploadFile = File(...), popularity_label: int = Form(...)):
    import traceback
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1️⃣ Extract features
    try:
        from src.audio_features import extract_audio_features
        features = extract_audio_features(save_path)
        features["popularity_label"] = popularity_label
    except Exception as e:
        return HTMLResponse(content=f"❌ Error extracting features: {e}<br>{traceback.format_exc()}", status_code=500)

    # 2️⃣ Load old CSV and merge safely
    csv_path = "data/songs_dataset.csv"
    os.makedirs("data", exist_ok=True)

    try:
        new_df = pd.DataFrame([features])

        if os.path.exists(csv_path):
            try:
                old_df = pd.read_csv(csv_path, on_bad_lines='skip')
            except Exception as e:
                print(f"⚠️ Warning: Skipped bad lines while reading {csv_path}: {e}")
                old_df = pd.read_csv(csv_path, on_bad_lines='skip')

            all_cols = list(set(old_df.columns).union(set(new_df.columns)))
            old_df = old_df.reindex(columns=all_cols, fill_value=0)
            new_df = new_df.reindex(columns=all_cols, fill_value=0)

            full_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            full_df = new_df

        full_df.to_csv(csv_path, index=False)
    except Exception as e:
        return HTMLResponse(content=f"❌ Error saving dataset: {e}<br>{traceback.format_exc()}", status_code=500)

    # 3️⃣ Retrain model
    try:
        data = pd.read_csv(csv_path, on_bad_lines='skip')
        if "popularity_label" not in data.columns:
            return HTMLResponse(content="❌ Missing 'popularity_label' column in dataset.", status_code=500)

        X = data.drop(columns=["popularity_label"]).fillna(0)
        y = data["popularity_label"]

        import ast
        import numpy as np

        def clean_value(v):
            if isinstance(v, str):
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    try:
                        val = ast.literal_eval(v)
                        if isinstance(val, list) and len(val) == 1:
                            return float(val[0])
                        elif isinstance(val, (int, float)):
                            return float(val)
                    except:
                        return np.nan
                try:
                    return float(v)
                except:
                    return np.nan
            return v

        X = X.applymap(clean_value).fillna(0)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, "models/classifier.pkl")

        print("✅ Classifier retrained successfully after adding new song.")

    except Exception as e:
        return HTMLResponse(content=f"⚠️ Song saved but retraining failed: {e}<br>{traceback.format_exc()}", status_code=500)

    return templates.TemplateResponse("add_song_result.html", {
        "request": request,
        "filename": file.filename,
        "label": popularity_label
    })


# Retrain from main CSVs
@app.get("/train", response_class=HTMLResponse)
def train_model(request: Request):
    try:
        high_path = "data/high_popularity_spotify_data.csv"
        low_path = "data/low_popularity_spotify_data.csv"

        if not os.path.exists(high_path) or not os.path.exists(low_path):
            return HTMLResponse(content="❌ Missing training files (high/low).", status_code=400)

        df_high = pd.read_csv(high_path)
        df_low = pd.read_csv(low_path)

        if "popularity_label" not in df_high.columns:
            df_high["popularity_label"] = 1
        if "popularity_label" not in df_low.columns:
            df_low["popularity_label"] = 0

        df = pd.concat([df_high, df_low], ignore_index=True)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/spotify.csv", index=False)

        features_cols = ["energy", "tempo", "danceability"]
        for col in features_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[features_cols].fillna(0)
        y = df["popularity_label"]

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        joblib.dump(clf, "models/classifier.pkl")

        return templates.TemplateResponse("train.html", {"request": request, "score": f"{score:.4f}"})
    except Exception as e:
        return HTMLResponse(content=f"❌ Error during training: {str(e)}", status_code=500)


# Generate melody
@app.post("/generate-melody", response_class=HTMLResponse)
async def generate_melody(request: Request, file: UploadFile = File(...), length_sec: float = Form(30.0), render_wav: bool = Form(False)):
    uid = str(uuid.uuid4())[:8]
    file_ext = os.path.splitext(file.filename)[1] or ".wav"
    save_path = os.path.join("uploads", f"gen_{uid}{file_ext}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        features = extract_audio_features(save_path)
    except Exception as e:
        return HTMLResponse(content=f"Error extracting features: {e}", status_code=500)

    try:
        midi_path, wav_path = generate_from_features(features, length_seconds=length_sec, out_dir="static/generated", render_wav=render_wav, soundfont_path=os.environ.get("SOUNDFONT_PATH"))
    except Exception as e:
        return HTMLResponse(content=f"Error during melody generation: {e}", status_code=500)

    return templates.TemplateResponse("generate_result.html", {
        "request": request,
        "midi_path": "/" + midi_path,
        "wav_path": ("/" + wav_path) if wav_path else None,
        "features": features,
        "prompt": None
    })


# Audio separation
from fastapi import UploadFile, File

@app.post("/separate-audio", response_class=HTMLResponse)
async def separate_audio_route(request: Request, file: UploadFile = File(...)):
    if not file:
        return HTMLResponse(content="❌ No file uploaded", status_code=400)

    save_path = os.path.join("uploads", file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_paths = separate_audio(save_path)

    return templates.TemplateResponse("separate_result.html", {
        "request": request,
        "vocals_path": "/" + result_paths["vocals"],
        "drums_path": "/" + result_paths["drums"],
        "bass_path": "/" + result_paths["bass"],
        "other_path": "/" + result_paths["other"],
    })


# Song identification (ACRCloud)
from src.music_identify import identify_with_acr
from pydub import AudioSegment
import json

@app.get("/identify-form", response_class=HTMLResponse)
def identify_form(request: Request):
    return templates.TemplateResponse("identify.html", {"request": request})


@app.post("/identify", response_class=HTMLResponse)
async def identify_route(request: Request, file: UploadFile = File(...)):
    uid = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".webm"
    save_path = os.path.join("uploads", f"identify_{uid}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    static_dir = os.path.join("static", "uploads")
    os.makedirs(static_dir, exist_ok=True)
    static_path = os.path.join(static_dir, os.path.basename(save_path))
    shutil.copy(save_path, static_path)
    original_url = "/" + static_path.replace("\\", "/")

    wav_path = os.path.splitext(save_path)[0] + ".wav"
    try:
        audio = AudioSegment.from_file(save_path)
        shorter = audio[:15000]  # first 15 seconds
        shorter.export(wav_path, format="wav")
    except Exception as e:
        wav_path = save_path

    try:
        result = identify_with_acr(wav_path, offset=0, duration=10)
    except Exception as e:
        return templates.TemplateResponse("identify_result.html", {
            "request": request,
            "error": f"Identification error: {e}",
            "result": {"success": False, "raw": {}},
            "original_url": original_url
        })

    return templates.TemplateResponse("identify_result.html", {
        "request": request,
        "result": result,
        "original_url": original_url
    })
