# src/audio_features.py
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf

def extract_audio_features(file_path, sr=22050, duration=60):
    """
    Load audio (first `duration` seconds) and extract features.
    Returns a dict of approximated features that match model inputs:
    - energy (approx RMS)
    - tempo
    - danceability (heuristic)
    - loudness (dB)
    - liveness (heuristic from onset strength)
    - valence (heuristic)
    - speechiness (heuristic from zcr + spectral_rolloff)
    - instrumentalness (heuristic from vocal energy proxy)
    - mode (0/1 guessed from spectral centroid average - not reliable)
    - key (not computed reliably -> set 0)
    - duration_ms
    - acousticness (heuristic)
    """
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    if y.size == 0:
        raise ValueError("Loaded audio is empty or invalid.")
    # Basic measures
    duration_ms = int(len(y) / sr * 1000)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    rmse = librosa.feature.rms(y=y).mean()  # energy proxy
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    # heuristics / approximations:
    energy = float(np.clip(rmse / 0.1, 0.0, 1.0))  # normalize assuming typical rmse ~0.02-0.2
    loudness_db = float(20 * np.log10(rmse + 1e-6))  # approximate loudness in dB (relative)
    # danceability proxy: depends on tempo and rhythmic clarity (onset strength)
    onset_strength_mean = float(onset_env.mean())
    danceability = float(np.clip((tempo / 200.0) * 0.6 + (onset_strength_mean / (onset_strength_mean + 1.0)) * 0.4, 0, 1))
    # liveness proxy: higher onset variance & spectral contrast -> more live feeling
    liveness = float(np.clip(spec_contrast / (spec_contrast + 50.0), 0, 1))
    # valence proxy: higher spectral centroid & tempo -> happier sounding (very rough)
    valence = float(np.clip((spectral_centroid / (spectral_centroid + 4000.0)) * 0.6 + (tempo / 200.0) * 0.4, 0, 1))
    # speechiness proxy: higher zcr & high spectral rolloff -> more speech-like
    speechiness = float(np.clip((zcr * 10.0) * 0.5 + (spectral_rolloff / 10000.0) * 0.5, 0, 1))
    # instrumentalness: if low voiced energy (low spectral centroid?) and low speechiness -> higher instrumentalness
    instrumentalness = float(np.clip(1.0 - speechiness - (spectral_centroid / (spectral_centroid + 8000.0)), 0, 1))
    # mode: crude guess (0 or 1) based on centroid (not reliable)
    mode = 1 if spectral_centroid > 2000 else 0
    key = 0  # not computed reliably here
    acousticness = float(np.clip(1.0 - (spectral_centroid / (spectral_centroid + 5000.0)), 0, 1))

    features = {
        "energy": round(energy, 4),
        "tempo": float(round(float(tempo) if not isinstance(tempo, (list, np.ndarray)) else tempo[0], 2)),
        "danceability": round(danceability, 4),
        "loudness": round(loudness_db, 3),
        "liveness": round(liveness, 4),
        "valence": round(valence, 4),
        "speechiness": round(speechiness, 4),
        "instrumentalness": round(instrumentalness, 4),
        "mode": int(mode),
        "key": int(key),
        "duration_ms": int(duration_ms),
        "acousticness": round(acousticness, 4),
        # extras (not used by model directly)
        "spectral_centroid": float(spectral_centroid),
        "zcr": float(zcr),
        "mfcc_mean": mfcc_mean.tolist()
    }
    return features

def save_waveform_plot(file_path, out_path="static/waveform.png"):
    """Save waveform image (matplotlib) to out_path."""
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def save_spectrogram_plot(file_path, out_path="static/spectrogram.png"):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.stft(y)
    Sdb = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram (dB)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
