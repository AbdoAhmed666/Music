# src/audio_separator.py
import os
import subprocess
from datetime import datetime

def separate_audio(file_path, out_dir="static/separated"):
    """
    Separates the song into 4 files:
    - vocals.wav (singer's voice)
    - drums.wav (drums)
    - bass.wav (bass)
    - other.wav (remaining music)
    """
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    target_dir = os.path.join(out_dir, f"song_{ts}")
    os.makedirs(target_dir, exist_ok=True)

    # Run demucs
    cmd = [
        "demucs", "-n", "htdemucs", "-o", target_dir, file_path
    ]
    subprocess.run(cmd, check=True)

    # Demucs places the separated files inside a folder named after the model (htdemucs)
    demucs_dir = os.path.join(
        target_dir,
        "htdemucs",
        os.path.basename(file_path).replace(".mp3", "").replace(".wav", "")
    )
    
    result = {
        "vocals": os.path.join(demucs_dir, "vocals.wav").replace("\\", "/"),
        "drums": os.path.join(demucs_dir, "drums.wav").replace("\\", "/"),
        "bass": os.path.join(demucs_dir, "bass.wav").replace("\\", "/"),
        "other": os.path.join(demucs_dir, "other.wav").replace("\\", "/"),
    }

    return result
