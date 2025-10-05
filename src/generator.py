# src/generator.py
import os
from datetime import datetime
import numpy as np
import pretty_midi

# -------------------------------
# 🎵 Dynamic music generation based on features
# -------------------------------
def generate_dynamic(features, length_seconds=30):
    tempo = float(features.get("tempo", 100.0))
    key_shift = int(features.get("key", 0))
    energy = float(features.get("energy", 0.7))
    valence = float(features.get("valence", 0.5))
    acousticness = float(features.get("acousticness", 0.5))
    danceability = float(features.get("danceability", 0.5))
    mode = int(features.get("mode", 1))

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Instruments
    piano = pretty_midi.Instrument(program=0)     # Piano
    bass = pretty_midi.Instrument(program=33)     # Electric Bass
    drums = pretty_midi.Instrument(program=0, is_drum=True)

    beat_time = 60.0 / tempo

    # Choose scale (Major/Minor) based on mode and valence
    if mode == 0 or valence < 0.4:
        chords_base = [[57, 60, 64], [55, 58, 62], [53, 57, 60], [50, 53, 57]]  # minor-ish
    else:
        chords_base = [[60, 64, 67], [65, 69, 72], [55, 59, 62], [50, 53, 57]]  # major-ish

    chords = [[int(np.clip(n + key_shift, 0, 127)) for n in chord] for chord in chords_base]

    # 🥁 Drums affected by energy & danceability
    def add_drums(start, beats=4):
        cur = start
        for b in range(beats * 4):
            if energy > 0.3 and danceability > 0.4:
                drums.notes.append(pretty_midi.Note(60, 42, cur, cur + 0.05))  # hi-hat
            if b % 8 == 0 and energy > 0.2:
                drums.notes.append(pretty_midi.Note(100, 36, cur, cur + 0.1))  # Kick
            if b % 8 == 4 and energy > 0.3:
                drums.notes.append(pretty_midi.Note(90, 38, cur, cur + 0.1))   # Snare
            cur += beat_time / 4

    # -------------------------------
    # 🎹 Chords + Bass + Melody
    # -------------------------------
    cur_time = 0.0
    while cur_time < length_seconds:
        for chord in chords:
            # Chord duration is longer if acousticness is high or danceability is low
            chord_duration = beat_time * (6 if acousticness > 0.6 or danceability < 0.4 else 4)

            # 🎶 Chord
            chord_vel = int(np.clip(60 + energy * 30, 40, 120))
            for note in chord:
                piano.notes.append(pretty_midi.Note(chord_vel, note, cur_time, cur_time + chord_duration))

            # 🎸 Bass (softer if acousticness is high)
            if acousticness < 0.8:
                bass_note = chord[0] - 12
                for i in range(4):
                    st = cur_time + i * beat_time
                    en = st + beat_time
                    bass_velocity = int(np.clip(70 + energy * 40, 50, 120))
                    bass.notes.append(pretty_midi.Note(bass_velocity, bass_note, st, en))

            # 🎼 Melody (higher if valence is high, lower if low)
            mel_shift = 12 if valence > 0.5 else 0
            mel_note = chord[np.random.randint(0, len(chord))] + mel_shift
            mel_vel = int(np.clip(65 + energy * 40, 40, 127))
            piano.notes.append(pretty_midi.Note(mel_vel, mel_note, cur_time, cur_time + beat_time * 2))

            # 🥁 Drums
            add_drums(cur_time, beats=4)

            cur_time += chord_duration
            if cur_time >= length_seconds:
                break

    pm.instruments.extend([piano, bass, drums])
    return pm


# -------------------------------
# 🛠️ Function compatible with the API
# -------------------------------
def generate_from_features(features, length_seconds=30.0, out_dir="static/generated", render_wav=False, soundfont_path=None):
    pm = generate_dynamic(features, length_seconds)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    midi_filename = f"melody_{ts}.mid"
    midi_path = os.path.join(out_dir, midi_filename)
    pm.write(midi_path)

    wav_path = None
    if render_wav:
        sf = soundfont_path or os.environ.get("SOUNDFONT_PATH")
        if not sf or not os.path.exists(sf):
            raise RuntimeError("❌ Soundfont (.sf2) file not found.")
        try:
            from midi2audio import FluidSynth
        except Exception as e:
            raise RuntimeError("❌ The 'midi2audio' library is not installed. Install it using: pip install midi2audio") from e

        wav_filename = os.path.splitext(midi_filename)[0] + ".wav"
        wav_path = os.path.join(out_dir, wav_filename)

        fs = FluidSynth(sf)
        fs.midi_to_audio(midi_path, wav_path)

    return midi_path.replace("\\", "/"), (wav_path.replace("\\", "/") if wav_path else None)
