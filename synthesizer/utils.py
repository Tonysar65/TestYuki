# synthesizer/utils.py
import librosa
import numpy as np

def adjust_pitch(mel, shift_semitones: float):
    return librosa.effects.pitch_shift(mel, sr=22050, n_steps=shift_semitones)

def adjust_speed(mel, rate: float):
    return librosa.effects.time_stretch(mel, rate)

def detect_language(text: str) -> str:
    if any(c in text for c in "あいうえお"):
        return "ja"
    elif any(c in text for c in "àèìòù"):
        return "it"
    return "en"
