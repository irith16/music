"""Audio loading, onset detection, slicing, and validity filtering."""

import numpy as np
import librosa

from config import (
    SR, HOP_LENGTH, SLICE_DURATION, ATTACK_MS,
    ONSET_BACKTRACK, RMS_MIN, RMS_MAX, FLATNESS_MAX,
    FMIN_YIN, FMAX_YIN,
)


def list_wavs(folder):
    """Return sorted list of .wav paths in *folder*."""
    return sorted(folder.glob("*.wav"))


def parse_string_fret(path):
    """
    Extract (string, fret) from filename.

    Expected format: S{string}_F{fret}.wav
    Example:  S6_F15.wav  → (6, 15)
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 2 or not parts[0].startswith("S") or not parts[1].startswith("F"):
        raise ValueError(f"Bad filename format: {path.name} (expected Sx_Fy.wav)")
    return int(parts[0][1:]), int(parts[1][1:])


def detect_onsets(y, sr=SR):
    """Return onset positions in samples."""
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=HOP_LENGTH,
        backtrack=ONSET_BACKTRACK,
        units="frames",
    )
    return librosa.frames_to_samples(onset_frames, hop_length=HOP_LENGTH)


def extract_slices(y, sr=SR, slice_duration=SLICE_DURATION):
    """Detect onsets and return list of (start_sample, y_slice) tuples."""
    onsets = detect_onsets(y, sr)
    win = int(slice_duration * sr)
    slices = []
    for o in onsets:
        start = int(o)
        end = start + win
        if end > len(y):
            if len(y) - start < 0.8 * win:
                continue
            end = len(y)
        slices.append((start, y[start:end]))
    return slices


def is_valid_slice(y_slice, sr=SR):
    """
    Gate out silence, noise, and unpitched sounds.

    Intentionally permissive — this keeps anything that looks like
    a guitar note, even quiet or slightly buzzy ones.
    """
    # RMS gate
    rms = float(np.sqrt(np.mean(y_slice ** 2)))
    if rms < RMS_MIN or rms > RMS_MAX:
        return False

    # Spectral flatness — reject broadband noise
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y_slice)))
    if flat > FLATNESS_MAX:
        return False

    # Pitch existence — YIN must find *some* pitch
    f0_vals = librosa.yin(y_slice, fmin=FMIN_YIN, fmax=FMAX_YIN, sr=sr)
    f0_vals = f0_vals[np.isfinite(f0_vals)]
    if len(f0_vals) == 0:
        return False

    return True


def slice_attack_steady(y_slice, sr=SR, attack_ms=ATTACK_MS):
    """Split a note slice into attack and steady-state portions."""
    n_attack = int((attack_ms / 1000.0) * sr)
    n_attack = max(128, min(n_attack, len(y_slice)))
    y_attack = y_slice[:n_attack]
    y_steady = y_slice[n_attack:] if len(y_slice) > n_attack else y_slice
    return y_attack, y_steady
