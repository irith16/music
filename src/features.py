"""
45-dimensional embedding for guitar note identification.

Dimensions:
    0–11   chroma bins (L1-normalised)
    12     f0 / 1000  (YIN median)
    13     spectral centroid — attack  (normalised by Nyquist)
    14     spectral centroid — steady  (normalised by Nyquist)
    15–22  harmonic deviation × 1000  (k = 1 … 8)
    23–30  harmonic energy ratios      (k = 1 … 8)
    31     inharmonicity coefficient B (scaled to 0–1)
    32–44  MFCC attack means           (13 coefficients)
"""

import numpy as np
import librosa

from config import (
    SR, HOP_LENGTH, ATTACK_MS,
    FMIN_YIN, FMAX_YIN,
    N_FFT_HARM, N_HARM, HARM_SEARCH_BW_HZ,
    OPEN_STRING_MIDI,
)
from src.audio import slice_attack_steady


# ── Music theory helpers ─────────────────────────────────────────────

def midi_from_string_fret(string_num, fret):
    return OPEN_STRING_MIDI[string_num] + fret


def note_name_from_midi(m):
    return librosa.midi_to_note(m)


def expected_f0_from_midi(m):
    return float(librosa.midi_to_hz(m))


# ── Pitch tracking ───────────────────────────────────────────────────

def compute_f0_series(y, sr=SR):
    """Return all finite YIN f0 estimates (Hz) over time."""
    try:
        f0 = librosa.yin(y, fmin=FMIN_YIN, fmax=FMAX_YIN, sr=sr)
        return f0[np.isfinite(f0)]
    except Exception:
        return np.array([])


# ── Spectral helpers ─────────────────────────────────────────────────

def stft_mag(y, sr=SR, n_fft=N_FFT_HARM, hop_length=HOP_LENGTH):
    """Return (frequency_bins, time-averaged magnitude)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            window="hann", center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mag = np.mean(S, axis=1)
    return freqs, mag


def spectral_centroid_from_mag(freqs, mag):
    s = float(np.sum(mag))
    if s <= 1e-12:
        return 0.0
    return float(np.sum(freqs * mag) / s)


def parabolic_peak_refine(freqs, mag, idx):
    """Refine a peak location via parabolic interpolation in log-mag."""
    if idx <= 0 or idx >= len(mag) - 1:
        return float(freqs[idx]), float(mag[idx])
    y0 = np.log(mag[idx - 1] + 1e-12)
    y1 = np.log(mag[idx]     + 1e-12)
    y2 = np.log(mag[idx + 1] + 1e-12)
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[idx]), float(mag[idx])
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    f_hat = freqs[idx] + delta * (freqs[1] - freqs[0])
    return float(f_hat), float(mag[idx])


# ── Harmonic analysis ────────────────────────────────────────────────

def harmonic_peaks(freqs, mag, f0, n_harm=N_HARM, bw_hz=HARM_SEARCH_BW_HZ):
    """Find frequency and amplitude of the first *n_harm* partials."""
    f_hats = np.zeros(n_harm)
    a_hats = np.zeros(n_harm)
    if f0 <= 0:
        return f_hats, a_hats

    for k in range(1, n_harm + 1):
        target = k * f0
        lo, hi = target - bw_hz, target + bw_hz
        if hi <= freqs[0] or lo >= freqs[-1]:
            continue
        i0 = int(np.searchsorted(freqs, max(lo, freqs[0])))
        i1 = int(np.searchsorted(freqs, min(hi, freqs[-1])))
        if i1 <= i0 + 2:
            idx = int(np.argmin(np.abs(freqs - target)))
        else:
            idx = i0 + int(np.argmax(mag[i0:i1]))
        f_hats[k - 1], a_hats[k - 1] = parabolic_peak_refine(freqs, mag, idx)

    return f_hats, a_hats


def harmonic_deviation(f0, f_hats, n_harm=N_HARM):
    """dev_k = (f_hat_k / (k·f0)) − 1  for k = 1 … n_harm."""
    dev = np.zeros(n_harm)
    if f0 <= 0:
        return dev
    for k in range(1, n_harm + 1):
        fk = f_hats[k - 1]
        denom = k * f0
        if fk > 0 and denom > 0:
            dev[k - 1] = (fk / denom) - 1.0
    return dev


def harmonic_energy_ratios(a_hats, n_harm=N_HARM):
    a = a_hats[:n_harm].copy()
    s = float(np.sum(a))
    if s <= 1e-12:
        return np.zeros(n_harm)
    return (a / s).astype(float)


def estimate_inharmonicity(dev, harm_energy=None, n_harm=N_HARM):
    """
    Estimate the inharmonicity coefficient B from harmonic deviations.

    Fits  dev_k ≈ a · k²  for k = 2 … 8, then B_est = 2a.
    Returns B scaled to [0, 1] via clip(B, 0, 0.01) / 0.01.
    """
    ks = np.arange(1, n_harm + 1, dtype=float)
    use = ks >= 2
    x = ks[use] ** 2
    y = dev[use].astype(float)

    w = np.ones_like(x) if harm_energy is None else harm_energy[use].astype(float)
    if np.sum(w) <= 1e-12:
        w = np.ones_like(x)

    den = float(np.sum(w * x ** 2))
    B_est = 0.0 if den <= 1e-12 else 2.0 * float(np.sum(w * x * y)) / den
    return float(np.clip(B_est, 0.0, 0.01)) / 0.01


# ── Embedding dimension schema ───────────────────────────────────────

def build_dimension_schema():
    names = []
    for b in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
        names.append(f"chroma_{b}")
    names.append("f0_norm")
    names.append("centroid_attack_norm")
    names.append("centroid_steady_norm")
    for k in range(1, N_HARM + 1):
        names.append(f"harm_dev_k{k}")
    for k in range(1, N_HARM + 1):
        names.append(f"harm_energy_ratio_k{k}")
    names.append("B_est_scaled")
    for i in range(13):
        names.append(f"mfcc_attack_{i}")
    assert len(names) == 45
    return names


DIM_NAMES = build_dimension_schema()


# ── Main embedding function ──────────────────────────────────────────

def compute_embedding(y_slice, sr=SR):
    """
    Compute the 45-dimensional embedding for a single note slice.

    Returns (embedding, debug_dict).
    """
    # Pitch
    f0_series = compute_f0_series(y_slice, sr)
    f0 = float(np.median(f0_series)) if len(f0_series) else 0.0

    # Attack / steady split
    y_attack, y_steady = slice_attack_steady(y_slice, sr, attack_ms=ATTACK_MS)

    # 0–11: chroma (from steady portion, L1-normalised)
    chroma = np.mean(librosa.feature.chroma_cqt(y=y_steady, sr=sr), axis=1)
    s = float(np.sum(chroma))
    if s > 1e-12:
        chroma /= s

    # Spectral centroids
    freqs_a, mag_a = stft_mag(y_attack, sr)
    freqs_s, mag_s = stft_mag(y_steady, sr)
    cent_a = spectral_centroid_from_mag(freqs_a, mag_a)
    cent_s = spectral_centroid_from_mag(freqs_s, mag_s)

    # Harmonic features (from steady)
    if f0 > 0:
        f_hats, a_hats = harmonic_peaks(freqs_s, mag_s, f0)
        dev = harmonic_deviation(f0, f_hats)
        hE = harmonic_energy_ratios(a_hats)
        B_scaled = estimate_inharmonicity(dev, harm_energy=hE)
    else:
        dev = np.zeros(N_HARM)
        hE = np.zeros(N_HARM)
        B_scaled = 0.0

    # 32–44: MFCC attack
    mfcc = np.mean(librosa.feature.mfcc(y=y_attack, sr=sr, n_mfcc=13), axis=1)

    # Normalise and concatenate
    emb = np.concatenate([
        chroma,                                                        # 12
        [np.clip(f0 / 1000.0, 0.0, 1.5)],                             # 1
        [np.clip(cent_a / (sr / 2), 0.0, 1.0)],                       # 1
        [np.clip(cent_s / (sr / 2), 0.0, 1.0)],                       # 1
        dev * 1000.0,                                                  # 8
        hE,                                                            # 8
        [B_scaled],                                                    # 1
        mfcc,                                                          # 13
    ]).astype(float)

    assert emb.shape == (45,), f"Expected 45 dims, got {emb.shape[0]}"
    return emb, {"f0_hz": f0, "B_est_scaled": B_scaled}
