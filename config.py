from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
GUITAR_DIR = BASE_DIR / "Guitar"
OUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# ── Audio ──────────────────────────────────────────────
SR = 44100
HOP_LENGTH = 512
SLICE_DURATION = 0.50          # seconds per note slice
ATTACK_MS = 40.0               # attack window for feature split
N_KEEP = 5                     # valid slices to keep per file

# ── YIN pitch tracking ────────────────────────────────
FMIN_YIN = 65.0
FMAX_YIN = 1200.0

# ── Onset detection ───────────────────────────────────
ONSET_BACKTRACK = True

# ── Validity filters ──────────────────────────────────
RMS_MIN = 0.001
RMS_MAX = 0.30
FLATNESS_MAX = 0.65

# ── Harmonic analysis ────────────────────────────────
N_FFT_HARM = 8192
N_HARM = 8
HARM_SEARCH_BW_HZ = 25.0

# ── Guitar tuning (standard, MIDI numbers) ───────────
OPEN_STRING_MIDI = {
    6: 40,   # E2
    5: 45,   # A2
    4: 50,   # D3
    3: 55,   # G3
    2: 59,   # B3
    1: 64,   # E4
}

MAX_FRET = 15
PITCH_TOL_CENTS = 40

# ── Training ─────────────────────────────────────────
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 2
