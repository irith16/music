# Guitar Note → Tablature Position Identification

Most guitar transcription tools detect **what note** is playing. This project detects **where on the fretboard** it's being played — the same pitch (e.g. A2 = 110 Hz) can be produced at string 6 fret 5, string 5 fret 0, or string 4 fret 7, and they each sound subtly different. Resolving this ambiguity is what makes automatic tablature generation hard.

This project solves it by combining **domain-informed audio feature engineering** with a **pitch-constrained two-stage classifier**, achieving **83.3% string+fret accuracy** on held-out data across 96 unique positions (6 strings × 16 frets).

---

## Why This Is Hard

A guitar in standard tuning has massive pitch overlap between strings. Of the 96 playable positions (frets 0–15 on 6 strings), many share the exact same fundamental frequency. A pure pitch detector cannot distinguish them.

The difference lies in **timbre** — the relative strength of harmonics, the inharmonicity of the string, the attack transient, and the spectral envelope. These vary by string gauge, scale length, and fret position. This project extracts those timbral cues into a compact embedding and uses them alongside pitch to identify the exact position.

---

## Approach

### 1. Audio → 45-Dimensional Embedding

Each detected note onset is sliced into a 500ms window and split into **attack** (first 40ms) and **steady-state** portions. Features are computed from each:

| Dims | Feature | Source | Why it helps |
|------|---------|--------|-------------|
| 0–11 | Chroma bins (L1-normalised) | Steady | Pitch class identity |
| 12 | f0 (YIN, normalised) | Full slice | Absolute pitch |
| 13–14 | Spectral centroid (attack & steady) | Both | Brightness / string gauge |
| 15–22 | Harmonic deviation (×1000) | Steady | Partial stretching patterns |
| 23–30 | Harmonic energy ratios | Steady | Timbre fingerprint per position |
| 31 | Inharmonicity coeff. B | Steady | String stiffness (thick vs thin) |
| 32–44 | MFCCs (attack) | Attack | Pluck transient shape |

**Inharmonicity (B)** is the most physically grounded feature. Real strings are stiff, so their partials are stretched above perfect integer multiples of f0. The amount of stretch follows B·k², where B depends on string material and diameter. Thick low strings (string 6) have higher B than thin high strings (string 1). This is estimated via weighted least-squares regression on the measured harmonic deviations.

### 2. Two-Stage Classification

Rather than a single 96-class classifier, the problem is decomposed:

1. **String classifier** — 6-class Random Forest on the full embedding
2. **Per-string fret classifiers** — one RF per string, trained only on that string's samples

This decomposition reflects the physics: strings differ primarily in timbre (captured by harmonics, MFCCs, centroid), while frets within a string differ primarily in pitch.

### 3. Pitch-Constrained Inference

At prediction time, the YIN pitch estimate is used to **eliminate impossible frets** — if f0 = 220 Hz and we're considering string 5 (open A2 = 110 Hz), only fret 12 (A3 = 220 Hz) is within tolerance (±40 cents). This dramatically shrinks the search space.

The final prediction maximises:

```
score(s, f) = log P_string(s) + log P_fret(f | s)
```

subject to the pitch constraint.

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐
│ Audio In │────▶│  45-dim Emb  │────▶│ String CLF  │──▶ top-K strings
└──────────┘     └──────────────┘     └─────────────┘
                                             │
                        ┌────────────────────┤
                        ▼                    ▼
                 ┌─────────────┐     ┌──────────────┐
                 │  YIN f0 Hz  │────▶│ Pitch Filter │──▶ valid frets
                 └─────────────┘     └──────────────┘
                                             │
                                             ▼
                                     ┌──────────────┐
                                     │  Fret CLFs   │──▶ best (string, fret)
                                     └──────────────┘
```

---

## Results

Evaluated on a held-out 25% test split (stratified by string):

| Metric | Value |
|--------|-------|
| **Full string+fret accuracy** | **83.3%** |
| String classifier accuracy | 80.0% |

Per-string fret accuracy (isolated):

| String | Fret Acc | Note |
|--------|----------|------|
| 6 (low E) | 95% | Thick string → distinct harmonics |
| 5 (A) | 95% | |
| 4 (D) | 75% | |
| 3 (G) | 35% | Thinnest wound string — transitional timbre |
| 2 (B) | 55% | |
| 1 (high E) | 45% | Thin plain string → less harmonic variation |

The accuracy gradient from low → high strings matches physical expectations: thicker wound strings produce more distinctive harmonic signatures per fret, while thin plain strings sound more uniform across positions.

---

## Dataset

96 `.wav` recordings of isolated guitar notes in standard tuning, one per string+fret combination (strings 1–6, frets 0–15). Each file contains multiple strikes of the same note. Onset detection extracts up to 5 valid slices per file, yielding ~480 training samples total.

Filenames follow the format `S{string}_F{fret}.wav` (e.g. `S6_F15.wav`).

---

## Usage

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

**Step 1 — Extract features** (requires wav files in `Guitar/`):
```bash
python extract.py
```

**Step 2 — Train classifiers:**
```bash
python train.py
```

**Step 3 — Evaluate:**
```bash
python evaluate.py
```

---

## Repo Structure

```
├── config.py           # All hyperparameters and paths
├── extract.py          # Feature extraction pipeline
├── train.py            # Model training + per-string evaluation
├── evaluate.py         # Held-out test set evaluation
├── src/
│   ├── audio.py        # Onset detection, slicing, validity
│   ├── features.py     # 45-dim embedding, harmonic analysis
│   └── pitch.py        # Pitch-constrained two-stage inference
├── requirements.txt
└── .gitignore
```

---

## Limitations & Future Work

- **Small dataset** — 480 samples across 96 classes is tight. Data augmentation (pitch shift, noise injection, gain variation) would help the weak strings significantly.
- **Strings 1–3 are undertrained** — plain (unwound) strings have less harmonic variation between frets, making them inherently harder. More recordings and richer features (e.g. string resonance, sustain decay rate) could help.
- **Monophonic only** — this handles single isolated notes. Extending to polyphonic audio (chords, simultaneous strings) is a different and much harder problem.
- **Single guitar** — the model is trained on one instrument. Generalising across guitars with different pickups, body shapes, and string brands would require a larger and more diverse dataset.
- **No real-time inference** — currently batch-only. Wrapping the pipeline in a streaming audio callback (e.g. via `sounddevice`) would enable a live fretboard tracker.

---

## License

MIT
