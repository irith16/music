#!/usr/bin/env python3
"""
Extract 45-dimensional embeddings from guitar note recordings.

Reads .wav files from Guitar/, writes embeddings + metadata to output/.
"""

import json
import numpy as np
import librosa

from config import SR, SLICE_DURATION, N_KEEP, GUITAR_DIR, OUT_DIR
from src.audio import list_wavs, parse_string_fret, extract_slices, is_valid_slice
from src.features import (
    midi_from_string_fret, note_name_from_midi, expected_f0_from_midi,
    compute_embedding, DIM_NAMES,
)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    wav_files = list_wavs(GUITAR_DIR)
    print(f"Found {len(wav_files)} wav files in {GUITAR_DIR}")

    all_embs = []
    all_labels = []
    all_meta = []
    low_valid = []

    for wf in wav_files:
        s, f = parse_string_fret(wf)
        midi = midi_from_string_fret(s, f)
        note = note_name_from_midi(midi)
        expected_f0 = expected_f0_from_midi(midi)

        y, sr = librosa.load(wf, sr=SR, mono=True)
        slices = extract_slices(y, sr, slice_duration=SLICE_DURATION)

        kept = 0
        for take_idx, (start_sample, y_slice) in enumerate(slices):
            if not is_valid_slice(y_slice, sr):
                continue

            emb, dbg = compute_embedding(y_slice, sr)
            label = f"S{s}_F{f}"

            all_embs.append(emb)
            all_labels.append(label)
            all_meta.append({
                "file": wf.name,
                "string": s,
                "fret": f,
                "label": label,
                "midi": midi,
                "note": note,
                "expected_f0_hz": expected_f0,
                "onset_sample": int(start_sample),
                "onset_time_sec": round(start_sample / sr, 4),
                "slice_duration_sec": SLICE_DURATION,
                "take_index": int(take_idx),
                "f0_hz": float(dbg["f0_hz"]),
            })

            kept += 1
            if kept >= N_KEEP:
                break

        if kept < N_KEEP:
            low_valid.append({"file": wf.name, "kept": kept})

    # Save
    X = np.vstack(all_embs).astype(float)
    labels = np.array(all_labels)

    np.save(OUT_DIR / "embeddings.npy", X)
    np.save(OUT_DIR / "labels_pos.npy", labels)
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)
    with open(OUT_DIR / "dimensions.json", "w") as f:
        json.dump({"dim_count": len(DIM_NAMES), "dim_names": DIM_NAMES}, f, indent=2)

    print(f"Saved {X.shape[0]} embeddings  ({X.shape[1]} dims)  →  {OUT_DIR}/")
    if low_valid:
        print(f"  ⚠  {len(low_valid)} files had fewer than {N_KEEP} valid slices")


if __name__ == "__main__":
    main()
