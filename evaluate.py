#!/usr/bin/env python3
"""
Evaluate the full string + fret pipeline on the held-out test set.

Uses pitch-constrained inference and reports per-string breakdowns.
"""

import json
import joblib
import numpy as np

from config import OUT_DIR, MODELS_DIR
from src.pitch import predict_string_fret


def main():
    # Load data
    X = np.load(OUT_DIR / "embeddings.npy")
    labels = np.load(OUT_DIR / "labels_pos.npy")
    with open(OUT_DIR / "meta.json") as f:
        meta = json.load(f)

    y_string, y_fret = [], []
    for lab in labels:
        s, fr = lab.split("_")
        y_string.append(int(s[1:]))
        y_fret.append(int(fr[1:]))
    y_string = np.array(y_string)
    y_fret = np.array(y_fret)

    # Load models + test indices
    string_clf = joblib.load(MODELS_DIR / "string_clf.pkl")
    fret_models = {}
    for s in np.unique(y_string):
        fret_models[s] = joblib.load(MODELS_DIR / f"fret_model_S{s}.pkl")
    idx_te = np.load(MODELS_DIR / "test_indices.npy")

    # Evaluate on held-out set only
    correct = 0
    total = len(idx_te)
    per_string = {}

    for i in idx_te:
        true_s, true_f = y_string[i], y_fret[i]
        f0 = meta[i].get("f0_hz", 0.0)

        pred_s, pred_f, _ = predict_string_fret(
            X[i], f0, string_clf, fret_models, top_k_strings=3
        )

        hit = (pred_s == true_s and pred_f == true_f)
        if hit:
            correct += 1

        per_string.setdefault(true_s, {"correct": 0, "total": 0})
        per_string[true_s]["total"] += 1
        if hit:
            per_string[true_s]["correct"] += 1

    print(f"Full string+fret accuracy (test, n={total}): {correct / total:.3f}\n")
    print("Per-string breakdown:")
    for s in sorted(per_string):
        r = per_string[s]
        print(f"  String {s}: {r['correct']}/{r['total']}  ({r['correct']/r['total']:.3f})")


if __name__ == "__main__":
    main()
