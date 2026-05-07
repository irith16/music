#!/usr/bin/env python3
"""
Train the two-stage classifier:
  1.  String classifier   (6-class Random Forest)
  2.  Per-string fret classifiers  (one RF per string)

Saves trained models to models/.
"""

import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from config import (
    OUT_DIR, MODELS_DIR,
    TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, MIN_SAMPLES_LEAF,
)


def load_data():
    X = np.load(OUT_DIR / "embeddings.npy")
    labels = np.load(OUT_DIR / "labels_pos.npy")

    y_string, y_fret = [], []
    for lab in labels:
        s, f = lab.split("_")
        y_string.append(int(s[1:]))
        y_fret.append(int(f[1:]))

    return X, np.array(y_string), np.array(y_fret)


def make_rf():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    X, y_string, y_fret = load_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} dims")
    print(f"Strings: {np.unique(y_string)},  Frets: {np.unique(y_fret)}\n")

    # ── String classifier ────────────────────────────────
    indices = np.arange(len(X))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y_string, indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_string,
    )

    string_clf = make_rf()
    string_clf.fit(X_tr, y_tr)

    y_pred = string_clf.predict(X_te)
    print("STRING CLASSIFIER")
    print(f"  Accuracy: {accuracy_score(y_te, y_pred):.3f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_te, y_pred)}\n")
    print(classification_report(y_te, y_pred))

    joblib.dump(string_clf, MODELS_DIR / "string_clf.pkl")

    # ── Per-string fret classifiers ──────────────────────
    fret_models = {}

    for s in sorted(np.unique(y_string)):
        idx = np.where(y_string == s)[0]
        X_s, y_f = X[idx], y_fret[idx]

        Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
            X_s, y_f,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_f,
        )

        clf = make_rf()
        clf.fit(Xf_tr, yf_tr)
        acc = accuracy_score(yf_te, clf.predict(Xf_te))
        print(f"  String {s} fret accuracy: {acc:.3f}")

        fret_models[s] = clf
        joblib.dump(clf, MODELS_DIR / f"fret_model_S{s}.pkl")

    # Save test indices for evaluate.py
    np.save(MODELS_DIR / "test_indices.npy", idx_te)

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
