"""Pitch-constrained string + fret prediction."""

import numpy as np

from config import OPEN_STRING_MIDI, MAX_FRET, PITCH_TOL_CENTS


def hz_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)


def cents_error(f_hz, midi_target):
    f_target = 440 * 2 ** ((midi_target - 69) / 12)
    return 1200 * np.log2(f_hz / f_target)


def pitch_valid_frets(f0_hz, string):
    """
    Return frets on *string* whose expected pitch is within
    PITCH_TOL_CENTS of the measured *f0_hz*.
    """
    if f0_hz <= 0:
        return []

    midi_open = OPEN_STRING_MIDI[string]
    candidates = []
    for fret in range(MAX_FRET + 1):
        if abs(cents_error(f0_hz, midi_open + fret)) <= PITCH_TOL_CENTS:
            candidates.append(fret)
    return candidates


def predict_string_fret(x, f0_hz, string_clf, fret_models, top_k_strings=3):
    """
    Two-stage prediction with pitch constraint.

    1. Rank strings by classifier probability.
    2. For each candidate string, restrict frets to those matching f0.
    3. Return (string, fret) that maximises  log P(string) + log P(fret).

    Returns (best_string, best_fret, debug_list).
    """
    string_probs = string_clf.predict_proba(x.reshape(1, -1))[0]
    strings = string_clf.classes_
    top_idx = np.argsort(string_probs)[::-1][:top_k_strings]

    best_score = -np.inf
    best_pair = (None, None)
    debug = []

    for s in strings[top_idx]:
        p_s = string_probs[np.where(strings == s)[0][0]]
        cand_frets = pitch_valid_frets(f0_hz, s) or list(range(MAX_FRET + 1))

        fret_clf = fret_models[s]
        fret_probs = fret_clf.predict_proba(x.reshape(1, -1))[0]
        fret_classes = fret_clf.classes_

        for f in cand_frets:
            if f not in fret_classes:
                continue
            p_f = fret_probs[np.where(fret_classes == f)[0][0]]
            score = np.log(p_s + 1e-9) + np.log(p_f + 1e-9)
            debug.append((s, f, score))
            if score > best_score:
                best_score = score
                best_pair = (s, f)

    return best_pair[0], best_pair[1], debug
