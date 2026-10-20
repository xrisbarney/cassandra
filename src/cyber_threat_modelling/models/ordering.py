"""Keep stable panel IDs separate from the loading-matrix anchor order."""

import numpy as np


def loading_order(initial_counts, category_ids, unknown=10):
    counts = np.asarray(initial_counts, dtype=float)
    ids = np.asarray(category_ids, dtype=int)
    if counts.ndim != 2 or counts.shape[1] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid initial panel dimensions or category IDs.")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Loading anchors require complete nonnegative initial counts.")
    totals = dict(zip(ids, counts.sum(axis=0), strict=True))
    ordered = sorted(ids, key=lambda k: (k == unknown, -totals[k], k))
    forward = np.array([int(np.flatnonzero(ids == k)[0]) for k in ordered])
    return forward, np.argsort(forward)
